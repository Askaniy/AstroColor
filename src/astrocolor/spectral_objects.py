from collections.abc import Callable
from copy import deepcopy
from typing import Self, cast, override

import numpy as np
import numpy.typing as npt

from .auxiliary import (
    extrapolating,
    grid_endpoints_preprocessing,
    integrate,
    interpolate,
    spectral_binning,
    spectral_downscaling,
    stretch,
)
from .config import Config
from .core import (
    BaseObject,
    Cube,
    Item,
    Set,
    nm_red_limit,
    nm_step,
    spectral_dist_dtype,
    wavelength_nm_dtype,
)

# No dependency on .photospectral_objects to avoid cycle!
from .errors import (
    InconsistentAxesError,
    InconsistentDimensionError,
    InconsistentUncertaintyShapeError,
    InconsistentUncertaintySizeError,
    UnsupportedDimensionError,
    empty_spectral_intersection_operator_warning,
    empty_spectral_intersection_warning,
    erasing_correlations_warning,
    nan_values_warning,
    zero_brightness_warning,
)


class SpectralObject(BaseObject):
    """
    Internal parent class for Spectrum (1D), SpectralSet (2D) and SpectralCube (3D).
    The first index of the "brightness" array iterates over the spectral axis.

    Attributes:
    - `wavelength_nm` (NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `standard_deviation` (NDArray): optional array of standard deviations
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    """

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        uncertainty: npt.ArrayLike | None = None,
        name: object = None,
        is_emission_spectrum: bool = False
    ) -> None:
        """
        Creates a SpectralObject from arrays of wavelength, brightness and (optionally) uncertainty.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): array of "brightness" in energy density units (not a photon counter)
        - `uncertainty`: (ArrayLike): optional array of standard deviations or a covariance matrix
        - `name` (object): human-readable identifier
        - `is_emission_spectrum` (bool): if `True`, creates an emission spectral object from the spectral lines
        """
        self.name: object = name
        # Spatial axis check
        sd: npt.NDArray[np.floating] = np.array(spectral_dist, dtype=spectral_dist_dtype)
        if self.ndim != sd.ndim:
            raise InconsistentDimensionError(self.ndim, sd.ndim, name)
        if np.any(np.isnan(sd)):
            sd = np.nan_to_num(sd)
            nan_values_warning('spectral_dist', name)
        # Spectral axis check
        nm: npt.NDArray[np.integer | np.floating] = np.array(wavelength_nm) # numpy decides int or float
        if (len_nm := nm.size) != (len_values := len(sd)):
            raise InconsistentAxesError(len_nm, len_values, name)
        # Uncertainty check
        if Config.ignore_uncertainty_for_cubes and self.ndim == 3:
            uncertainty = None
        if uncertainty is not None:
            uncertainty = np.array(uncertainty, dtype=spectral_dist_dtype)
        uncertainty = cast(npt.NDArray[np.floating] | None, uncertainty)
        if uncertainty is not None and (len_error := len(uncertainty)) != len_values:
            raise InconsistentUncertaintySizeError(len_error, len_values, name)
        is_cov_matrix = None # flag to switch between standard deviation and covariance matrix
        if uncertainty is not None:
            if uncertainty.ndim == sd.ndim:
                is_cov_matrix = False
            elif uncertainty.ndim == sd.ndim + 1:
                is_cov_matrix = True
            else:
                raise InconsistentUncertaintyShapeError(uncertainty.ndim, sd.ndim, name)
        # Fast increasing check
        if np.any(nm[:-1] > nm[1:]):
            order = np.argsort(nm)
            nm = nm[order]
            sd = sd[order]
            if uncertainty is not None:
                uncertainty = uncertainty[order]
        # Red limit check
        if nm[-1] > nm_red_limit:
            mask = np.where(nm < nm_red_limit + nm_step) # with reserve to be averaged
            nm = nm[mask]
            sd = sd[mask]
            if uncertainty is not None:
                uncertainty = uncertainty[mask]
        # Assign type of the edges
        nm_0 = cast(float, nm[0])
        sd_0 = cast(float, sd[0])
        std_0 = None if uncertainty is None else cast(float, uncertainty[0])
        nm_1 = cast(float, nm[-1])
        sd_1 = cast(float, sd[-1])
        std_1 = None if uncertainty is None else cast(float, uncertainty[-1])
        if is_emission_spectrum:
            # Input in the sum of spectral lines
            if uncertainty is not None and is_cov_matrix:
                erasing_correlations_warning(name)
                uncertainty = np.sqrt(uncertainty.diagonal())
                is_cov_matrix = False
            # The first spectral line
            spectral_lines_sum = self.monochromatic(nm_0, sd_0, std_0)
            if nm.size > 1:
                # The last spectral line
                spectral_line = self.monochromatic(nm_1, sd_1, std_1)
                nm_range = (spectral_lines_sum.wavelength_nm[0], spectral_line.wavelength_nm[-1])
                nm_min, nm_max = spectral_line.get_extremal_grid_endpoints(nm_range)
                nm_uniform = spectral_line.uniform_grid(nm_min, nm_max)
                spectral_lines_sum = spectral_lines_sum.determine_at_trusted_wavelengths(nm_uniform)
                spectral_lines_sum += spectral_line.determine_at_trusted_wavelengths(nm_uniform)
                if nm.size > 2:
                    # Adding the remaining spectral lines to the overall wavelength range
                    # Reason for manually loading the boundary lines:
                    # to ensure that boundary zero values are processed correctly
                    for i in range(nm.size)[1:-1]:
                        nm_i = cast(float, nm[i])
                        sd_i = cast(float, sd[i])
                        std_i = None if uncertainty is None else cast(float, uncertainty[i])
                        spectral_line = self.monochromatic(nm_i, sd_i, std_i)
                        spectral_lines_sum += spectral_line.determine_at_trusted_wavelengths(nm_uniform)
            self.wavelength_nm: npt.NDArray[np.integer] = spectral_lines_sum.wavelength_nm
            self.spectral_dist: npt.NDArray[np.floating] = spectral_lines_sum.spectral_dist
            self.covariance_matrix: npt.NDArray[np.floating] | None = spectral_lines_sum.covariance_matrix
        else:
            # Spectral grid check to be a uniform 5 nm grid
            diff = np.diff(nm)
            if np.all(cast(npt.NDArray[np.bool], diff == nm_step)) and nm_0 % nm_step == 0:
                nm_uniform = cast(npt.NDArray[np.integer], nm)
            else:
                if uncertainty is not None and is_cov_matrix:
                    erasing_correlations_warning(name)
                    uncertainty = cast(npt.NDArray[np.floating], np.sqrt(uncertainty.diagonal()))
                    is_cov_matrix = False
                nm_uniform = self.uniform_grid(nm_0, nm_1)
                if diff.mean() >= self.nm_step:
                    # Option 1: loose spectral grid, increasing resolution
                    sd = interpolate(nm, sd, nm_uniform, nm_step)
                    if uncertainty is not None:
                        uncertainty = interpolate(nm, uncertainty, nm_uniform, nm_step)
                elif nm[-1] - nm[0] < 2 * nm_step:
                    # Option 2: a very narrow spectrum
                    # TODO: check for spectral sets and cubes
                    template = self.monochromatic(cast(float, np.average(nm, weights=sd)))
                    nm_uniform = template.wavelength_nm
                    integral = cast(npt.NDArray[np.floating], np.sum(0.5 * (sd[:-1] + sd[1:]) * diff, axis=0)) # Riemann sum
                    sd = cast(npt.NDArray[np.floating], template.spectral_dist * integral)
                    if uncertainty is not None:
                        # Problem 4
                        cov_scale = cast(npt.NDArray[np.floating], template.covariance_matrix)
                        cov_scale *= cast(npt.NDArray[np.floating], np.sum(0.5 * (uncertainty[:-1] + uncertainty[1:]) * diff, axis=0))**2
                        template.covariance_matrix = cov_scale
                elif diff.max() < nm_step:
                    # Option 3: dense spectral grid -> flux-conserving binning cumulative-integral (CDF) method
                    sd, uncertainty = spectral_binning(nm, sd, uncertainty, nm_uniform, nm_step, diff)
                else:
                    # Option 4: dense spectral grid with gaps -> convolution with variable core
                    sd, uncertainty = spectral_downscaling(nm, sd, uncertainty, nm_uniform, nm_step)
            self.wavelength_nm = nm_uniform
            self.spectral_dist = sd
            if uncertainty is None:
                self.covariance_matrix = None
            else:
                if is_cov_matrix:
                    self.covariance_matrix = uncertainty
                else:
                    self.covariance_matrix = np.diag(uncertainty**2)

        # Negative brightness values check
        # Not used because color matching functions may have negative values
        # if self.br.min() < 0:
        #    self.spectral_dist = np.clip(self.spectral_dist, 0, None)

    @override
    @classmethod
    def stub(cls, name: object = None) -> Self:
        """ Initializes an object in case of the data problems """
        return cls((555,), np.zeros((1,) * cls.ndim), name=name)

    @classmethod
    def monochromatic(
        cls,
        wavelength: float,
        intensity: float | npt.ArrayLike = 1.,
        standard_deviation: float | npt.ArrayLike | None = None
    ) -> Self:
        """
        Creates a monochromatic SpectralObject, the integral of which matches the intensity.
        By default, the integral intensity of a spectral line is 1.

        The input intensity must be one dimension lower than the class dimension
        (float for Spectrum, 1D for SpectralSet, 2D for SpectralCube).
        If the intensity value is a float, the result will be uniform along the spatial axes.

        Make sure you use the rectangle method for integration,
        otherwise the intensity would not conserve.
        """
        name = f'{wavelength} nm'
        # Spatial axis check
        sd0: npt.NDArray[np.floating] = np.array(intensity, dtype=spectral_dist_dtype)
        if sd0.ndim == 0:
            if cls.ndim == 2:
                sd0 = np.expand_dims(sd0, axis=0)
            elif cls.ndim == 3:
                sd0 = np.expand_dims(sd0, axis=(0, 1))
        if cls.ndim - 1 != sd0.ndim:
            raise InconsistentDimensionError(cls.ndim - 1, sd0.ndim, name)
        if np.any(np.isnan(sd0)):
            sd0 = np.nan_to_num(sd0)
            nan_values_warning('intensity', name)
        # Uncertainty check
        std0 = None
        if not (Config.ignore_uncertainty_for_cubes and cls.ndim == 3) and standard_deviation is not None:
            std0 = np.array(standard_deviation, dtype=spectral_dist_dtype)
            if (len_error := len(std0)) != (len_values := len(sd0)):
                raise InconsistentUncertaintySizeError(len_error, len_values, name)
        # Calculating position of the spectral line
        nm_point = wavelength / cls.nm_step
        nm_point_int = int(nm_point)
        nm_ref = nm_point_int * cls.nm_step
        # Creating spectral distribution
        if nm_point == nm_point_int:
            nm1 = (nm_ref - cls.nm_step, nm_ref, nm_ref + cls.nm_step)
            sd1 = (0., 1., 0.)
        else:
            proximity_factor = nm_point - nm_point_int
            nm1 = (nm_ref - cls.nm_step, nm_ref, nm_ref + cls.nm_step, nm_ref + 2 * cls.nm_step)
            sd1 = (0., 1.-proximity_factor, proximity_factor, 0.)
        # Scaling spectral distribution
        match cls.ndim:
            case 1:
                sd1 = np.array(sd1)
            case 2:
                sd1 = np.expand_dims(sd1, axis=1)
            case 3:
                sd1 = np.expand_dims(sd1, axis=(1, 2))
            case _:
                raise UnsupportedDimensionError(cls.ndim, name)
        std1 = None
        if std0 is not None:
            std1 = sd1 * sd0 / nm_step
        sd1 = sd1 * sd0 / nm_step
        return cls(nm1, sd1, std1, name=name)

    def integrate(self) -> float | npt.NDArray[np.floating]:
        """
        Integrates the SpectralObject along the spectral axis.
        Uses the rectangle method to match with matrix multiplication used for the spectral reconstruction.
        """
        return integrate(self.spectral_dist, nm_step, precisely=False)

    def normalized(self) -> Self:
        """ Returns a new SpectralObject with each spectrum divided by its area """
        return self / self.integrate()

    def normalize(self) -> None:
        """ Divides SpectralObject by its integral """
        result = self.normalized()
        self.spectral_dist = result.spectral_dist

    @override
    def convert_from_photon_spectral_density(self) -> Self:
        """
        Returns a new SpectralObject converted from photon spectral density
        to energy spectral density, using the fact that E = h c / λ.
        """
        return (self / self.wavelength_nm).normalized()

    @override
    def convert_from_energy_spectral_density_per_frequency(self) -> Self:
        """
        Returns a new SpectralObject converted from energy spectral density per frequency
        to energy spectral density per wavelength, using the fact that f_λ = f_ν c / λ².
        """
        scale_factors = 1 / self.wavelength_nm**2
        return (self / scale_factors).normalized()

    def mean_spectrum(self) -> 'Spectrum':
        """ Returns the mean spectrum along the spatial axes """
        # TODO: add cov matrix
        match self.ndim:
            case 1:
                sd = self.spectral_dist
            case 2:
                sd = cast(npt.NDArray[np.floating], np.mean(self.spectral_dist, axis=1))
            case 3:
                sd = cast(npt.NDArray[np.floating], np.mean(self.spectral_dist, axis=(1, 2)))
            case _:
                raise UnsupportedDimensionError(self.ndim, name=self.name)
        return Spectrum(self.wavelength_nm, sd, name=self.name)

    def median_spectrum(self) -> 'Spectrum':
        """ Returns the median spectrum along the spatial axes """
        match self.ndim:
            case 1:
                sd = self.spectral_dist
            case 2:
                sd = np.median(self.spectral_dist, axis=1)
            case 3:
                sd = np.median(self.spectral_dist, axis=(1, 2))
            case _:
                raise UnsupportedDimensionError(self.ndim, name=self.name)
        return Spectrum(self.wavelength_nm, sd, name=self.name)

    def mean_nm(self) -> float | npt.NDArray[np.floating]:
        """
        Returns the weighted average wavelength for each element of spatial axis:
        float value for a Spectrum, arrays for SpectralSet and SpectralCube.
        """
        try:
            return np.average(stretch(self.wavelength_nm, self.spatial_shape), weights=self.spectral_dist, axis=0)
        except ZeroDivisionError:
            zero_brightness_warning(self.name)
            return np.nan

    def std_of_nm(self) -> float | npt.NDArray[np.floating]:
        """ Returns uncorrected standard deviation or an array of uncorrected standard deviations """
        return np.sqrt(np.average((stretch(self.wavelength_nm, self.spatial_shape) - self.mean_nm())**2, weights=self.spectral_dist, axis=0))

    def get_spectral_dist_at_wavelengths(
        self,
        start: float,
        end: float
    ) -> npt.NDArray[np.floating]:
        """ Returns standard deviation values over a range of wavelengths (endpoints included) """
        start, end = grid_endpoints_preprocessing(start, end, nm_step)
        mask = (self.wavelength_nm >= start) & (self.wavelength_nm < end)
        intersection = self.spectral_dist[mask]
        if len(intersection) == 0:
            nm_0 = cast(int, self.wavelength_nm[0])
            nm_1 = cast(int, self.wavelength_nm[-1])
            empty_spectral_intersection_warning(nm_0, nm_1, start, end)
        return intersection

    def get_covariance_matrix_at_wavelengths(
        self,
        start: float,
        end: float
    ) -> npt.NDArray[np.floating] | None:
        """ Returns standard deviation values over a range of wavelengths (endpoints included) """
        if self.covariance_matrix is None:
            return None
        else:
            start, end = grid_endpoints_preprocessing(start, end, nm_step)
            slice_indices = np.where((self.wavelength_nm >= start) & (self.wavelength_nm < end))[0]
            if len(slice_indices) == 0:
                nm_0 = cast(int, self.wavelength_nm[0])
                nm_1 = cast(int, self.wavelength_nm[-1])
                empty_spectral_intersection_warning(nm_0, nm_1, start, end)
            return self.covariance_matrix[np.ix_(slice_indices, slice_indices)]

    def determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray[np.integer]) -> Self:
        """
        Directly uses the provided wavelength grid to create a new object.
        See `get_spectrometry()` for the general case.
        """
        # Preparing standard deviation
        std = None
        if self.covariance_matrix is not None:
            erasing_correlations_warning(self.name)
            std = np.sqrt(self.covariance_matrix.diagonal())
        # Extrapolating
        nm, br, std = extrapolating(self.wavelength_nm, self.spectral_dist, std, requested_wavelengths, nm_step)
        # Creating new object
        obj = deepcopy(self)
        obj.wavelength_nm = nm
        obj.spectral_dist = br
        obj.covariance_matrix = np.diag(std**2) if std is not None else None
        return obj

    def edges_to_zero(self) -> None:
        """
        Ensures that the SpectralObject has zero brightness at the spectral edges.
        Recommended for use with filters: improves the integral and the profile graph.

        Works for 1D (Spectrum/Filter), 2D (SpectralSet/FilterSet) and 3D (SpectralCube) objects:
        - If an edge has non-zero values at *any* spatial pixel, a zero point is added.
        - If the first/last two spectral points are all zeros across every spatial pixel,
          those extra leading/trailing zeros are trimmed back to one.
        """
        is_stub = self.spectral_size == 1  # stub objects have a single spectral point
        # - Left edge (index 0 along spectral axis)
        sd_0 = cast(npt.NDArray[np.floating], self.spectral_dist[0])
        if sd_0.any():
            # No zero on the left edge at some spatial pixel -> prepend a zero point
            new_wl = np.array([self.wavelength_nm[0] - nm_step], dtype=wavelength_nm_dtype)
            self.wavelength_nm = np.concatenate((new_wl, self.wavelength_nm))
            zeros_left = np.zeros(self.spatial_shape, dtype=wavelength_nm_dtype)
            self.spectral_dist = np.concatenate((zeros_left[np.newaxis], self.spectral_dist), axis=0)
        elif not is_stub:
            sd_1 = cast(npt.NDArray[np.floating], self.spectral_dist[1])
            if not sd_1.any():
                # Consecutive zeros on the left -> trim back to exactly one
                idx = None
                for i in range(2, self.spectral_size):
                    sd_i = cast(npt.NDArray[np.floating], self.spectral_dist[i])
                    if sd_i.any():
                        idx = i - 1
                        break
                if idx is not None:
                    self.wavelength_nm = self.wavelength_nm[idx:]
                    self.spectral_dist = self.spectral_dist[idx:]
        # else: exactly one zero on the left — nothing to do
        # - Right edge (index -1 along spectral axis)
        sd_1 = cast(npt.NDArray[np.floating], self.spectral_dist[-1])
        if sd_1.any():
            # No zero on the right edge at some spatial pixel -> append a zero point
            new_wl = np.array([self.wavelength_nm[-1] + nm_step], dtype=wavelength_nm_dtype)
            self.wavelength_nm = np.concatenate((self.wavelength_nm, new_wl))
            zeros_right = np.zeros(self.spatial_shape, dtype=wavelength_nm_dtype)
            self.spectral_dist = np.concatenate((self.spectral_dist, zeros_right[np.newaxis]), axis=0)
        elif not is_stub:
            sd_2 = cast(npt.NDArray[np.floating], self.spectral_dist[-2])
            if not sd_2.any():
                # Consecutive zeros on the right -> trim back to exactly one
                idx = None
                for i in range(-3, -self.spectral_size - 1, -1):
                    sd_i = cast(npt.NDArray[np.floating], self.spectral_dist[i])
                    if sd_i.any():
                        idx = i + 2
                        break
                if idx is not None:
                    self.wavelength_nm = self.wavelength_nm[:idx]
                    self.spectral_dist = self.spectral_dist[:idx]
        # else: exactly one zero on the right -> nothing to do

    def is_zero_edged(self) -> bool:
        """ Checks that the first and last brightness entries on the spectral axis are zero """
        sd_0 = cast(npt.NDArray[np.floating], self.spectral_dist[0])
        sd_1 = cast(npt.NDArray[np.floating], self.spectral_dist[-1])
        return not (sd_0.any() or sd_1.any())

    @override
    def _apply_element_wise_operation(
        self,
        other: 'BaseObject',
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating]],
        error_handling: Callable[[npt.ArrayLike, npt.ArrayLike | None, npt.ArrayLike, npt.ArrayLike | None], npt.NDArray[np.floating] | None]
    ) -> 'SpectralObject':
        """
        Returns a new SpectralObject formed from element-wise operation between SpectralObjects
        of the same nature or with a Spectrum.

        Only works at the intersection of the spectral axes! If you need to extrapolate one axis
        to the range of another, use the `get_spectrometry()` method.
        """
        if isinstance(other, SpectralObject):
            higher_dim = (self, other)[self.ndim < other.ndim]
            self_nm_0 = cast(int, self.wavelength_nm[0])
            self_nm_1 = cast(int, self.wavelength_nm[-1])
            other_nm_0 = cast(int, other.wavelength_nm[0])
            other_nm_1 = cast(int, other.wavelength_nm[-1])
            start = max(self_nm_0, other_nm_0)
            end = min(self_nm_1, other_nm_1)
            if start > end: # `>` is needed to process operations with stub objects with no extra logs
                the_first = other.name
                the_second = other.name
                if self.wavelength_nm[0] > other.wavelength_nm[0]:
                    the_first, the_second = the_second, the_first
                empty_spectral_intersection_operator_warning(value_handling.__name__, start, end, the_first, the_second)
                return higher_dim.__class__.stub(self.name)
            else:
                value1 = self.get_spectral_dist_at_wavelengths(start, end)
                value2 = other.get_spectral_dist_at_wavelengths(start, end)
                value = value_handling(value1, value2)
                error = error_handling(value1, self.get_covariance_matrix_at_wavelengths(start, end), value2, other.get_covariance_matrix_at_wavelengths(start, end))
                return higher_dim.__class__(self.uniform_grid(start, end), value, error, name=higher_dim.name)
        else:
            return NotImplemented


class Spectrum(SpectralObject, Item['SpectralSet']):
    """
    Class to work with a single spectrum (1D SpectralObject).

    Attributes:
    - `wavelength_nm` (NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    """


class SpectralSet(SpectralObject, Set['Spectrum', 'SpectralCube']):
    """
    Class to work with a line of continuous spectra (2D SpectralObject).

    Attributes:
    - `wavelength_nm` (NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `size` (int): spatial axis length
    """


class SpectralCube(SpectralObject, Cube['SpectralSet']):
    """
    Class to work with an image of continuous spectra (3D SpectralObject).

    Attributes:
    - `wavelength_nm` (NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `size` (int): number of pixels
    """

    @override
    def flatten(self) -> SpectralSet:
        """ Returns a SpectralSet with linearized spatial axis """
        value = self.spectral_dist.reshape(self.spectral_size, self.spatial_size)
        error = None
        if self.covariance_matrix is not None:
            error = self.covariance_matrix.reshape(self.spectral_size, self.spectral_size, self.spatial_size)
        return SpectralSet(self.wavelength_nm, value, error, self.name)
