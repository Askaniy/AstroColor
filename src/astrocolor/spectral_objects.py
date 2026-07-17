import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from typing import cast, Self, Any
from copy import deepcopy

from .auxiliary import grid_endpoints_preprocessing, integrate, stretch, interpolate, extrapolating, spectral_binning, spectral_downscaling
from .core import BaseObject, Item, Set, Cube, nm_step, wavelength_nm_dtype, spectral_dist_dtype, nm_red_limit
# No dependency on .photospectral_objects to avoid cycle!
from .errors import UnsupportedDimensionError, InconsistentDimensionError, \
    InconsistentAxesError, InconsistentUncertaintySizeError, InconsistentUncertaintyShapeError, \
    erasing_correlations_warning, nan_values_warning, zero_brightness_warning, empty_spectral_intersection_operator_warning, \
    empty_spectral_intersection_warning


class SpectralObject(BaseObject):
    """
    Internal parent class for Spectrum (1D), SpectralSet (2D) and SpectralCube (3D).
    The first index of the "brightness" array iterates over the spectral axis.

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `standard_deviation` (npt.NDArray): optional array of standard deviations
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): human-readable identifier
    """

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        uncertainty: npt.ArrayLike | None = None,
        name: Any = None,
        is_emission_spectrum: bool = False
    ) -> None:
        """
        Creates a SpectralObject from arrays of wavelength, brightness and (optionally) uncertainty.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): array of "brightness" in energy density units (not a photon counter)
        - `uncertainty`: (ArrayLike): optional array of standard deviations or a covariance matrix
        - `name` (Any): human-readable identifier
        - `is_emission_spectrum` (bool): if `True`, creates an emission spectral object from the spectral lines
        """
        self.name = name
        # Spatial axis check
        spectral_dist = np.array(spectral_dist, dtype=spectral_dist_dtype)
        if self.ndim != spectral_dist.ndim:
            raise InconsistentDimensionError(self.ndim, spectral_dist.ndim, name)
        if np.any(np.isnan(spectral_dist)):
            spectral_dist = np.nan_to_num(spectral_dist)
            nan_values_warning('spectral_dist', name)
        # Spectral axis check
        wavelength_nm = np.array(wavelength_nm) # numpy decides int or float
        if (len_nm := wavelength_nm.size) != (len_values := len(spectral_dist)):
            raise InconsistentAxesError(len_nm, len_values, name)
        # Uncertainty check
        if self.ignore_uncertainty_forCubes and self.ndim == 3:
            uncertainty = None
        if uncertainty is not None:
            uncertainty = np.array(uncertainty, dtype=spectral_dist_dtype)
        if uncertainty is not None and (len_error := len(uncertainty)) != len_values:
            raise InconsistentUncertaintySizeError(len_error, len_values, name)
        is_cov_matrix = None # flag to switch between standard deviation and covariance matrix
        if uncertainty is not None:
            if uncertainty.ndim == spectral_dist.ndim:
                is_cov_matrix = False
            elif uncertainty.ndim == spectral_dist.ndim + 1:
                is_cov_matrix = True
            else:
                raise InconsistentUncertaintyShapeError(uncertainty.ndim, spectral_dist.ndim, name)
        # Fast increasing check
        if np.any(wavelength_nm[:-1] > wavelength_nm[1:]):
            order = np.argsort(wavelength_nm)
            wavelength_nm = wavelength_nm[order]
            spectral_dist = spectral_dist[order]
            if uncertainty is not None:
                uncertainty = uncertainty[order]
        # Red limit check
        if wavelength_nm[-1] > nm_red_limit:
            mask = np.where(wavelength_nm < nm_red_limit + nm_step) # with reserve to be averaged
            wavelength_nm = wavelength_nm[mask]
            spectral_dist = spectral_dist[mask]
            if uncertainty is not None:
                uncertainty = uncertainty[mask]
        if is_emission_spectrum:
            if uncertainty is not None and is_cov_matrix:
                erasing_correlations_warning(name)
                uncertainty = np.sqrt(np.diag(uncertainty, axis=0))
                is_cov_matrix = False
            # The first spectral line
            spectral_lines_sum = self.monochromatic(wavelength_nm[0], spectral_dist[0], None if uncertainty is None else uncertainty[0])
            if wavelength_nm.size > 1:
                # The last spectral line
                spectral_line = self.monochromatic(wavelength_nm[-1], spectral_dist[-1], None if uncertainty is None else uncertainty[-1])
                nm_range = (spectral_lines_sum.wavelength_nm[0], spectral_line.wavelength_nm[-1])
                spectral_lines_sum.determine_at_wavelengths(nm_range)
                spectral_line.determine_at_wavelengths(nm_range)
                spectral_lines_sum += spectral_line
                if wavelength_nm.size > 2:
                    # Adding the remaining spectral lines to the overall wavelength range
                    # Reason for manually loading the boundary lines:
                    # to ensure that boundary zero values are processed correctly
                    for i in range(wavelength_nm.size)[1:-1]:
                        spectral_line = self.monochromatic(wavelength_nm[i], spectral_dist[i], None if uncertainty is None else uncertainty[i])
                        spectral_lines_sum += spectral_line.determine_at_wavelengths(nm_range)
            self.wavelength_nm = spectral_lines_sum.wavelength_nm
            self.spectral_dist = spectral_lines_sum.spectral_dist
            self.covariance_matrix = spectral_lines_sum.covariance_matrix
        else:
            # Spectral grid check to be a uniform 5 nm grid
            if np.any((diff := np.diff(wavelength_nm)) != nm_step) or wavelength_nm[0] % nm_step != 0:
                if uncertainty is not None and is_cov_matrix:
                    erasing_correlations_warning(name)
                    uncertainty = np.sqrt(np.diag(uncertainty, axis=0))
                    is_cov_matrix = False
                nm_uniform = self._uniform_grid(wavelength_nm[0], wavelength_nm[-1])
                if diff.mean() >= self.nm_step:
                    # Option 1: loose spectral grid, increasing resolution
                    spectral_dist = interpolate(wavelength_nm, spectral_dist, nm_uniform, nm_step)
                    if uncertainty is not None:
                        uncertainty = interpolate(wavelength_nm, uncertainty, nm_uniform, nm_step)
                elif wavelength_nm[-1] - wavelength_nm[0] < 2 * nm_step:
                    # Option 2: a very narrow spectrum
                    template = self.monochromatic(np.average(wavelength_nm, weights=spectral_dist))
                    nm_uniform = template.wavelength_nm
                    integral = np.sum(0.5 * (spectral_dist[:-1] + spectral_dist[1:]) * diff, axis=0) # Riemann sum
                    spectral_dist = template.spectral_dist * integral
                    if uncertainty is not None:
                        # Problem 4
                        template.covariance_matrix *= np.sum(0.5 * (uncertainty[:-1] + uncertainty[1:]) * diff, axis=0)**2
                elif diff.max() < nm_step:
                    # Option 3: dense spectral grid -> flux-conserving binning cumulative-integral (CDF) method
                    spectral_dist, uncertainty = spectral_binning(wavelength_nm, spectral_dist, uncertainty, nm_uniform, nm_step, diff)
                else:
                    # Option 4: dense spectral grid with gaps -> convolution with variable core
                    spectral_dist, uncertainty = spectral_downscaling(wavelength_nm, spectral_dist, uncertainty, nm_uniform, nm_step)
                wavelength_nm = nm_uniform
            self.wavelength_nm = wavelength_nm
            self.spectral_dist = spectral_dist
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

    @classmethod
    def stub(cls, name=None) -> Self:
        """ Initializes an object in case of the data problems """
        return cls((555,), np.zeros((1,) * cls.ndim), name=name)

    @classmethod
    def monochromatic(
        cls,
        wavelength: int | float,
        intensity: int | float = 1,
        standard_deviation: int | float | None = None
    ):
        """
        Creates a monochromatic SpectralObject on the 1- or 2-point spectral grid.
        It is normaized by default and have zeroed edges.
        Make sure you use the rectangle method for integration, otherwise the intensity would not converve.
        """
        name = f'{wavelength} nm'
        nm_point = wavelength / cls.nm_step
        nm_point_int = int(nm_point)
        nm0 = nm_point_int * cls.nm_step
        cov_matrix = None
        if nm_point == nm_point_int:
            nm = (nm0 - cls.nm_step, nm0, nm0 + cls.nm_step)
            br = (0., intensity, 0.)
            if standard_deviation is not None:
                cov_matrix = np.zeros((3, 3))
                cov_matrix[1,1] = standard_deviation * standard_deviation
        else:
            proximity_factor = nm_point - nm_point_int
            nm = (nm0 - cls.nm_step, nm0, nm0 + cls.nm_step, nm0 + 2*cls.nm_step)
            br = (0., 1.-proximity_factor, proximity_factor, 0.)
            if standard_deviation is not None:
                cov_matrix = np.zeros((4, 4))
                # Problem 1
                cov_matrix[1,1] = standard_deviation * standard_deviation
        # Normalization
        br = np.array(br, dtype=spectral_dist_dtype) / nm_step
        # Expending spatial dimension if needed (not tested!)
        match cls.ndim:
            case 1:  # Single spectrum — no spatial expansion
                pass
            case 2:
                br = np.expand_dims(br, axis=1)
                if cov_matrix is not None:
                    cov_matrix = np.expand_dims(cov_matrix, axis=2)
            case 3:
                br = np.expand_dims(br, axis=(1, 2))
                if cov_matrix is not None:
                    cov_matrix = np.expand_dims(cov_matrix, axis=(2, 3))
            case _:
                raise UnsupportedDimensionError(cls.ndim, name)
        return cls(nm, br, cov_matrix, name=name)

    def integrate(self) -> float | npt.NDArray[np.floating]:
        """
        Integrates the SpectralObject along the spectral axis.
        Uses the rectangle method to match with matrix multiplication used for the spectral reconstruction.
        """
        return integrate(self.spectral_dist, nm_step, precisely=False)

    def normalize(self) -> Self:
        """ Returns a new SpectralObject with each spectrum divided by its area """
        return self / self.integrate()

    def convert_from_photon_spectral_density(self) -> Self:
        """
        Returns a new SpectralObject converted from photon spectral density
        to energy spectral density, using the fact that E = h c / λ.
        """
        return (self / self.wavelength_nm).normalize()

    def convert_from_energy_spectral_density_per_frequency(self) -> Self:
        """
        Returns a new SpectralObject converted from energy spectral density per frequency
        to energy spectral density per wavelength, using the fact that f_λ = f_ν c / λ².
        """
        scale_factors = 1 / self.wavelength_nm**2
        return (self / scale_factors).normalize()

    def mean_spectrum(self) -> 'Spectrum':
        """ Returns the mean spectrum along the spatial axes """
        # TODO: add cov matrix
        match self.ndim:
            case 1:
                br = self.spectral_dist
            case 2:
                br = np.mean(self.spectral_dist, axis=1)
            case 3:
                br = np.mean(self.spectral_dist, axis=(1, 2))
            case _:
                raise UnsupportedDimensionError(self.name)
        return Spectrum(self.wavelength_nm, br, name=self.name)

    def median_spectrum(self) -> 'Spectrum':
        """ Returns the median spectrum along the spatial axes """
        match self.ndim:
            case 1:
                br = self.spectral_dist
            case 2:
                br = np.median(self.spectral_dist, axis=1)
            case 3:
                br = np.median(self.spectral_dist, axis=(1, 2))
            case _:
                raise UnsupportedDimensionError(self.name)
        return Spectrum(self.wavelength_nm, br, name=self.name)

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
        start: int | float,
        end: int | float
    ) -> npt.NDArray[np.floating]:
        """ Returns standard deviation values over a range of wavelengths (endpoints included) """
        start, end = grid_endpoints_preprocessing(start, end, nm_step)
        mask = (self.wavelength_nm >= start) & (self.wavelength_nm < end)
        intersection = self.spectral_dist[mask]
        if len(intersection) == 0:
            empty_spectral_intersection_warning(self.wavelength_nm[0], self.wavelength_nm[-1], start, end)
        return intersection

    def get_covariance_matrix_at_wavelengths(
        self,
        start: int | float,
        end: int | float
    ) -> npt.NDArray[np.floating] | None:
        """ Returns standard deviation values over a range of wavelengths (endpoints included) """
        if self.covariance_matrix is None:
            return None
        else:
            start, end = grid_endpoints_preprocessing(start, end, nm_step)
            slice_indices = np.where((self.wavelength_nm >= start) & (self.wavelength_nm < end))[0]
            if len(slice_indices) == 0:
                empty_spectral_intersection_warning(self.wavelength_nm[0], self.wavelength_nm[-1], start, end)
            return self.covariance_matrix[np.ix_(slice_indices, slice_indices)]

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray) -> Self:
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        """
        # Preparing standard deviation
        std = None
        if self.covariance_matrix is not None:
            erasing_correlations_warning(self.name)
            std = np.sqrt(np.diag(self.covariance_matrix, axis=0))
        # Extrapolating
        nm, br, std = extrapolating(self.wavelength_nm, self.spectral_dist, std, requested_wavelengths, nm_step)
        # Creating new object
        obj = deepcopy(self)
        obj.wavelength_nm = nm
        obj.spectral_dist = br
        obj.covariance_matrix = np.diag(std**2) if std is not None else None
        return obj

    def edges_to_zero(self) -> Self:
        """
        Returns a new SpectralObject with zero brightness at the spectral edges.
        Recommended for use with filters: improves the integral and the profile graph.

        Works for 1D (Spectrum/Filter), 2D (SpectralSet/FilterSet) and 3D (SpectralCube) objects:
        - If an edge has non-zero values at *any* spatial pixel, a zero point is added.
        - If the first/last two spectral points are all zeros across every spatial pixel,
          those extra leading/trailing zeros are trimmed back to one.
        """
        obj = deepcopy(self)
        is_stub = obj.spectral_size == 1  # stub objects have a single spectral point
        # - Left edge (index 0 along spectral axis)
        if not np.all(obj.spectral_dist[0] == 0):
            # No zero on the left edge at some spatial pixel -> prepend a zero point
            new_wl = np.array([obj.wavelength_nm[0] - nm_step], dtype=wavelength_nm_dtype)
            obj.wavelength_nm = np.concatenate((new_wl, obj.wavelength_nm))
            zeros_left = np.zeros(obj.spatial_shape, dtype=wavelength_nm_dtype)
            obj.spectral_dist = np.concatenate((zeros_left[np.newaxis], obj.spectral_dist), axis=0)
        elif not is_stub and np.all(obj.spectral_dist[1] == 0):
            # Consecutive zeros on the left -> trim back to exactly one
            idx = None
            for i in range(2, obj.spectral_size):
                if not np.all(obj.spectral_dist[i] == 0):
                    idx = i - 1
                    break
            if idx is not None:
                obj.wavelength_nm = obj.wavelength_nm[idx:]
                obj.spectral_dist = obj.spectral_dist[idx:]
        # else: exactly one zero on the left — nothing to do
        # - Right edge (index -1 along spectral axis)
        if not np.all(obj.spectral_dist[-1] == 0):
            # No zero on the right edge at some spatial pixel -> append a zero point
            new_wl = np.array([obj.wavelength_nm[-1] + nm_step], dtype=wavelength_nm_dtype)
            obj.wavelength_nm = np.concatenate((obj.wavelength_nm, new_wl))
            zeros_right = np.zeros(obj.spatial_shape, dtype=wavelength_nm_dtype)
            obj.spectral_dist = np.concatenate((obj.spectral_dist, zeros_right[np.newaxis]), axis=0)
        elif not is_stub and np.all(obj.spectral_dist[-2] == 0):
            # Consecutive zeros on the right -> trim back to exactly one
            idx = None
            for i in range(-3, -obj.spectral_size - 1, -1):
                if not np.all(obj.spectral_dist[i] == 0):
                    idx = i + 2
                    break
            if idx is not None:
                obj.wavelength_nm = obj.wavelength_nm[:idx]
                obj.spectral_dist = obj.spectral_dist[:idx]
        # else: exactly one zero on the right -> nothing to do
        return obj

    def is_zero_edged(self) -> bool:
        """ Checks that the first and last brightness entries on the spectral axis are zero """
        return bool(np.all(self.spectral_dist[0] == 0) and np.all(self.spectral_dist[-1] == 0))

    def _apply_element_wise_operation(
        self,
        other: 'BaseObject',
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
        error_handling: Callable[[npt.ArrayLike, npt.NDArray | None, npt.ArrayLike, npt.NDArray | None], npt.NDArray | None]
    ) -> 'SpectralObject':
        """
        Returns a new SpectralObject formed from element-wise operation between SpectralObjects
        of the same nature or with a Spectrum.

        Only works at the intersection of the spectral axes! If you need to extrapolate one axis
        to the range of another, use the `determine_at_wavelengths()` method.
        """
        if isinstance(other, SpectralObject):
            higher_dim = (self, other)[self.ndim < other.ndim]
            start = max(self.wavelength_nm[0], other.wavelength_nm[0])
            end = min(self.wavelength_nm[-1], other.wavelength_nm[-1])
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
                return higher_dim.__class__(self._uniform_grid(start, end), value, error, name=higher_dim.name)
        else:
            return NotImplemented


class Spectrum(SpectralObject, Item):
    """
    Class to work with a single spectrum (1D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): human-readable identifier
    """


class SpectralSet(SpectralObject, Set):
    """
    Class to work with a line of continuous spectra (2D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): human-readable identifier
    - `size` (int): spatial axis length
    """
    pass


class SpectralCube(SpectralObject, Cube):
    """
    Class to work with an image of continuous spectra (3D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): human-readable identifier
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `size` (int): number of pixels
    """

    def flatten(self) -> 'SpectralSet':
        """ Returns a SpectralSet with linearized spatial axis """
        value = self.spectral_dist.reshape(self.spectral_size, self.spatial_size)
        error = None if self.covariance_matrix is None else self.covariance_matrix.reshape(self.spectral_size, self.spatial_size)
        return SpectralSet(self.wavelength_nm, value, cast(npt.NDArray | None, error), self.name)
