from collections.abc import Callable
from copy import deepcopy
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve
from scipy.optimize import minimize

from .auxiliary import smoothness_matrix
from .core import Cube, Item, Set
from .errors import UnsupportedDimensionError
from .photospectral_objects import (
    PhotospectralCube,
    PhotospectralObject,
    PhotospectralSet,
    Photospectrum,
)
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum


class ReconstructedSpectralObject(SpectralObject):
    """
    Internal parent class for reconstructed spectral objects.
    The first index of the `spectral_dist` array iterates over the spectral axis.

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `photospectral_obj` (PhotospectralObject): optional, a way to store the pre-reconstructed data
    """

    photospectral_obj: PhotospectralObject | None = None  # type: ignore[assignment]

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        uncertainty: npt.ArrayLike | None = None,
        name: object = None,
        photospectral_obj: PhotospectralObject | None = None  # type: ignore[assignment]
    ) -> None:

        """
        Creates a ReconstructedSpectralObject from arrays of wavelength, brightness and (optionally) uncertainty.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): array of "brightness" in energy density units (not a photon counter)
        - `uncertainty`: (ArrayLike): optional array of standard deviations or covariance matrix
        - `name` (object): human-readable identifier
        - `is_emission_spectrum` (bool): if `True`, creates an emission spectral object from the spectral lines
        - `photospectral_obj` (PhotospectralObject): optional, a way to store the pre-reconstructed data
        """
        super().__init__(wavelength_nm, spectral_dist, uncertainty, name)
        self.photospectral_obj = photospectral_obj

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray[np.integer]):
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        """
        if self.photospectral_obj is None:
            extrapolated = super()._determine_at_trusted_wavelengths(requested_wavelengths)
        else:
            # Repeating the spectral reconstruction on the new wavelength range
            extrapolated = self.photospectral_obj._determine_at_trusted_wavelengths(requested_wavelengths)  # type: ignore[union-attr]
        return extrapolated

    def _apply_scalar_operation(
        self,
        operand: npt.ArrayLike,
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating]],
        error_handling: Callable[[npt.ArrayLike, npt.ArrayLike | None, npt.ArrayLike, npt.ArrayLike | None], npt.NDArray[np.floating] | None]
    ) -> Self:
        """
        Returns a new object of the same class transformed according to the linear operator.
        Operand is assumed to be a number or an array along the spectral axis.
        Linearity is needed because values and uncertainty are handled uniformly.
        """
        output = super()._apply_scalar_operation(operand, value_handling, error_handling)
        if self.photospectral_obj is not None:
            output.photospectral_obj = self.photospectral_obj._apply_scalar_operation(  # type: ignore[union-attr]
                operand, value_handling, error_handling
            )
        return output


class ReconstructedSpectrum(ReconstructedSpectralObject, Item):
    photospectral_obj: Photospectrum | None = None  # type: ignore[assignment]

    """
    Class to work with a single reconstructed spectrum (1D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `photospectral_obj` (Photospectrum): optional, a way to store the pre-reconstructed data
    """


class ReconstructedSpectralSet(ReconstructedSpectralObject, Set):
    photospectral_obj: PhotospectralSet | None = None  # type: ignore[assignment]

    """
    Class to work with a line of continuous spectra (2D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `photospectral_obj` (PhotospectralSet): optional, a way to store the pre-reconstructed data
    - `size` (int): spatial axis length
    """


class ReconstructedSpectralCube(ReconstructedSpectralObject, Cube):
    photospectral_obj: PhotospectralCube | None = None  # type: ignore[assignment]

    """
    Class to work with an image of continuous spectra (3D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `photospectral_obj` (PhotospectralCube): optional, a way to store the pre-reconstructed data
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `size` (int): number of pixels
    """

    def flatten(self) -> 'ReconstructedSpectralSet':
        """ Returns a SpectralSet with linearized spatial axis """
        output = super().flatten()  # type: ignore[return-value]
        if self.photospectral_obj is not None:
            output.photospectral_obj = self.photospectral_obj.flatten()  # type: ignore[union-attr]
        return output  # TODO: fix it properly! # pyright: ignore[reportReturnType]


def spectral_reconstruction(
    photospectral_obj: PhotospectralObject,
    requested_wavelengths: npt.ArrayLike,
    spectral_reconstruction_mode: str = '',
    attach_photospectral_obj: bool = True
) -> SpectralObject | ReconstructedSpectralObject:
    """
    Reconstructs a SpectralObject from photospectral data on the wavelength array.

    Interpolation is not used because it is not a solution to the inverse ill-posed problem
    (i.e., looking at the spectrum through the filters does not give exactly the original photospectral_obj).

    The function uses the Tikhonov regularization method, with a combination of first-order
    and second-order differential operators for the Tikhonov matrix.
    That is, it tries to minimize height variations and curvature in the spectrum.

    Confidence bands for spectral sets and cubes are not computed by default.
    """
    br0 = photospectral_obj.spectral_dist
    filter_set = photospectral_obj.filter_set.determine_at_wavelengths(requested_wavelengths, strictly=False)
    nm1 = filter_set.wavelength_nm
    if photospectral_obj.ignore_uncertainty_forCubes and photospectral_obj.ndim == 3:
        cov0 = None
    else:
        cov0 = photospectral_obj.covariance_matrix
    cov1: npt.NDArray[np.floating] | None = None
    if len(filter_set) == 1:
        # single-point PhotospectralObject support
        br1 = np.full((nm1.size, 1, 1)[:photospectral_obj.ndim], br0) # not tested
    else:
        filter_matrix = filter_set.matrix
        order1_matrix = smoothness_matrix(filter_matrix.shape[1], order=1, step=filter_set.nm_step)
        order2_matrix = smoothness_matrix(filter_matrix.shape[1], order=2, step=filter_set.nm_step)
        # TODO: research on some known spectra to find which alpha and beta fit best
        alpha = 1/8
        beta = 5000
        tikhonov_matrix = alpha * (order1_matrix.T @ order1_matrix + beta * order2_matrix.T @ order2_matrix)
        right_matrix = filter_matrix.T @ filter_matrix + tikhonov_matrix
        if photospectral_obj.ndim == 3:
            # scipy supports batch mode for 2d arrays, but not for 3D arrays
            br0 = br0.reshape(filter_matrix.shape[0], -1)
        left_vector = filter_matrix.T @ br0
        br1 = solve(right_matrix, left_vector) # x1.5 faster than np.linalg.inv(A) @ b
        if photospectral_obj.ndim == 3:
            # Reshape spectral cube back from square
            br1 = br1.reshape(-1, *photospectral_obj.spatial_shape)
        if photospectral_obj.ndim == 1 and br1.min() < 0:
            # To avoid negative spectra, a lower bound is set and iterative
            # optimization is performed using quadratic programming methods.
            # The processing speed drops by a factor of about five,
            # so the use is blocked for spectral squares and cubes:
            # background noise near zero can be most of the pixels.
            # TODO: RECHECK! MAY CONTAIN ERRORS!
            def objective(vector):
                # Tikhonov-regularized quadratic objective: 0.5 * Y^T A Y - b^T Y
                return 0.5 * vector @ right_matrix @ vector - left_vector @ vector
            def gradient(vector):
                # Gradient of the objective
                return right_matrix @ vector - left_vector
            bounds = ((0, None) for _ in range(right_matrix.shape[1]))
            result = minimize(
                fun=objective,
                x0=np.maximum(br1, 0),
                jac=gradient,
                bounds=bounds,
                method='L-BFGS-B',
            )
            if not result.success:
                raise ValueError(f'Optimization failed: {result.message}')
            br1 = result.x
        if photospectral_obj.ndim == 1 and cov0 is not None:
            # Measurement confidence band calculation
            # Confidence bands for spectral squares and cubes are not computed to save computational resources
            right_matrix_inv = np.linalg.inv(right_matrix)
            cov1 = right_matrix_inv @ filter_matrix.T @ cov0 @ filter_matrix @ right_matrix_inv
            #cov1 = filter_matrix.T @ cov0 @ filter_matrix + tikhonov_matrix # doesn't work
    if attach_photospectral_obj:
        match photospectral_obj:
            # An implementation suitable for type checking
            case Photospectrum():
                spectral_obj = ReconstructedSpectrum(
                    nm1, br1, cov1,
                    name=photospectral_obj.name,
                    photospectral_obj=deepcopy(photospectral_obj)
                )
            case PhotospectralSet():
                spectral_obj = ReconstructedSpectralSet(
                    nm1, br1, cov1,
                    name=photospectral_obj.name,
                    photospectral_obj=deepcopy(photospectral_obj)
                )
            case PhotospectralCube():
                spectral_obj = ReconstructedSpectralCube(
                    nm1, br1, cov1,
                    name=photospectral_obj.name,
                    photospectral_obj=deepcopy(photospectral_obj)
                )
            case _:
                raise ValueError(f'For {photospectral_obj.name} to be reconstructed, it must be of class Photospectrum, PhotospectralSet or PhotospectralCube')
    else:
        try:
            target_class = (Spectrum, SpectralSet, SpectralCube)[photospectral_obj.ndim - 1]
        except IndexError:
            raise UnsupportedDimensionError(photospectral_obj.ndim, photospectral_obj.name)
        spectral_obj = target_class(nm1, br1, cov1, name=photospectral_obj.name)
    return spectral_obj
