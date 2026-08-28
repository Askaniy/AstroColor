from collections.abc import Sequence
from typing import cast, overload

import numpy as np
import numpy.typing as npt

from astrocolor.errors import (
    InconsistentAxesError,
    InconsistentUncertaintySizeError,
    UnsupportedDimensionError,
)

from .algebra import mul_error, mul_value
from .auxiliary import integrate
from .filter_objects import Filter, FilterSet
from .photospectral_objects import (
    PhotospectralCube,
    PhotospectralObject,
    PhotospectralSet,
    Photospectrum,
)
from .physical_models import PhysicalModel
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum


@overload
def get_spectrometry(
    target: Filter,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> Filter:
    ...

@overload
def get_spectrometry(
    target: Spectrum | Photospectrum | PhysicalModel,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> Spectrum:
    ...

@overload
def get_spectrometry(
    target: FilterSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> FilterSet:
    ...

@overload
def get_spectrometry(
    target: SpectralSet | PhotospectralSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralSet:
    ...

@overload
def get_spectrometry(
    target: SpectralCube | PhotospectralCube,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralCube:
    ...

@overload
def get_spectrometry(
    target: SpectralObject | PhotospectralObject,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralObject:
    ...

def get_spectrometry(
    target: SpectralObject | PhotospectralObject | PhysicalModel,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralObject:
    """
    Returns a new SpectralObject, guaranteeing that the specified wavelength range
    has been determined or reconstructed for it.
    If `strictly=True`, then the new object is defined exclusively
    on the specified wavelength range.
    Only the minimum and maximum wavelengths are extracted from the specified range,
    based on which a uniform grid is constructed.

    Args:
    - requested_wavelengths: Wavelength values to determine at.
    - strictly: If True, clip the result to the exact requested range.

    Returns:
    - A new SpectralObject with data determined at the specified wavelengths.

    Example:
    ```
    >>> spectrum = get_spectrometry(photospectrum, [400, 700])
    ```
    """
    nm_min, nm_max = target.get_extremal_grid_endpoints(requested_wavelengths)
    requested_wavelengths = target.uniform_grid(nm_min, nm_max)
    spectral_obj = target.determine_at_trusted_wavelengths(requested_wavelengths)
    # Spectral range clipping
    if strictly and not np.array_equal(spectral_obj.wavelength_nm, requested_wavelengths):
        spectral_obj.spectral_dist = spectral_obj.get_spectral_dist_at_wavelengths(nm_min, nm_max)
        spectral_obj.covariance_matrix = spectral_obj.get_covariance_matrix_at_wavelengths(nm_min, nm_max)
        spectral_obj.wavelength_nm = requested_wavelengths
    # Sanity checks
    if (len_nm := spectral_obj.wavelength_nm.size) != (len_values := len(spectral_obj.spectral_dist)):
        raise InconsistentAxesError(len_nm, len_values, spectral_obj.name)
    if spectral_obj.covariance_matrix is not None and (len_error := len(spectral_obj.covariance_matrix)) != len_nm:
        raise InconsistentUncertaintySizeError(len_error, len_values, spectral_obj.name)
    return spectral_obj


@overload
def get_photometry(
    target: Spectrum | Photospectrum | SpectralSet | PhotospectralSet | SpectralCube | PhotospectralCube | SpectralObject | PhotospectralObject,
    bandpass: Filter
) -> tuple[float, float | None]:
    ...

@overload
def get_photometry(
    target: Spectrum | Photospectrum,
    bandpass: FilterSet
) -> Photospectrum:
    ...

@overload
def get_photometry(
    target: SpectralSet | PhotospectralSet,
    bandpass: FilterSet
) -> PhotospectralSet:
    ...

@overload
def get_photometry(
    target: SpectralCube | PhotospectralCube,
    bandpass: FilterSet
) -> PhotospectralCube:
    ...

@overload
def get_photometry(
    target: SpectralObject | PhotospectralObject,
    bandpass: FilterSet
) -> Photospectrum | PhotospectralSet | PhotospectralCube:
    ...

def get_photometry(
    target: SpectralObject | PhotospectralObject,
    bandpass: Filter | FilterSet
) -> tuple[float, float | None] | PhotospectralObject:
    """
    Implementation of convolution between a (photo)spectral object and a filter or a filter set.
    Ignores the uncertainty of filter profiles.
    """
    target = get_spectrometry(target, bandpass.wavelength_nm, strictly=True)
    ndim = target.ndim
    sd = target.spectral_dist
    cov = target.covariance_matrix
    match bandpass:
        case Filter():
            value = cast(float, integrate(mul_value(sd, bandpass.spectral_dist), target.nm_step))
            error = mul_error(sd, cov, bandpass.spectral_dist, None)
            if error is not None:
                error = cast(float, integrate(error, target.nm_step))
            return value, error
        case FilterSet():
            value = cast(npt.NDArray[np.floating], np.einsum('ij, j... -> i...', bandpass.matrix, sd))
            # compare! 1D value = integrate(mul_value(sd, bandpass.spectral_dist), nm_step)
            # compare! 2D value = integrate(sd[:, :, np.newaxis] * bandpass.spectral_dist[:, np.newaxis, :], nm_step).T
            # 3D: value = np.empty((len(bandpass), *target.spatial_shape))
                # for i in range(len(bandpass)):
                #     profile = bandpass.spectral_dist[:,i]
                #     br[i] = integrate((sd.T * profile).T, nm_step)
            error = None
            if cov is not None:
                error = cast(npt.NDArray[np.floating], np.einsum('ij, jk..., lk -> il...', bandpass.matrix, cov, bandpass.matrix))
            match ndim:
                case 1:
                    return Photospectrum(bandpass, value, error, name=target.name)
                case 2:
                    return PhotospectralSet(bandpass, value, error, name=target.name)
                case 3:
                    return PhotospectralCube(bandpass, value, error, name=target.name)
                case _:
                    raise UnsupportedDimensionError(ndim, name=target.name)


def scale_spectrum(
    target: Spectrum,
    bandpass: Filter,
    requested_value: float | tuple[float, float] = 1
) -> Spectrum:
    """
    Returns a new spectrum that matches the query brightness value (1 by default)
    at the specified filter.
    """
    current_value, _ = get_photometry(target, bandpass)
    if current_value <= 0:
        # Prevents errors of dividing by zero and inversion
        return target
    if isinstance(requested_value, Sequence):
        requested_value = requested_value[0] # likely a [value, std]
    # TODO: process std?
    return target * (requested_value / current_value)
