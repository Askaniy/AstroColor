from collections.abc import Sequence
from copy import deepcopy
from typing import Literal, cast, overload

import numpy as np
import numpy.typing as npt

from astrocolor.errors import (
    InconsistentAxesError,
    InconsistentUncertaintySizeError,
    UnsupportedDimensionError,
)

from .algebra import mul_value
from .auxiliary import integrate
from .filter_objects import Filter, FilterSet
from .photospectral_objects import (
    PhotospectralCube,
    PhotospectralObject,
    PhotospectralSet,
    Photospectrum,
)
from .physical_models import PhysicalModel
from .reconstructed_objects import (
    ReconstructedSpectralCube,
    ReconstructedSpectralSet,
    ReconstructedSpectrum,
)
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum


# Only `attach_photospectral_obj=True` is specialized below. The `False`
# case falls back to `SpectralObject`: an already reconstructed input can
# preserve its reconstructed class when no source photometry is stored, and
# a separate required `Literal[False]` overload cannot follow the defaulted
# `strictly` parameter without changing the public signature.
@overload
def get_spectrometry(
    target: Filter,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> Filter:
    ...

@overload
def get_spectrometry(
    target: FilterSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> FilterSet:
    ...

@overload
def get_spectrometry(
    target: Spectrum | PhysicalModel,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> Spectrum:
    ...

@overload
def get_spectrometry(
    target: SpectralSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> SpectralSet:
    ...

@overload
def get_spectrometry(
    target: SpectralCube,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> SpectralCube:
    ...

@overload
def get_spectrometry(
    target: Photospectrum,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectrum:
    ...

@overload
def get_spectrometry(
    target: PhotospectralSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectralSet:
    ...

@overload
def get_spectrometry(
    target: PhotospectralCube,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectralCube:
    ...

@overload
def get_spectrometry(
    target: ReconstructedSpectrum,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectrum:
    ...

@overload
def get_spectrometry(
    target: ReconstructedSpectralSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectralSet:
    ...

@overload
def get_spectrometry(
    target: ReconstructedSpectralCube,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectralCube:
    ...

@overload
def get_spectrometry(
    target: PhotospectralObject,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: Literal[True] = True
) -> ReconstructedSpectrum | ReconstructedSpectralSet | ReconstructedSpectralCube:
    ...

@overload
def get_spectrometry(
    target: PhotospectralObject,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> SpectralObject:
    ...

@overload
def get_spectrometry(
    target: SpectralObject,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> SpectralObject:
    ...

def get_spectrometry(
    target: SpectralObject | PhotospectralObject | PhysicalModel,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False,
    attach_photospectral_obj: bool = True
) -> SpectralObject | ReconstructedSpectrum | ReconstructedSpectralSet | ReconstructedSpectralCube:
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
    - attach_photospectral_obj: If True (default), photospectral inputs are
      returned as reconstructed objects that keep a reference to the source
      photometry. If False, they are returned as plain Spectrum/SpectralSet/
      SpectralCube. An already reconstructed object without stored source
      photometry keeps its reconstructed class. This controls only the
      returned object's type and metadata, not the reconstructed values.

    Returns:
    - A new SpectralObject with data determined at the specified wavelengths.

    Example:
    ```
    >>> spectrum = get_spectrometry(photospectrum, [400, 700])
    ```
    """
    nm_min, nm_max = target.get_extremal_grid_endpoints(requested_wavelengths)
    requested_wavelengths = target.uniform_grid(nm_min, nm_max)
    spectral_obj = target.determine_at_trusted_wavelengths(requested_wavelengths, attach_photospectral_obj=attach_photospectral_obj)
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
    # The data is already measured in these filters: do not lose it to a reconstruction round trip.
    if isinstance(target, PhotospectralObject) and isinstance(bandpass, FilterSet) and target.filter_set == bandpass:
        output = deepcopy(target)
        output.filter_set = bandpass
        return output
    target = get_spectrometry(target, bandpass.wavelength_nm, strictly=True)
    ndim = target.ndim
    sd = target.spectral_dist
    cov = target.covariance_matrix
    match bandpass:
        case Filter():
            value = cast(float, integrate(mul_value(sd, bandpass.spectral_dist), target.nm_step))
            error = None
            if cov is not None:
                # TODO: check!
                error = cast(
                    float | npt.NDArray[np.floating],
                    np.einsum('i...,ij...,j...->...', bandpass.spectral_dist, cov, bandpass.spectral_dist)
                    * target.nm_step**2
                )
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


def scale_to_match_value(
    spectral_data: Spectrum | Photospectrum,
    filter_or_nm: Filter | str | float,
    requested_value: float | tuple[float, float | None] = 1
) -> Spectrum | Photospectrum:
    """
    Returns a new object that matches the query brightness value (1 by default)
    at the specified filter or wavelength.
    """
    # Input data preparation
    bandpass = None
    match filter_or_nm:
        case Filter():
            bandpass = filter_or_nm
        case str():
            bandpass = Filter.get(filter_or_nm)
        case int() | float():
            bandpass = Filter.monochromatic(filter_or_nm)
    if not isinstance(bandpass, Filter):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError('The input cannot be used to set a filter object!')
    # Synthetic measurement
    current_value, _ = get_photometry(spectral_data, bandpass)
    if current_value <= 0:
        # Prevents errors of dividing by zero and inversion
        return spectral_data
    if isinstance(requested_value, Sequence):
        requested_value = requested_value[0] # likely a [value, std]
    # TODO: process std?
    return spectral_data * (requested_value / current_value)
