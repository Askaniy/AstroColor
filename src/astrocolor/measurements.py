"""
Synthetic photometry, spectrometry and photometric scaling.

`get_photometry()` integrates (photo)spectral data through filters,
`get_spectrometry()` reconstructs or resamples spectra, and
`scale_to_match_value()` applies a brightness constraint at a filter or
wavelength.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import cast, overload

import numpy as np
import numpy.typing as npt

from astrocolor.errors import (
    InconsistentAxesError,
    NegativeStandardDeviationError,
    UnsupportedDimensionError,
)

from .auxiliary import parse_value_std
from .filter_objects import Filter, FilterSet
from .photospectral_objects import (
    PhotospectralCube,
    PhotospectralObject,
    PhotospectralSet,
    Photospectrum,
)
from .physical_models import PhysicalModel
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum
from .uncertainty import Uncertainty


@overload
def get_spectrometry(
    target: Filter,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> Filter:
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
    target: Spectrum | PhysicalModel,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> Spectrum:
    ...

@overload
def get_spectrometry(
    target: SpectralSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralSet:
    ...

@overload
def get_spectrometry(
    target: SpectralCube,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralCube:
    ...

@overload
def get_spectrometry(
    target: Photospectrum,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> Spectrum:
    ...

@overload
def get_spectrometry(
    target: PhotospectralSet,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralSet:
    ...

@overload
def get_spectrometry(
    target: PhotospectralCube,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralCube:
    ...

@overload
def get_spectrometry(
    target: PhotospectralObject,
    requested_wavelengths: npt.ArrayLike,
    strictly: bool = False
) -> SpectralObject:
    ...

@overload
def get_spectrometry(
    target: SpectralObject,
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
    Returns a new `Spectrum`/`SpectralSet`/`SpectralCube` object, guaranteeing that
    the specified wavelength range has been determined or reconstructed for it.
    If `strictly=True`, then the new object is defined exclusively
    on the specified wavelength range.
    Only the minimum and maximum wavelengths are extracted from the specified range,
    based on which a uniform grid is constructed.

    Args:
    - requested_wavelengths: Wavelength values to determine at.
    - strictly: If True, clip the result to the exact requested range.

    Returns:
    - A new object with data determined at the specified wavelengths.

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
        spectral_obj.uncertainty = spectral_obj.get_uncertainty_at_wavelengths(nm_min, nm_max)
        spectral_obj.wavelength_nm = requested_wavelengths
    # Sanity checks
    if (len_nm := spectral_obj.wavelength_nm.size) != (len_values := len(spectral_obj.spectral_dist)):
        raise InconsistentAxesError(len_nm, len_values, spectral_obj.name)
    if spectral_obj.uncertainty is not None:
        spectral_obj.uncertainty.validate(spectral_obj.spectral_dist.shape, spectral_obj.name)
    return spectral_obj


def _aligned_spectral_data(
    target: SpectralObject | PhotospectralObject,
    bandpass: Filter | FilterSet
) -> tuple[npt.NDArray[np.floating], Uncertainty | None] | None:
    """
    Returns target data sampled exactly on `bandpass.wavelength_nm`, when no
    reconstruction or extrapolation is needed. Avoids copying the whole object.
    """
    # Perf: avoid get_spectrometry() deepcopy when the band grid is already
    # present. Single-filter photometry on a 61 x 100 x 100 cube: 1.88 -> 0.44 ms.
    if not isinstance(target, SpectralObject):
        return None
    wavelengths = np.asarray(bandpass.wavelength_nm)
    if wavelengths.size == 0:
        return None
    indices = np.searchsorted(target.wavelength_nm, wavelengths)
    if np.any(indices >= target.spectral_size):
        return None
    if not np.array_equal(target.wavelength_nm[indices], wavelengths):
        return None
    uncertainty = None if target.uncertainty is None else target.uncertainty.select(indices)
    return target.spectral_dist[indices], uncertainty


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
) -> tuple[float, float | npt.NDArray[np.floating] | None] | PhotospectralObject:
    """
    Implementation of convolution between a (photo)spectral object and a filter or a filter set.
    Ignores the uncertainty of filter profiles.
    """
    if isinstance(target, PhotospectralObject) and isinstance(bandpass, FilterSet) and target.filter_set == bandpass:
        # The data is already measured in these filters: do not lose it to a reconstruction round trip.
        output = deepcopy(target)
        output.filter_set = bandpass
        return output
    aligned = _aligned_spectral_data(target, bandpass)
    if aligned is None:
        target = get_spectrometry(target, bandpass.wavelength_nm, strictly=True)
        sd = target.spectral_dist
        uncertainty = target.uncertainty
    else:
        sd, uncertainty = aligned
    ndim = target.ndim
    spatial_shape = target.spatial_shape
    n = cast(int, sd.shape[0])
    match bandpass:
        case Filter():
            # Perf: one BLAS dot instead of element-wise product plus sum.
            # 61 x 100 x 100 cube: 0.67 -> 0.04 ms.
            value_array = (bandpass.spectral_dist @ sd.reshape(n, -1)).reshape(spatial_shape) * target.nm_step
            value = float(value_array) if value_array.ndim == 0 else value_array
            error = None
            if uncertainty is not None:
                error = cast(
                    float | npt.NDArray[np.floating],
                    uncertainty.integrate(bandpass.spectral_dist, target.nm_step),
                )
                if np.ndim(error) == 0:
                    error = float(error)
            return cast(float, value), error
        case FilterSet():
            # Perf: BLAS matmul instead of einsum over spatial axes.
            # 3 filters on 61 x 100 x 100: 0.77 -> 0.18 ms.
            value = cast(
                npt.NDArray[np.floating],
                (bandpass.matrix @ sd.reshape(n, -1)).reshape(
                    (len(bandpass), *spatial_shape)
                ),
            )
            error = None if uncertainty is None else uncertainty.propagate(bandpass.matrix)
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
    requested_value: float | Sequence[float | None] = 1
) -> Spectrum | Photospectrum:
    """
    Returns a new object that matches the query brightness value (1 by default)
    at the specified filter or wavelength.

    If the requested value contains a standard deviation, it is propagated as a
    fully correlated (rank-one) covariance component, i.e. a common
    multiplicative error of the scaled spectrum. Such an error cancels in
    scale-invariant quantities, including color indices, but contributes to
    absolute brightness.
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
    if isinstance(requested_value, int | float):
        value, value_std = float(requested_value), None
    else:
        value, value_std = parse_value_std(
            cast(Sequence[float], requested_value)
        )
    if value_std is not None:
        if not np.isfinite(value_std):
            raise ValueError(
                f'Requested standard deviation must be finite, not {value_std}'
            )
        if value_std < 0:
            raise NegativeStandardDeviationError(spectral_data.name)
    # Synthetic measurement
    current_value, _ = get_photometry(spectral_data, bandpass)
    if current_value <= 0:
        # Prevents errors of dividing by zero and inversion
        return spectral_data
    output = spectral_data * (value / current_value)
    if value_std is not None and value_std > 0:
        # A common multiplicative error: all spectral points are scaled by the
        # same random factor, so the covariance is rank-one.
        shape = cast(
            npt.NDArray[np.floating],
            np.asarray(spectral_data.spectral_dist, dtype=np.float64),
        ) / current_value
        amplitude = value_std * shape
        scale_covariance = np.outer(amplitude, amplitude)
        if output.uncertainty is None:
            output.uncertainty = Uncertainty.from_covariance(
                scale_covariance, output.spectral_dist.shape, output.name
            )
        else:
            covariance = output.uncertainty.covariance_matrix + scale_covariance
            output.uncertainty = Uncertainty.from_covariance(
                covariance, output.spectral_dist.shape, output.name
            )
    return output
