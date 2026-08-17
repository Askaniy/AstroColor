from collections.abc import Sequence
from typing import cast, overload

import numpy as np
import numpy.typing as npt

from astrocolor.errors import UnsupportedDimensionError

from .algebra import mul_error, mul_value
from .auxiliary import integrate
from .core import Cube, Item, Set
from .filter_objects import Filter, FilterSet
from .photospectral_objects import (
    PhotospectralCube,
    PhotospectralObject,
    PhotospectralSet,
    Photospectrum,
)
from .spectral_objects import SpectralObject, Spectrum


@overload
def observe(
    target: Item | Set | Cube | SpectralObject | PhotospectralObject,
    bandpass: Filter
) -> tuple[float, float | None]:
    ...

@overload
def observe(
    target: Item,
    bandpass: FilterSet
) -> Photospectrum:
    ...

@overload
def observe(
    target: Set,
    bandpass: FilterSet
) -> PhotospectralSet:
    ...

@overload
def observe(
    target: Cube,
    bandpass: FilterSet
) -> PhotospectralCube:
    ...

@overload
def observe(
    target: SpectralObject,
    bandpass: FilterSet
) -> Photospectrum | PhotospectralSet | PhotospectralCube:
    ...

@overload
def observe(
    target: PhotospectralObject,
    bandpass: FilterSet
) -> Photospectrum | PhotospectralSet | PhotospectralCube:
    ...

def observe(
    target: Item | Set | Cube | SpectralObject | PhotospectralObject,
    bandpass: Filter | FilterSet
) -> tuple[float, float | None] | Photospectrum | PhotospectralSet | PhotospectralCube:
    """
    Implementation of convolution between a (photo)spectral object and a filter or a filter set.
    Ignores the uncertainty of filter profiles.
    """
    target = target.determine_at_wavelengths(bandpass.wavelength_nm, strictly=True)
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
    current_value, _ = observe(target, bandpass)
    if current_value <= 0:
        # Prevents errors of dividing by zero and inversion
        return target
    if isinstance(requested_value, Sequence):
        requested_value = requested_value[0] # likely a [value, std]
    # TODO: process std?
    return target * (requested_value / current_value)
