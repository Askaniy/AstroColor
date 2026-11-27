"""Independent oracles for color transfer-function derivatives."""

from typing import cast

import numpy as np
import numpy.typing as npt


def srgb_decode_derivative(encoded: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """ Derivative of the sRGB EOTF (IEC 61966-2-1). """
    encoded = np.asarray(encoded, dtype=np.float64)
    nonlinear = 2.4 / 1.055 * np.power(
        np.maximum((encoded + 0.055) / 1.055, 0.),
        1.4,
    )
    return cast(
        npt.NDArray[np.floating],
        np.where(encoded < 0.04045, 1. / 12.92, nonlinear),
    )


def srgb_encode_derivative(linear: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """ Derivative of the sRGB OETF (IEC 61966-2-1). """
    linear = np.asarray(linear, dtype=np.float64)
    nonlinear = 1.055 / 2.4 * np.power(
        np.maximum(linear, 0.),
        1. / 2.4 - 1.,
    )
    return cast(
        npt.NDArray[np.floating],
        np.where(linear < 0.0031308, 12.92, nonlinear),
    )
