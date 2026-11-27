"""
Broadcasting and element-wise arithmetic shared by the data classes.

The helpers align spectral and spatial axes before applying basic operations
and define division behavior at zero.
"""

import numpy as np
import numpy.typing as npt


def make_same_ndim(
    arr1: npt.ArrayLike,
    arr2: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Equalizes the arrays dimensions along spatial axes
    to make it possible to broadcast.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if (ndim_delta := arr2.ndim - arr1.ndim) != 0:
        if ndim_delta > 0:
            arr1 = arr1.reshape(arr1.shape + (1,) * ndim_delta)
        else:
            arr2 = arr2.reshape(arr2.shape + (1,) * (-ndim_delta))
    return arr1, arr2


def _align_operand(
    mean: npt.NDArray[np.floating],
    operand: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """
    Aligns a scalar-operation operand to the spectral axis of `mean`.

    An operand whose leading axes match `mean` is reshaped with trailing
    singleton axes: a vector of spectral length is applied to every spatial
    pixel, and a leading match wins over a trailing one. Any other shape is left
    for NumPy broadcasting; pass an explicit spatial shape such as `(1, m)` to
    scale pixels in the ambiguous case.
    """
    operand = np.asarray(operand)
    if mean.ndim == 0 or operand.ndim == 0:
        return operand
    if operand.ndim < mean.ndim and mean.shape[:operand.ndim] == operand.shape:
        return operand.reshape((
            *operand.shape,
            *((1,) * (mean.ndim - operand.ndim)),
        ))
    return operand


def _align_operands(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Aligns two operands along their spectral and spatial axes.

    Scalar operands are returned as-is: NumPy broadcasts them in the ufunc
    itself, and materializing a broadcast view costs more than the arithmetic on
    small spectra. `broadcast_means` expands them when a concrete shape is needed.
    """
    x_arr = np.asarray(mean_x)
    y_arr = np.asarray(mean_y)
    # Perf: skip np.broadcast_arrays for scalars and equal shapes. A broadcast
    # view of a 0-d operand costs ~6 us, more than the arithmetic itself on a
    # 61-point spectrum (~0.6 us).
    if x_arr.ndim == 0 or y_arr.ndim == 0 or x_arr.shape == y_arr.shape:
        return x_arr, y_arr
    if x_arr.ndim >= y_arr.ndim:
        y_arr = _align_operand(x_arr, y_arr)
    else:
        x_arr = _align_operand(y_arr, x_arr)
    x_values, y_values = np.broadcast_arrays(x_arr, y_arr)
    return x_values, y_values


def broadcast_means(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ Broadcasts two data arrays along their common spectral and spatial axes. """
    x_arr, y_arr = _align_operands(mean_x, mean_y)
    # Perf: expand scalar shapes only here, where covariance code needs a
    # concrete data shape; value operands keep scalars as-is.
    if x_arr.ndim == 0 or y_arr.ndim == 0:
        x_values, y_values = np.broadcast_arrays(x_arr, y_arr)
        return x_values, y_values
    return x_arr, y_arr


def add_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise sum of two aligned arrays. """
    mean_x, mean_y = _align_operands(mean_x, mean_y)
    return mean_x + mean_y

def sub_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise difference of two aligned arrays. """
    mean_x, mean_y = _align_operands(mean_x, mean_y)
    return mean_x - mean_y

def mul_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise product of two aligned arrays. """
    mean_x, mean_y = _align_operands(mean_x, mean_y)
    return mean_x * mean_y

def div_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """
    Returns the element-wise quotient with IEEE 754 semantics.

    `0 / 0` is `NaN` and `x / 0` keeps its sign as ±inf. Zero-area spectra are
    handled by `SpectralObject.normalized()` before division, so no global
    `0 / 0 = 0` convention is needed here.
    """
    mean_x, mean_y = _align_operands(mean_x, mean_y)
    # Perf: no NaN post-processing pass; zero-area spectra are handled by
    # `SpectralObject.normalized()` before division. This removes a full-array
    # pass from every division (cube / cube: 1.01 -> 0.79 ms).
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.asarray(mean_x, dtype=np.float64) / mean_y

def rsub_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise difference `mean_y - mean_x` for reflected operators. """
    return sub_value(mean_y, mean_x)

def rdiv_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise quotient `mean_y / mean_x` for reflected operators. """
    return div_value(mean_y, mean_x)
