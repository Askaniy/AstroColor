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
    Aligns a scalar-operation operand to the spectral or spatial axes of `mean`.

    A trailing array matching the spatial shape is left as-is (normal NumPy
    broadcasting). A one-dimensional array matching the spectral length is
    reshaped to `(n, 1, ...)` so that it is applied to every spatial pixel.
    """
    operand = np.asarray(operand)
    if mean.ndim == 0 or operand.ndim == 0:
        return operand
    if operand.ndim < mean.ndim and mean.shape[-operand.ndim:] == operand.shape:
        return operand
    if operand.ndim < mean.ndim and mean.shape[:operand.ndim] == operand.shape:
        return operand.reshape((
            *operand.shape,
            *((1,) * (mean.ndim - operand.ndim)),
        ))
    return operand


def _broadcast_pair(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ Aligns and broadcasts two data arrays of different dimensions. """
    x_arr = np.asarray(mean_x)
    y_arr = np.asarray(mean_y)
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
    return _broadcast_pair(mean_x, mean_y)


def add_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise sum of two aligned arrays. """
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
    return mean_x + mean_y

def sub_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise difference of two aligned arrays. """
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
    return mean_x - mean_y

def mul_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise product of two aligned arrays. """
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
    return mean_x * mean_y

def div_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """ Returns the element-wise quotient, handling division by zero. """
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
    try:
        with np.errstate(divide='raise', invalid='raise'):
            return mean_x / mean_y
    except FloatingPointError:
        if not np.asarray(mean_x).any():
            # Case of dividing zero by zero
            return np.zeros_like(mean_x)
        else:
            # Case of dividing by zero
            return np.full_like(mean_x, np.inf)
