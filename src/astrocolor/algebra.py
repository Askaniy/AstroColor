from typing import cast

import numpy as np
import numpy.typing as npt


def make_same_ndim(
    arr1: npt.ArrayLike,
    arr2: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ Equalizes the arrays dimensions along spatial axes to make it possible to broadcast """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if (ndim_delta := arr2.ndim - arr1.ndim) != 0:
        if ndim_delta > 0:
            arr1 = arr1.reshape(arr1.shape + (1,) * ndim_delta)
        else:
            arr2 = arr2.reshape(arr2.shape + (1,) * (-ndim_delta))
    return arr1, arr2


# 1. Addition

def add_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    mean_x, mean_y = make_same_ndim(mean_x, mean_y)
    return mean_x + mean_y

def add_error(
    mean_x: npt.ArrayLike,  # pyright: ignore[reportUnusedParameter]
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,  # pyright: ignore[reportUnusedParameter]
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    if cov_x is None and cov_y is None:
        return None
    else:
        if cov_x is None:
            return np.asarray(cov_y)
        elif cov_y is None:
            return np.asarray(cov_x)
        else:
            cov_x, cov_y = make_same_ndim(cov_x, cov_y)
            return cov_x + cov_y


# 2. Subtraction

def sub_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    mean_x, mean_y = make_same_ndim(mean_x, mean_y)
    return mean_x - mean_y

def sub_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    return add_error(mean_x, cov_x, mean_y, cov_y)


# 3. Multiplication

def mul_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    mean_x = np.asarray(mean_x)
    mean_y = np.asarray(mean_y)
    try:
        # Numeric and same-ndim numpy arrays cases
        return mean_x * mean_y
    except ValueError:
        # Different ndim case
        if mean_y.ndim > mean_x.ndim:
            mean_x, mean_y = mean_y, mean_x
        return (mean_x.T * mean_y).T

def mul_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    if cov_x is None and cov_y is None:
        return None
    else:
        n = cast(int, np.asarray(mean_x).shape[0])
        cov_z = np.zeros((n, n)) # TODO: fix the case of x.ndim != y.ndim
        if cov_x is not None:
            diag_y = np.diag(mean_y)
            cov_z += diag_y @ cov_x @ diag_y
        if cov_y is not None:
            diag_x = np.diag(mean_x)
            cov_z += diag_x @ cov_y @ diag_x
        return cast(npt.NDArray[np.floating], cov_z)


# 4. Division

def div_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    mean_x = np.asarray(mean_x)
    mean_y = np.asarray(mean_y)
    try:
        # Numeric and same-ndim numpy arrays cases
        with np.errstate(divide='raise', invalid='raise'):
            return mean_x / mean_y
    except ValueError:
        # Different ndim case
        return (mean_x.T / mean_y).T
    except FloatingPointError:
        if not np.asarray(mean_x).any():
            # Case of dividing zero by zero
            return np.zeros_like(mean_x)
        else:
            # Case of dividing by zero
            return np.full_like(mean_x, np.inf)

def div_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    mean_x = np.asarray(mean_x)
    mean_y = np.asarray(mean_y)
    if cov_x is None and cov_y is None:
        return None
    else:
        n = cast(int, mean_x.shape[0])
        cov_z = np.zeros((n, n)) # TODO: fix the case of x.ndim != y.ndim
        if cov_x is not None:
            cov_x = np.asarray(cov_x)
            cov_z += cov_x
        if cov_y is not None:
            cov_y = np.asarray(cov_y)
            diag_z = np.diag(div_value(mean_x, mean_y))
            cov_z += diag_z @ cov_y @ diag_z
        diag_y_inv = np.diag(1 / mean_y)
        return diag_y_inv @ cov_z @ diag_y_inv
