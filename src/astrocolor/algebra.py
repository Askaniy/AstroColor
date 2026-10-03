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

# mul_error() was updated with AI and passed clean-up and checks, but needs more.
# TODO:
# - ensure spectral sets and cubes processing, "array.reshape(-1)" approach may break it
# - apply similar improvements to all operations
def mul_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | float | None:
    """
    Computes the covariance matrix of the product z = x * y

    Uses the error propagation formula:
    cov_z = diag(y) @ cov_x @ diag(y) + diag(x) @ cov_y @ diag(x)

    Parameters
    ----------
    mean_x : array-like or scalar
        Mean values of x
    cov_x : array-like or None
        Covariance matrix of x (shape (n, n)) or variance vector (shape (n,))
        For scalars - variance (number). None if no uncertainty.
    mean_y : array-like or scalar
        Mean values of y
    cov_y : array-like or None
        Covariance matrix of y (shape (n, n)) or variance vector (shape (n,))
        For scalars - variance (number). None if no uncertainty.

    Returns
    -------
    cov_z : ndarray of shape (n, n), float, or None
        Covariance matrix of the product z = x * y
        For scalars returns a number (variance)
        If both uncertainties are None, returns None

    Examples
    --------
    >>> # Multiplication of two scalars with errors
    >>> mul_error(10, 4, 20, 25)  # dx^2=4, dy^2=25
    700.0

    >>> # Multiplication of scalar by array
    >>> mul_error(3, None, [1, 2, 3], [0.1, 0.2, 0.3])
    array([[0.9, 0. , 0. ],
           [0. , 1.8, 0. ],
           [0. , 0. , 2.7]])

    >>> # Multiplication of arrays with correlation
    >>> x = [1, 2, 3]
    >>> cov_x = [[0.1, 0.02], [0.02, 0.2]]  # incomplete example
    >>> y = [4, 5, 6]
    >>> result = mul_error(x, cov_x, y, None)
    """
    if cov_x is None and cov_y is None:
        return None

    # Convert to numpy arrays
    mean_x_arr = np.asarray(mean_x)
    mean_y_arr = np.asarray(mean_y)

    # Determine dimensionalities
    x_is_scalar = mean_x_arr.ndim == 0
    y_is_scalar = mean_y_arr.ndim == 0

    # Case 1: Both scalars
    if x_is_scalar and y_is_scalar:
        cov_z = 0.
        if cov_x is not None:
            cov_z += (float(mean_y_arr) ** 2) * float(cov_x)
        if cov_y is not None:
            cov_z += (float(mean_x_arr) ** 2) * float(cov_y)

    # Cases 2 and 3: x is scalar, y is array OR y is scalar, x is array
    elif x_is_scalar or y_is_scalar:
        mean_scalar = mean_x if x_is_scalar else mean_y
        mean_array = mean_x_arr if y_is_scalar else mean_y_arr
        var_scalar = cov_x if x_is_scalar else cov_y
        cov_array = cov_x if y_is_scalar else cov_y

        # Flatten to 1D if needed
        if mean_array.ndim > 1:
            mean_arr_flat = mean_array.reshape(-1)
        else:
            mean_arr_flat = mean_array

        n = len(mean_arr_flat)
        cov_z = np.zeros((n, n))

        if cov_array is not None:
            # If cov_array is already a matrix (n, n)
            if hasattr(cov_array, 'shape') and len(np.asarray(cov_array).shape) == 2:
                cov_array = np.asarray(cov_array)
            else:
                # If cov_y is a variance vector, create diagonal matrix
                cov_array = np.diag(np.asarray(cov_array))

            # First part: scalar^2 * cov_array
            cov_z += (float(mean_scalar) ** 2) * cov_array

        if var_scalar is not None:
            # Second part: diag(array^2 * var_scalar)
            cov_z += np.diag(mean_arr_flat**2 * float(var_scalar))

    # Case 4: Both arrays
    else:
        # Flatten to 1D if needed
        if mean_x_arr.ndim > 1:
            mean_x_flat = mean_x_arr.reshape(-1)
        else:
            mean_x_flat = mean_x_arr

        if mean_y_arr.ndim > 1:
            mean_y_flat = mean_y_arr.reshape(-1)
        else:
            mean_y_flat = mean_y_arr

        # Check size compatibility
        n_x = len(mean_x_flat)
        n_y = len(mean_y_flat)
        if n_x != n_y:
            raise ValueError(
                f'Incompatible sizes: x has {n_x} elements, y has {n_y} elements'
            )

        n = n_x
        cov_z = np.zeros((n, n))

        if cov_x is not None:
            # Convert cov_x to matrix if needed
            if hasattr(cov_x, 'shape') and len(np.asarray(cov_x).shape) == 2:
                cov_x_arr = np.asarray(cov_x)
            else:
                cov_x_arr = np.diag(np.asarray(cov_x))
            diag_y = np.diag(mean_y_flat)
            cov_z += diag_y @ cov_x_arr @ diag_y

        if cov_y is not None:
            # Convert cov_y to matrix if needed
            if hasattr(cov_y, 'shape') and len(np.asarray(cov_y).shape) == 2:
                cov_y_arr = np.asarray(cov_y)
            else:
                cov_y_arr = np.diag(np.asarray(cov_y))
            diag_x = np.diag(mean_x_flat)
            cov_z += diag_x @ cov_y_arr @ diag_x

    return cov_z if not np.allclose(cov_z, 0) else None


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
