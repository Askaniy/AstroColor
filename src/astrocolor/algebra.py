from typing import cast

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


def _make_diagonal_covariance_from_variance(
    var: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """
    Builds a covariance matrix `(n, n, *spatial)`
    from variances `(n, *spatial)`.
    """
    var = np.asarray(var, dtype=np.float64)
    n = cast(int, var.shape[0])
    cov = np.zeros((n, n, *var.shape[1:]), dtype=np.float64)
    indices = np.arange(n)
    cov[indices, indices] = var
    return cov

def make_diagonal_covariance(
    std: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """
    Builds a covariance matrix `(n, n, *spatial)`
    from standard deviations `(n, *spatial)`.
    """
    std = np.asarray(std, dtype=np.float64)
    return _make_diagonal_covariance_from_variance(std**2)

def get_standard_deviation(
    cov: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """
    Extracts standard deviations `(n, *spatial)`
    from a covariance matrix `(n, n, *spatial)`.
    """
    cov = np.asarray(cov, dtype=np.float64)
    return np.sqrt(cast(npt.NDArray[np.floating], np.einsum('ii...->i...', cov)))


def propagate_covariance(
    cov: npt.ArrayLike,
    matrix: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """
    Applies a linear operator to a covariance matrix:
    `C_out = matrix @ C_in @ matrix.T`.

    Supports covariance matrices with spatial axes: `C_in` has shape
    `(n, n, *spatial)`, `matrix` has shape `(m, n)`, and the result has shape
    `(m, m, *spatial)`.
    """
    cov = np.asarray(cov, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if cov.ndim < 2:
        raise ValueError(f'Covariance matrix must have at least 2 dimensions, not {cov.ndim}')
    if matrix.ndim != 2:
        raise ValueError(f'Linear operator must have 2 dimensions, not {matrix.ndim}')
    return cast(
        npt.NDArray[np.floating],
        np.einsum('ai,ij...,bj->ab...', matrix, cov, matrix),
    )


def _prepare_covariance(
    cov: npt.ArrayLike,
    mean: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """ Brings a covariance or variance array to the `(n, n, *spatial)` form. """
    cov = np.asarray(cov, dtype=np.float64)
    if mean.ndim == 0:
        return cov
    n = cast(int, mean.shape[0])
    if cov.shape == mean.shape:
        return _make_diagonal_covariance_from_variance(cov)
    if cov.ndim == 1 and cov.shape[0] == n:
        cov = _make_diagonal_covariance_from_variance(cov)
    if cov.ndim >= 2 and cov.shape[:2] == (n, n):
        extra = mean.ndim - (cov.ndim - 1)
        if extra < 0:
            raise ValueError(f'Covariance shape {cov.shape} does not match data shape {mean.shape}')
        return cov.reshape((*cov.shape, *((1,) * extra)))
    raise ValueError(f'Covariance shape {cov.shape} does not match data shape {mean.shape}')


def _covariance_term(
    mean: npt.ArrayLike,
    cov: npt.ArrayLike | None,
    weight: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating] | None:
    """ Returns `cov[i, j] * weight[i] * weight[j]` for the first-order propagation. """
    if cov is None:
        return None
    mean = np.asarray(mean, dtype=np.float64)
    if mean.ndim == 0:
        return _make_diagonal_covariance_from_variance(weight**2 * cast(float, np.asarray(cov).item()))
    cov = _prepare_covariance(cov, mean)
    target_ndim = weight.ndim + 1
    if cov.ndim < target_ndim:
        cov = cov.reshape((*cov.shape, *((1,) * (target_ndim - cov.ndim))))
    target_shape = (weight.shape[0], weight.shape[0], *weight.shape[1:])
    return np.broadcast_to(cov, target_shape) * weight[:, np.newaxis] * weight[np.newaxis, :]


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


def _broadcast_means(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ Broadcasts two data arrays along their common spectral and spatial axes. """
    return _broadcast_pair(mean_x, mean_y)


def _broadcast_covariance_to_spatial(
    cov: npt.NDArray[np.floating],
    mean: npt.NDArray[np.floating],
    target_spatial_shape: tuple[int, ...]
) -> npt.NDArray[np.floating]:
    """
    Brings a covariance or variance array to the full
    `(n, n, *target_spatial_shape)` form.
    """
    cov = _prepare_covariance(cov, mean)
    if cov.ndim < 2:
        raise ValueError('A covariance term must have at least two dimensions')
    extra_axes = len(target_spatial_shape) - (cov.ndim - 2)
    if extra_axes < 0:
        raise ValueError(
            f'Covariance shape {cov.shape} has more spatial axes than target {target_spatial_shape}'
        )
    cov = cov.reshape((*cov.shape, *((1,) * extra_axes)))
    return np.broadcast_to(
        cov,
        (cov.shape[0], cov.shape[1], *target_spatial_shape),
    ).copy()


# 1. Addition

def add_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
    return mean_x + mean_y

def add_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    if cov_x is None and cov_y is None:
        return None
    mean_x = np.asarray(mean_x, dtype=np.float64)
    mean_y = np.asarray(mean_y, dtype=np.float64)
    # Scalar-by-scalar addition is not used by the object algebra, but keeping
    # it here makes the helper complete and preserves the old behaviour.
    if mean_x.ndim == 0 and mean_y.ndim == 0:
        total = 0.
        if cov_x is not None:
            total += cast(float, np.asarray(cov_x).item())
        if cov_y is not None:
            total += cast(float, np.asarray(cov_y).item())
        return np.asarray(total)
    n = cast(int, mean_x.shape[0] if mean_x.ndim > 0 else mean_y.shape[0])
    target_spatial_shape = np.broadcast_shapes(mean_x.shape[1:], mean_y.shape[1:])
    terms: list[npt.NDArray[np.floating]] = []
    for mean, cov in ((mean_x, cov_x), (mean_y, cov_y)):
        if cov is None:
            continue
        cov_array = np.asarray(cov, dtype=np.float64)
        if mean.ndim == 0:
            if cov_array.ndim != 0:
                raise ValueError(
                    f'Covariance shape {cov_array.shape} does not match scalar data'
                )
            term = np.zeros((n, n, *target_spatial_shape), dtype=np.float64)
            indices = np.arange(n)
            term[indices, indices] = cov_array
            terms.append(term)
        else:
            terms.append(_broadcast_covariance_to_spatial(
                cov_array, mean, target_spatial_shape
            ))
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


# 2. Subtraction

def sub_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
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
    mean_x, mean_y = _broadcast_pair(mean_x, mean_y)
    return mean_x * mean_y

def mul_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    """
    Computes the covariance of the product z = x * y:
    cov_z[i, j] = cov_x[i, j] * y[i] * y[j] + cov_y[i, j] * x[i] * x[j].

    Supports scalars, 1D arrays and arrays with spatial axes. Covariances are
    expected in the `(n, n, *spatial)` form; variance arrays are accepted too.
    """
    if cov_x is None and cov_y is None:
        return None
    mean_x = np.asarray(mean_x, dtype=np.float64)
    mean_y = np.asarray(mean_y, dtype=np.float64)
    if mean_x.ndim == 0 and mean_y.ndim == 0:
        cov_z = 0.
        if cov_x is not None:
            cov_z += np.asarray(mean_y).item()**2 * cast(float, np.asarray(cov_x).item())
        if cov_y is not None:
            cov_z += np.asarray(mean_x).item()**2 * cast(float, np.asarray(cov_y).item())
        # Scalar inputs preserve the historical scalar return type.
        return cast(npt.NDArray[np.floating], cast(object, cov_z)) if not np.allclose(cov_z, 0) else None
    x_values, y_values = _broadcast_means(mean_x, mean_y)
    terms = (
        _covariance_term(mean_x, cov_x, y_values),
        _covariance_term(mean_y, cov_y, x_values),
    )
    cov_z = sum(term for term in terms if term is not None)
    return cast(npt.NDArray[np.floating], np.asarray(cov_z)) if not np.allclose(cov_z, 0) else None


# 4. Division

def div_value(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> npt.NDArray[np.floating]:
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

def div_error(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
) -> npt.NDArray[np.floating] | None:
    """
    Computes the covariance of the quotient z = x / y by first-order propagation.
    Supports scalars, 1D arrays and arrays with spatial axes.
    """
    if cov_x is None and cov_y is None:
        return None
    mean_x = np.asarray(mean_x, dtype=np.float64)
    mean_y = np.asarray(mean_y, dtype=np.float64)
    if mean_x.ndim == 0 and mean_y.ndim == 0:
        cov_z = 0.
        if cov_x is not None:
            cov_z += cast(float, np.asarray(cov_x).item()) / np.asarray(mean_y).item()**2
        if cov_y is not None:
            cov_z += np.asarray(mean_x).item()**2 * cast(float, np.asarray(cov_y).item()) / np.asarray(mean_y).item()**4
        # Scalar inputs preserve the historical scalar return type.
        return cast(npt.NDArray[np.floating], cast(object, cov_z)) if not np.allclose(cov_z, 0) else None
    x_values, y_values = _broadcast_means(mean_x, mean_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_x = 1 / y_values
        weight_y = x_values / y_values**2
    terms = (
        _covariance_term(mean_x, cov_x, weight_x),
        _covariance_term(mean_y, cov_y, weight_y),
    )
    cov_z = sum(term for term in terms if term is not None)
    return cast(npt.NDArray[np.floating], np.asarray(cov_z)) if not np.allclose(cov_z, 0) else None
