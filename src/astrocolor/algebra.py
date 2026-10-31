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
        return _make_diagonal_covariance_from_variance(weight**2 * float(cov))
    cov = _prepare_covariance(cov, mean)
    target_ndim = weight.ndim + 1
    if cov.ndim < target_ndim:
        cov = cov.reshape((*cov.shape, *((1,) * (target_ndim - cov.ndim))))
    target_shape = (weight.shape[0], weight.shape[0], *weight.shape[1:])
    return np.broadcast_to(cov, target_shape) * weight[:, np.newaxis] * weight[np.newaxis, :]


def _broadcast_means(
    mean_x: npt.ArrayLike,
    mean_y: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ Broadcasts two data arrays along their common spectral and spatial axes. """
    mean_x, mean_y = make_same_ndim(mean_x, mean_y)
    x_values, y_values = np.broadcast_arrays(mean_x, mean_y)
    return x_values, y_values


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
            cov_z += float(mean_y)**2 * float(cov_x)
        if cov_y is not None:
            cov_z += float(mean_x)**2 * float(cov_y)
    else:
        x_values, y_values = _broadcast_means(mean_x, mean_y)
        terms = (
            _covariance_term(mean_x, cov_x, y_values),
            _covariance_term(mean_y, cov_y, x_values),
        )
        cov_z = sum(term for term in terms if term is not None)
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
            cov_z += float(cov_x) / float(mean_y)**2
        if cov_y is not None:
            cov_z += float(mean_x)**2 * float(cov_y) / float(mean_y)**4
    else:
        x_values, y_values = _broadcast_means(mean_x, mean_y)
        with np.errstate(divide='ignore', invalid='ignore'):
            weight_x = 1 / y_values
            weight_y = x_values / y_values**2
        terms = (
            _covariance_term(mean_x, cov_x, weight_x),
            _covariance_term(mean_y, cov_y, weight_y),
        )
        cov_z = sum(term for term in terms if term is not None)
    return cov_z if not np.allclose(cov_z, 0) else None
