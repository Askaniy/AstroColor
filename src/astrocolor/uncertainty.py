"""
Measurement uncertainty storage and first-order propagation.

`Uncertainty` stores either compact variances `(n, *spatial)` or a full
covariance matrix `(n, n, *spatial)`. The compact form is kept as long as no
operation creates correlations between spectral points.

A shared `(n, n)` covariance matrix may be stored once and broadcast over
spatial axes. `Uncertainty.covariance_matrix` materializes a full array on
access; internal code uses `variance`, `covariance_input()` and `propagate()`
to avoid that allocation.

`Config.ignore_correlations` disables correlation handling: covariance inputs
are reduced to their diagonal, and operations on an existing full-covariance
object produce compact results. The flag is read at construction/operation
time; the stored representation of the source object is not changed.
"""

from math import ceil, sqrt
from typing import Self, cast, override

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, issparse

from .algebra import broadcast_means
from .config import Config
from .errors import (
    InconsistentUncertaintyShapeError,
    InconsistentUncertaintySizeError,
    NegativeStandardDeviationError,
    NegativeVarianceError,
    NonFiniteCovarianceError,
    non_finite_values_warning,
)

type CovarianceResult = npt.NDArray[np.floating] | float | None


def _broadcast_to_shape(
    array: npt.NDArray[np.floating],
    shape: tuple[int, ...]
) -> npt.NDArray[np.floating]:
    """ Appends missing spatial axes and broadcasts to `shape` without copying. """
    if array.shape == shape:
        return array
    if array.ndim < len(shape):
        array = array.reshape((*array.shape, *((1,) * (len(shape) - array.ndim))))
    return np.broadcast_to(array, shape)


def _diagonal_covariance_to_variance(
    covariance: npt.NDArray[np.floating],
    data_shape: tuple[int, ...]
) -> npt.NDArray[np.floating]:
    """ Extracts the diagonal of a covariance matrix as a variance array. """
    diagonal = cast(npt.NDArray[np.floating], np.einsum('ii...->i...', covariance))
    variance = _broadcast_to_shape(np.asarray(diagonal, dtype=np.float64), data_shape)
    return np.maximum(variance, 0.)


def _diagonal_from_variance(
    variance: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """ Builds a `(n, n, *spatial)` covariance matrix from variances `(n, *spatial)`. """
    n = cast(int, variance.shape[0])
    covariance = np.zeros((n, n, *variance.shape[1:]), dtype=np.float64)
    indices = np.arange(n)
    covariance[indices, indices] = variance
    return covariance


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
    n = cast(int, cov.shape[0])
    if cov.shape[1] != n:
        raise ValueError(f'Covariance matrix must be square, not {cov.shape[:2]}')
    if matrix.shape[1] != n:
        raise ValueError(
            f'Linear operator shape {matrix.shape} does not match the covariance size {n}'
        )
    m = cast(int, matrix.shape[0])
    spatial_shape = cov.shape[2:]
    # Fold all spatial axes into one to use a single BLAS matrix multiplication
    # for `matrix @ cov`. A direct three-operand `einsum` is orders of magnitude
    # slower for large spectra because it does not use the optimized BLAS path.
    intermediate = matrix @ cov.reshape(n, -1)
    intermediate = intermediate.reshape((m, n, *spatial_shape))
    return cast(
        npt.NDArray[np.floating],
        np.einsum('aj...,bj->ab...', intermediate, matrix, optimize=True),
    )


def _prepare_covariance(
    cov: npt.ArrayLike,
    weight: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """ Brings a covariance or variance array to the `(n, n, *spatial)` form. """
    cov = np.asarray(cov, dtype=np.float64)
    if weight.ndim == 0:
        return cov
    n = cast(int, weight.shape[0])
    if cov.shape == weight.shape:
        return _diagonal_from_variance(cov)
    if cov.ndim == 1 and cov.shape[0] == n:
        cov = _diagonal_from_variance(cov)
    if cov.ndim >= 2 and cov.shape[:2] == (n, n):
        extra = weight.ndim - (cov.ndim - 1)
        if extra < 0:
            raise ValueError(f'Covariance shape {cov.shape} does not match data shape {weight.shape}')
        return cov.reshape((*cov.shape, *((1,) * extra)))
    raise ValueError(f'Covariance shape {cov.shape} does not match data shape {weight.shape}')


def _covariance_term(
    cov: npt.ArrayLike | None,
    weight: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating] | None:
    """ Returns `cov[i, j] * weight[i] * weight[j]` for first-order propagation. """
    if cov is None:
        return None
    cov_array = np.asarray(cov, dtype=np.float64)
    if cov_array.ndim == 0:
        return _diagonal_from_variance(weight**2 * cov_array.item())
    covariance = _prepare_covariance(cov_array, weight)
    target_ndim = weight.ndim + 1
    if covariance.ndim < target_ndim:
        covariance = covariance.reshape((*covariance.shape, *((1,) * (target_ndim - covariance.ndim))))
    target_shape = (weight.shape[0], weight.shape[0], *weight.shape[1:])
    return np.broadcast_to(covariance, target_shape) * weight[:, np.newaxis] * weight[np.newaxis, :]


def combine_covariance_terms(
    cov_x: npt.ArrayLike | None,
    weight_x: npt.ArrayLike,
    cov_y: npt.ArrayLike | None,
    weight_y: npt.ArrayLike
) -> CovarianceResult:
    """
    Combines two first-order covariance contributions:
    `C_x[i, j] * weight_x[i] * weight_x[j] + C_y[i, j] * weight_y[i] * weight_y[j]`.

    The weights are the derivatives of the output with respect to each input.
    Covariances may be full matrices or compact variance arrays; `None` means
    that the corresponding input has no uncertainty.
    """
    if cov_x is None and cov_y is None:
        return None
    weight_x = np.asarray(weight_x, dtype=np.float64)
    weight_y = np.asarray(weight_y, dtype=np.float64)
    if weight_x.ndim == 0 and weight_y.ndim == 0:
        total = 0.
        if cov_x is not None:
            total += weight_x.item()**2 * cast(float, np.asarray(cov_x).item())
        if cov_y is not None:
            total += weight_y.item()**2 * cast(float, np.asarray(cov_y).item())
        return total if not np.allclose(total, 0) else None
    if weight_x.ndim == 0:
        weight_x = np.broadcast_to(weight_x, weight_y.shape)
    elif weight_y.ndim == 0:
        weight_y = np.broadcast_to(weight_y, weight_x.shape)
    terms = (
        _covariance_term(cov_x, weight_x),
        _covariance_term(cov_y, weight_y),
    )
    covariance = sum(term for term in terms if term is not None)
    return cast(npt.NDArray[np.floating], np.asarray(covariance)) if not np.allclose(covariance, 0) else None


def _broadcast_covariance(
    covariance: npt.NDArray[np.floating],
    data_shape: tuple[int, ...]
) -> npt.NDArray[np.floating]:
    """ Broadcasts a shared `(n, n)` covariance matrix over the spatial axes. """
    target = (data_shape[0], data_shape[0], *data_shape[1:])
    if covariance.shape == target:
        return covariance
    if covariance.ndim < len(target):
        covariance = covariance.reshape((*covariance.shape, *((1,) * (len(target) - covariance.ndim))))
    return np.broadcast_to(covariance, target)


class Uncertainty:
    """
    Measurement uncertainty of a (photo)spectral object.

    The compact representation stores only variances `(n, *spatial)` while the
    spectral axis is uncorrelated. A full covariance matrix `(n, n, *spatial)`
    is stored when the input or an operation introduces correlations between
    spectral points. A shared covariance matrix may also be stored as `(n, n)`
    and broadcast over spatial axes when needed.

    `standard_deviation` is the square root of the diagonal, and
    `covariance_matrix` is a full materialization. Both are read-only views of
    the stored uncertainty: mutating the returned array does not change the
    representation. Use `from_covariance()` or `from_input()` to replace it.

    When `Config.ignore_correlations` is True, covariance inputs are reduced to
    their diagonal at construction and operations on a full-covariance object
    produce compact results. The flag is checked dynamically; the stored
    representation itself is not changed until it is replaced.
    """

    __slots__: tuple[str, ...] = ('_covariance_matrix', '_data_shape', '_variance')

    _variance: npt.NDArray[np.floating] | None
    _covariance_matrix: npt.NDArray[np.floating] | None
    _data_shape: tuple[int, ...]

    def __init__(
        self,
        *,
        variance: npt.NDArray[np.floating] | None = None,
        covariance_matrix: npt.NDArray[np.floating] | None = None,
        data_shape: tuple[int, ...]
    ) -> None:
        """ Stores exactly one of variances or a covariance matrix. """
        if covariance_matrix is not None and Config.ignore_correlations:
            variance = _diagonal_covariance_to_variance(covariance_matrix, data_shape)
            covariance_matrix = None
        if (variance is None) == (covariance_matrix is None):
            raise ValueError('Exactly one of variance and covariance_matrix must be provided')
        self._variance = variance
        self._covariance_matrix = covariance_matrix
        self._data_shape = data_shape

    @classmethod
    def from_variance_unchecked(
        cls,
        variance: npt.ArrayLike,
        data_shape: tuple[int, ...]
    ) -> Self:
        """ Creates a compact uncertainty without validation (internal use). """
        array = np.asarray(variance, dtype=np.float64)
        return cls(variance=array, data_shape=data_shape)

    @classmethod
    def from_covariance_unchecked(
        cls,
        covariance_matrix: npt.ArrayLike,
        data_shape: tuple[int, ...]
    ) -> Self:
        """ Creates a covariance uncertainty without validation (internal use). """
        array = np.asarray(covariance_matrix, dtype=np.float64)
        return cls(covariance_matrix=array, data_shape=data_shape)

    @classmethod
    def from_standard_deviation(
        cls,
        standard_deviation: npt.ArrayLike,
        data_shape: tuple[int, ...] | None = None,
        name: object = None
    ) -> Self:
        """ Creates an uncertainty from standard deviations `(n, *spatial)`. """
        std = np.array(standard_deviation, dtype=np.float64)
        if data_shape is None:
            data_shape = std.shape
        if std.shape != data_shape:
            raise InconsistentUncertaintyShapeError(std.shape, data_shape, name)
        non_finite = ~np.isfinite(std)
        if np.any(non_finite):
            std[non_finite] = 0.
            non_finite_values_warning('standard deviation', name)
        if np.any(std < 0):
            raise NegativeStandardDeviationError(name)
        return cls.from_variance_unchecked(std**2, data_shape)

    @classmethod
    def from_covariance(
        cls,
        covariance_matrix: npt.ArrayLike,
        data_shape: tuple[int, ...],
        name: object = None
    ) -> Self:
        """
        Creates an uncertainty from a covariance matrix `(n, n, *spatial)`.

        The matrix is validated even when `Config.ignore_correlations=True`
        then reduced to its diagonal.
        """
        covariance = np.array(covariance_matrix, dtype=np.float64)
        cls._validate_covariance(covariance, data_shape, name)
        return cls.from_covariance_unchecked(covariance, data_shape)

    @classmethod
    def from_input(
        cls,
        uncertainty: 'npt.ArrayLike | Uncertainty',
        data_shape: tuple[int, ...],
        name: object = None
    ) -> Self:
        """
        Classifies an input as standard deviations or a covariance matrix,
        validates it, and returns the corresponding `Uncertainty`.

        A shared `(n, n)` covariance is accepted for objects with spatial axes
        and is stored once. With `Config.ignore_correlations=True`, only its
        diagonal is retained.
        """
        if isinstance(uncertainty, Uncertainty):
            uncertainty.validate(data_shape, name)
            if Config.ignore_correlations and uncertainty.is_covariance:
                return cls.from_variance_unchecked(uncertainty.variance, data_shape)
            return cast(Self, uncertainty)
        array = np.array(uncertainty, dtype=np.float64)
        n = data_shape[0]
        if array.ndim == 0 or array.shape[0] != n:
            received = 0 if array.ndim == 0 else array.shape[0]
            raise InconsistentUncertaintySizeError(received, n, name)
        if array.shape == data_shape:
            return cls.from_standard_deviation(array, data_shape, name)
        expected_covariance_shape = (n, n, *data_shape[1:])
        if array.shape == expected_covariance_shape:
            return cls.from_covariance(array, data_shape, name)
        if array.ndim == 2 and array.shape == (n, n):
            return cls.from_covariance(array, data_shape, name)
        if array.ndim == len(data_shape) + 1:
            raise InconsistentUncertaintyShapeError(array.shape, expected_covariance_shape, name)
        raise InconsistentUncertaintyShapeError(array.shape, data_shape, name)

    @staticmethod
    def _validate_covariance(
        covariance: npt.NDArray[np.floating],
        data_shape: tuple[int, ...],
        name: object = None
    ) -> None:
        n = data_shape[0]
        expected_shape = (n, n, *data_shape[1:])
        if covariance.shape not in (expected_shape, (n, n)):
            raise InconsistentUncertaintyShapeError(covariance.shape, expected_shape, name)
        if not np.all(np.isfinite(covariance)):
            raise NonFiniteCovarianceError(name)
        diagonal = np.diagonal(covariance, axis1=0, axis2=1)
        scale = max(1.0, float(cast(float, np.max(np.abs(covariance)))))
        if np.any(diagonal < -1e-10 * scale):
            raise NegativeVarianceError(name)

    def validate(self, data_shape: tuple[int, ...], name: object = None) -> None:
        """ Checks that the storage matches the shape of the data it describes. """
        n = data_shape[0]
        if self.spectral_size != n:
            raise InconsistentUncertaintySizeError(self.spectral_size, n, name)
        if self._variance is not None:
            if self._variance.shape != data_shape:
                raise InconsistentUncertaintyShapeError(self._variance.shape, data_shape, name)
        else:
            covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
            self._validate_covariance(covariance, data_shape, name)

    @property
    def data_shape(self) -> tuple[int, ...]:
        return self._data_shape

    @property
    def spectral_size(self) -> int:
        return self._data_shape[0]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self._data_shape[1:]

    @property
    def is_covariance(self) -> bool:
        """ `True` when correlations between spectral points are stored. """
        return self._covariance_matrix is not None

    @property
    def nbytes(self) -> int:
        """ Number of bytes used by the stored array, excluding object overhead. """
        if self._variance is not None:
            return self._variance.nbytes
        return cast(npt.NDArray[np.floating], self._covariance_matrix).nbytes

    @property
    def variance(self) -> npt.NDArray[np.floating]:
        """ Variances `(n, *spatial)`, extracted from the covariance diagonal if needed. """
        if self._variance is not None:
            return self._variance
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        variance = cast(npt.NDArray[np.floating], np.einsum('ii...->i...', covariance))
        return _broadcast_to_shape(variance, self._data_shape)

    @property
    def standard_deviation(self) -> npt.NDArray[np.floating]:
        """ Standard deviations `(n, *spatial)`. """
        return cast(npt.NDArray[np.floating], np.sqrt(self.variance))

    @property
    def covariance_matrix(self) -> npt.NDArray[np.floating]:
        """
        Full covariance matrix `(n, n, *spatial)`.

        For the compact representation the diagonal matrix is materialized on
        access; this does not change the stored representation. For a shared
        matrix the result may be a read-only broadcast view.
        """
        if self._covariance_matrix is not None:
            return _broadcast_covariance(self._covariance_matrix, self._data_shape)
        return _diagonal_from_variance(cast(npt.NDArray[np.floating], self._variance))

    def covariance_input(self) -> npt.NDArray[np.floating]:
        """ Returns compact variance or the stored covariance without broadcasting. """
        if self._covariance_matrix is not None:
            return self._covariance_matrix
        return cast(npt.NDArray[np.floating], self._variance)

    def compact(self) -> Self:
        """
        Returns a diagonal-only representation when correlations are ignored.

        When the flag is off, the object itself is returned unchanged.
        """
        if Config.ignore_correlations and self._covariance_matrix is not None:
            return self.from_variance_unchecked(self.variance, self._data_shape)
        return self

    def select(self, indices: npt.ArrayLike | slice) -> Self:
        """ Selects spectral points, preserving the uncertainty representation. """
        if Config.ignore_correlations and self.is_covariance:
            return self.from_variance_unchecked(self.variance, self.data_shape).select(indices)
        if isinstance(indices, slice):
            indices = np.arange(self.spectral_size)[indices]
        else:
            indices = np.asarray(indices)
            if indices.dtype == np.bool_:
                indices = np.flatnonzero(indices)
        new_shape = (indices.size, *self.spatial_shape)
        if self._variance is not None:
            return self.from_variance_unchecked(self._variance[indices], new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        return self.from_covariance_unchecked(covariance[np.ix_(indices, indices)], new_shape)

    def index_spatial(self, item: int | slice) -> Self:
        """ Selects the first spatial axis, preserving the representation. """
        if Config.ignore_correlations and self.is_covariance:
            return self.from_variance_unchecked(self.variance, self.data_shape).index_spatial(item)
        new_spatial = self._indexed_spatial_shape(item)
        new_shape = (self.spectral_size, *new_spatial)
        if self._variance is not None:
            return self.from_variance_unchecked(self._variance[(slice(None), item)], new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        if covariance.ndim == 2:
            return self.from_covariance_unchecked(covariance, new_shape)
        return self.from_covariance_unchecked(covariance[(slice(None), slice(None), item)], new_shape)

    def _indexed_spatial_shape(self, item: int | slice) -> tuple[int, ...]:
        if not self.spatial_shape:
            raise IndexError('The uncertainty has no spatial axes')
        if isinstance(item, int):
            return self.spatial_shape[1:]
        start, stop, step = item.indices(self.spatial_shape[0])
        return (len(range(start, stop, step)), *self.spatial_shape[1:])

    def reshape_spatial(self, shape: tuple[int, ...]) -> Self:
        """ Reinterprets spatial axes, e.g. for cube flattening. """
        if Config.ignore_correlations and self.is_covariance:
            return self.from_variance_unchecked(self.variance, self.data_shape).reshape_spatial(shape)
        new_shape = (self.spectral_size, *shape)
        if self._variance is not None:
            return self.from_variance_unchecked(self._variance.reshape(new_shape), new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        if covariance.ndim == 2:
            return self.from_covariance_unchecked(covariance, new_shape)
        return self.from_covariance_unchecked(
            covariance.reshape((self.spectral_size, self.spectral_size, *shape)),
            new_shape,
        )

    def pad_spectral(self, *, left: bool) -> Self:
        """ Adds one uncorrelated spectral point with zero variance. """
        if Config.ignore_correlations and self.is_covariance:
            return self.from_variance_unchecked(self.variance, self.data_shape).pad_spectral(left=left)
        n = self.spectral_size
        new_shape = (n + 1, *self.spatial_shape)
        if self._variance is not None:
            zeros = np.zeros((1, *self.spatial_shape), dtype=np.float64)
            variance = np.concatenate((zeros, self._variance)) if left else np.concatenate((self._variance, zeros))
            return self.from_variance_unchecked(variance, new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        extended = np.zeros(
            (n + 1, n + 1, *covariance.shape[2:]),
            dtype=np.float64,
        )
        if left:
            extended[1:, 1:] = covariance
        else:
            extended[:-1, :-1] = covariance
        return self.from_covariance_unchecked(extended, new_shape)

    def downscale_spatial(self, pixels_limit: int) -> Self:
        """ Decimates spatial axes in the same way as `spatial_downscaling()`. """
        if Config.ignore_correlations and self.is_covariance:
            return self.from_variance_unchecked(self.variance, self.data_shape).downscale_spatial(pixels_limit)
        if len(self.spatial_shape) != 2:
            return self
        x, y = self.spatial_shape
        factor = ceil(sqrt(x * y / pixels_limit))
        new_spatial = (
            (x + factor - 1) // factor,
            (y + factor - 1) // factor,
        )
        new_shape = (self.spectral_size, *new_spatial)
        if self._variance is not None:
            return self.from_variance_unchecked(self._variance[:, ::factor, ::factor], new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        if covariance.ndim == 2:
            return self.from_covariance_unchecked(covariance, new_shape)
        return self.from_covariance_unchecked(covariance[:, :, ::factor, ::factor], new_shape)

    def repeat_spatial(self, times: int) -> Self:
        """ Repeats every spatial axis, matching `np.repeat`-style upscaling. """
        if times < 1:
            raise ValueError(f'Repeat count must be positive, not {times}')
        if Config.ignore_correlations and self.is_covariance:
            return self.from_variance_unchecked(self.variance, self.data_shape).repeat_spatial(times)
        new_shape = (self.spectral_size, *(dim * times for dim in self.spatial_shape))
        if self._variance is not None:
            variance = self._variance
            for axis in range(1, variance.ndim):
                variance = np.repeat(variance, times, axis=axis)
            return self.from_variance_unchecked(variance, new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        if covariance.ndim == 2:
            return self.from_covariance_unchecked(covariance, new_shape)
        for axis in range(2, covariance.ndim):
            covariance = np.repeat(covariance, times, axis=axis)
        return self.from_covariance_unchecked(covariance, new_shape)

    def propagate(self, matrix: npt.ArrayLike | csr_matrix) -> Self:
        """
        Applies a linear operator to the uncertainty:
        `C_out = matrix @ C_in @ matrix.T`.

        The compact representation stays compact when the operator only scales
        or permutes spectral points (one non-zero per output row). Otherwise the
        result is a full covariance matrix. With `Config.ignore_correlations`
        enabled, only the resulting diagonal variance is computed and the
        result always stays compact.

        Sparse operators are applied without densifying them, which keeps
        dense-grid spectral binning at O(n + m) memory.
        """
        if issparse(matrix):
            return self._propagate_sparse(matrix)
        matrix = cast(
            npt.NDArray[np.floating],
            np.asarray(matrix, dtype=np.float64),
        )
        if matrix.ndim != 2:
            raise ValueError(f'Linear operator must have 2 dimensions, not {matrix.ndim}')
        if matrix.shape[1] != self.spectral_size:
            raise ValueError(
                f'Linear operator shape {matrix.shape} does not match the covariance size {self.spectral_size}'
            )
        output_size = cast(int, matrix.shape[0])
        new_shape = (output_size, *self.spatial_shape)
        if Config.ignore_correlations:
            n = self.spectral_size
            flattened = self.variance.reshape(n, -1)
            propagated = cast(npt.NDArray[np.floating], matrix**2 @ flattened)
            return self.from_variance_unchecked(propagated.reshape(new_shape), new_shape)
        if self._variance is not None:
            diagonal = self._diagonal_operator(matrix)
            if diagonal is not None:
                values, columns = diagonal
                scaled = values.reshape((-1, *((1,) * len(self.spatial_shape))))
                variance = self._variance[columns] * scaled**2
                return self.from_variance_unchecked(variance, new_shape)
            covariance = self._propagate_diagonal(matrix)
            return self.from_covariance_unchecked(covariance, new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        return self.from_covariance_unchecked(propagate_covariance(covariance, matrix), new_shape)

    def _propagate_sparse(self, matrix: csr_matrix) -> Self:
        """ Applies a sparse linear operator without materializing it as dense. """
        if matrix.shape[1] != self.spectral_size:
            raise ValueError(
                f'Linear operator shape {matrix.shape} does not match the covariance size {self.spectral_size}'
            )
        output_size = matrix.shape[0]
        new_shape = (output_size, *self.spatial_shape)
        if Config.ignore_correlations:
            n = self.spectral_size
            flattened = self.variance.reshape(n, -1)
            propagated = matrix.multiply(matrix) @ flattened
            return self.from_variance_unchecked(
                np.asarray(propagated).reshape(new_shape), new_shape
            )
        if self._variance is not None:
            diagonal = self._sparse_diagonal_operator(matrix)
            if diagonal is not None:
                values, columns = diagonal
                scaled = values.reshape((-1, *((1,) * len(self.spatial_shape))))
                variance = self._variance[columns] * scaled**2
                return self.from_variance_unchecked(variance, new_shape)
            covariance = self._propagate_sparse_diagonal(matrix)
            return self.from_covariance_unchecked(covariance, new_shape)
        covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
        return self.from_covariance_unchecked(
            self._propagate_sparse_covariance(matrix, covariance), new_shape
        )

    def _sparse_diagonal_operator(
        self,
        matrix: csr_matrix
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]] | None:
        """
        Sparse counterpart of `_diagonal_operator()`: every output point must
        depend on a single distinct input point.
        """
        operator = matrix.tocsr()
        row_counts = cast(npt.NDArray[np.integer], np.diff(operator.indptr))
        column_counts = cast(
            npt.NDArray[np.integer],
            np.bincount(operator.indices, minlength=self.spectral_size),
        )
        if (
            np.all(cast(npt.NDArray[np.bool_], row_counts == 1))
            and np.all(column_counts <= 1)
        ):
            return operator.data, cast(npt.NDArray[np.integer], operator.indices)
        return None

    def _propagate_sparse_diagonal(self, matrix: csr_matrix) -> npt.NDArray[np.floating]:
        """
        Computes `matrix @ diag(variance) @ matrix.T` for a sparse operator
        without densifying it.
        """
        variance = cast(npt.NDArray[np.floating], self._variance)
        n = self.spectral_size
        flattened = np.sqrt(variance.reshape(n, -1))
        output_size = matrix.shape[0]
        pixel_count = cast(int, flattened.shape[1])
        covariance = np.empty(
            (output_size, output_size, pixel_count), dtype=np.float64
        )
        for index in range(pixel_count):
            weighted = matrix.multiply(flattened[:, index][np.newaxis, :]).tocsr()
            covariance[:, :, index] = (weighted @ weighted.T).toarray()
        return cast(
            npt.NDArray[np.floating],
            covariance.reshape((output_size, output_size, *self.spatial_shape)),
        )

    def _propagate_sparse_covariance(
        self,
        matrix: csr_matrix,
        covariance: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """ Computes `matrix @ covariance @ matrix.T` for a sparse operator. """
        n = self.spectral_size
        output_size = matrix.shape[0]
        matrix_t = matrix.T.tocsr()
        spatial_shape = covariance.shape[2:]
        if not spatial_shape:
            return cast(
                npt.NDArray[np.floating],
                matrix @ covariance @ matrix_t,
            )
        flattened = covariance.reshape(n, n, -1)
        pixel_count = flattened.shape[2]
        result = np.empty(
            (output_size, output_size, pixel_count), dtype=np.float64
        )
        for index in range(pixel_count):
            intermediate = matrix @ flattened[:, :, index]
            result[:, :, index] = intermediate @ matrix_t
        return result.reshape((output_size, output_size, *spatial_shape))

    def _diagonal_operator(
        self,
        matrix: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]] | None:
        """
        Returns `(values, input_indices)` when every output point depends on a
        single distinct input point, so no correlations are created.
        """
        nonzero = cast(npt.NDArray[np.bool_], matrix != 0)
        row_counts = cast(npt.NDArray[np.integer], np.count_nonzero(nonzero, axis=1))
        column_counts = cast(npt.NDArray[np.integer], np.count_nonzero(nonzero, axis=0))
        row_single = cast(npt.NDArray[np.bool_], row_counts == 1)
        column_single = column_counts <= 1
        if np.all(row_single) and np.all(column_single):
            rows, columns = np.nonzero(nonzero)
            return matrix[rows, columns], columns
        return None

    def _propagate_diagonal(
        self,
        matrix: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        Computes `matrix @ diag(variance) @ matrix.T` without building the
        input covariance matrix. Spatial axes are folded to use batched BLAS.
        """
        variance = cast(npt.NDArray[np.floating], self._variance)
        n = self.spectral_size
        flattened = variance.reshape(n, -1)
        # (output, input, pixels) -> (pixels, output, input) @ (pixels, input, output)
        weighted = matrix[:, :, np.newaxis] * np.sqrt(flattened)[np.newaxis, :, :]
        batched = cast(
            npt.NDArray[np.floating],
            np.matmul(weighted.transpose(2, 0, 1), weighted.transpose(2, 1, 0)),
        )
        return batched.transpose(1, 2, 0).reshape(
            (matrix.shape[0], matrix.shape[0], *self.spatial_shape)
        )

    def integrate(
        self,
        weights: npt.ArrayLike,
        step: float = 1.
    ) -> npt.NDArray[np.floating]:
        """
        Variance of `sum(weights * x) * step`, without materializing the full
        covariance for the compact representation. With
        `Config.ignore_correlations` enabled, only the diagonal is used even if
        the object stores a full matrix.
        """
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.spectral_size,):
            raise ValueError(
                f'Integration weights shape {weights.shape} does not match the spectral size {self.spectral_size}'
            )
        if Config.ignore_correlations:
            squared = weights.reshape((-1, *((1,) * len(self.spatial_shape))))**2
            variance = cast(npt.NDArray[np.floating], np.sum(squared * self.variance, axis=0))
            return variance * step**2
        if self._variance is not None:
            squared = weights.reshape((-1, *((1,) * len(self.spatial_shape))))**2
            variance = cast(npt.NDArray[np.floating], np.sum(squared * self._variance, axis=0))
        else:
            covariance = cast(npt.NDArray[np.floating], self._covariance_matrix)
            projected = propagate_covariance(covariance, weights[np.newaxis, :])
            variance = cast(npt.NDArray[np.floating], projected[0, 0])
        return variance * step**2

    @override
    def __repr__(self) -> str:
        kind = 'covariance' if self.is_covariance else 'standard deviation'
        return f'Uncertainty({kind}, shape={self._data_shape})'


def _reduce_spatial_weight(
    weight: npt.NDArray[np.floating] | None,
    data_shape: tuple[int, ...]
) -> npt.NDArray[np.floating] | None:
    """
    Returns a spectral-only weight when it is constant over all spatial axes,
    or `None` when it varies across space.
    """
    if weight is None:
        return np.ones(data_shape[0], dtype=np.float64)
    if weight.ndim <= 1:
        return weight
    spatial_shape = data_shape[1:]
    if not spatial_shape or weight.shape != data_shape:
        return None
    spectral = weight[(slice(None),) + (0,) * len(spatial_shape)]
    broadcast = spectral.reshape((data_shape[0], *((1,) * len(spatial_shape))))
    if np.array_equal(weight, np.broadcast_to(broadcast, data_shape)):
        return spectral
    return None


def _combine(
    uncertainty_x: Uncertainty | None,
    weight_x: npt.NDArray[np.floating] | None,
    uncertainty_y: Uncertainty | None,
    weight_y: npt.NDArray[np.floating] | None,
    data_shape: tuple[int, ...]
) -> Uncertainty | None:
    """
    Combines first-order uncertainty terms:
    `C_z = diag(w_x) C_x diag(w_x)^T + diag(w_y) C_y diag(w_y)^T`.
    """
    if uncertainty_x is None and uncertainty_y is None:
        return None
    # Perf: diagonal path never materializes an n x n matrix. At N=121,
    # addition takes 0.020 ms vs 0.294 ms with full covariance; at N=5000,
    # storage is 40 KB vs 200 MB (5000x).
    if Config.ignore_correlations or (
        (uncertainty_x is None or not uncertainty_x.is_covariance)
        and (uncertainty_y is None or not uncertainty_y.is_covariance)
    ):
        variance = np.zeros(data_shape, dtype=np.float64)
        if uncertainty_x is not None:
            term = _broadcast_to_shape(uncertainty_x.variance, data_shape)
            variance += term if weight_x is None else weight_x**2 * term
        if uncertainty_y is not None:
            term = _broadcast_to_shape(uncertainty_y.variance, data_shape)
            variance += term if weight_y is None else weight_y**2 * term
        return Uncertainty.from_variance_unchecked(variance, data_shape)
    # A shared (n, n) covariance stays shared if all weights are spatially constant.
    shared = all(
        uncertainty is None or (uncertainty.is_covariance and uncertainty.covariance_input().ndim == 2)
        for uncertainty in (uncertainty_x, uncertainty_y)
    )
    if shared:
        spectral_weight_x = _reduce_spatial_weight(weight_x, data_shape)
        spectral_weight_y = _reduce_spatial_weight(weight_y, data_shape)
        if spectral_weight_x is not None and spectral_weight_y is not None:
            covariance = combine_covariance_terms(
                None if uncertainty_x is None else uncertainty_x.covariance_input(),
                spectral_weight_x,
                None if uncertainty_y is None else uncertainty_y.covariance_input(),
                spectral_weight_y,
            )
            if covariance is None:
                return None
            return Uncertainty.from_covariance_unchecked(
                cast(npt.NDArray[np.floating], covariance), data_shape
            )
    if weight_x is None:
        weight_x = np.ones(data_shape, dtype=np.float64)
    if weight_y is None:
        weight_y = np.ones(data_shape, dtype=np.float64)
    covariance = combine_covariance_terms(
        None if uncertainty_x is None else uncertainty_x.covariance_input(),
        weight_x,
        None if uncertainty_y is None else uncertainty_y.covariance_input(),
        weight_y,
    )
    if covariance is None:
        return None
    covariance = cast(npt.NDArray[np.floating], covariance)
    covariance = _broadcast_to_shape(np.asarray(covariance, dtype=np.float64), (data_shape[0], data_shape[0], *data_shape[1:]))
    return Uncertainty.from_covariance_unchecked(covariance, data_shape)


def add_uncertainty(
    mean_x: npt.ArrayLike,
    uncertainty_x: Uncertainty | None,
    mean_y: npt.ArrayLike,
    uncertainty_y: Uncertainty | None
) -> Uncertainty | None:
    """ Propagates uncertainty through `x + y` or `x - y`. """
    if uncertainty_x is None and uncertainty_y is None:
        return None
    x, _ = broadcast_means(mean_x, mean_y)
    return _combine(uncertainty_x, None, uncertainty_y, None, x.shape)


def sub_uncertainty(
    mean_x: npt.ArrayLike,
    uncertainty_x: Uncertainty | None,
    mean_y: npt.ArrayLike,
    uncertainty_y: Uncertainty | None
) -> Uncertainty | None:
    """ Propagates uncertainty through `x - y` (the same covariance as addition). """
    return add_uncertainty(mean_x, uncertainty_x, mean_y, uncertainty_y)


def mul_uncertainty(
    mean_x: npt.ArrayLike,
    uncertainty_x: Uncertainty | None,
    mean_y: npt.ArrayLike,
    uncertainty_y: Uncertainty | None
) -> Uncertainty | None:
    """ Propagates uncertainty through `x * y`. """
    if uncertainty_x is None and uncertainty_y is None:
        return None
    x, y = broadcast_means(mean_x, mean_y)
    return _combine(uncertainty_x, y, uncertainty_y, x, x.shape)


def div_uncertainty(
    mean_x: npt.ArrayLike,
    uncertainty_x: Uncertainty | None,
    mean_y: npt.ArrayLike,
    uncertainty_y: Uncertainty | None
) -> Uncertainty | None:
    """ Propagates uncertainty through `x / y`. """
    if uncertainty_x is None and uncertainty_y is None:
        return None
    x, y = broadcast_means(mean_x, mean_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_x = cast(npt.NDArray[np.floating], 1 / y)
        weight_y = cast(npt.NDArray[np.floating], x / y**2)
    return _combine(uncertainty_x, weight_x, uncertainty_y, weight_y, x.shape)
