"""
Continuous spectra reconstructed from photometry.

`reconstruct_from_photometry()` solves the inverse problem on a requested
wavelength grid and returns a plain `SpectralObject`.
"""

from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import (
    solve,
)
from scipy.optimize import (
    nnls,
)

from .auxiliary import smoothness_matrix
from .config import Config
from .errors import UnsupportedDimensionError, uncertainty_loss_warning
from .filter_objects import FilterSet
from .measurements import get_spectrometry
from .photospectral_objects import PhotospectralObject
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum
from .uncertainty import Uncertainty


def _regularized_normal_matrix(filter_set: FilterSet) -> npt.NDArray[np.floating]:
    """
    Builds the normal matrix of the Tikhonov-regularized reconstruction.

    The objective is `0.5 * y^T A y - b^T y` with `A = F^T F + T`, where `F`
    is the filter matrix. `T` combines first- and second-order smoothness
    operators, so it damps both height changes and curvature of the spectrum.
    """
    filter_matrix = filter_set.matrix
    n_size = cast(int, filter_matrix.shape[1])
    order1_matrix = smoothness_matrix(n_size, order=1, step=filter_set.nm_step)
    order2_matrix = smoothness_matrix(n_size, order=2, step=filter_set.nm_step)
    # TODO: research on some known spectra to find which alpha and beta fit best
    alpha = 1 / 8
    beta = 5000
    smoothness = alpha * (order1_matrix.T @ order1_matrix + beta * order2_matrix.T @ order2_matrix)
    return cast(
        npt.NDArray[np.floating],
        filter_matrix.T @ filter_matrix + smoothness,
    )


def _solve_normal_equations(
    right_matrix: npt.NDArray[np.floating],
    left_vector: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
    """
    Solves `A y = b` for one or many right-hand sides.

    Returns the solution and, when it was computed, the inverse of `A`; the
    covariance propagation reuses it instead of inverting `A` twice.
    """
    n_size = cast(int, right_matrix.shape[0])
    # Perf: one inversion plus a BLAS product beats a generic LAPACK solve
    # for many right-hand sides: 3 filters x 10000 pixels: 0.177 -> 0.079 ms.
    if n_size <= 16 or (left_vector.ndim == 2 and left_vector.shape[1] >= 64):
        right_matrix_inv = np.linalg.inv(right_matrix)
        return cast(npt.NDArray[np.floating], right_matrix_inv @ left_vector), right_matrix_inv
    return cast(npt.NDArray[np.floating], solve(right_matrix, left_vector)), None


def _nonnegative_factor(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Returns `G` with `G.T @ G = matrix`.

    The constrained problem `min 0.5 y^T A y - b^T y` with `y >= 0` is the
    nonnegative least-squares problem `min ||G y - c||^2` with `G^T c = b`.
    Cholesky is the cheap path; the eigen fallback also handles a
    positive-semidefinite matrix, for example a one-wavelength grid where the
    smoothness terms disappear.
    """
    try:
        lower = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = cast(
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
            np.linalg.eigh(matrix),
        )
        largest_eigenvalue = cast(float, np.max(np.abs(eigenvalues)))
        if np.any(eigenvalues < -1e-10 * max(1.0, largest_eigenvalue)):
            raise
        return np.sqrt(np.maximum(eigenvalues, 0.0))[:, np.newaxis] * eigenvectors.T
    return lower.T


def _project_nonnegative(
    values: npt.NDArray[np.floating],
    left_vector: npt.NDArray[np.floating],
    right_matrix: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Projects unconstrained solutions onto the nonnegative cone.

    The quadratic program separates over spatial pixels: `A` is shared, while
    every pixel has its own right-hand side. A pixel whose unconstrained
    solution is already nonnegative is optimal, so only negative pixels are
    re-solved. A 1D spectrum is the same path with a single "pixel", so 1D
    behavior is unchanged apart from the solver itself.

    The input array is not modified.
    """
    output_shape = values.shape
    n_size = cast(int, right_matrix.shape[0])
    columns = np.array(values, dtype=np.float64, copy=True).reshape(n_size, -1)
    right_hand_sides = np.asarray(left_vector, dtype=np.float64).reshape(n_size, -1)
    negative = np.any(columns < 0, axis=0)
    if not negative.any():
        return values
    # Perf: only pixels that violate `y >= 0` are re-solved, and NNLS with a
    # precomputed factor costs ~0.7 ms per pixel instead of ~120 ms for a
    # generic L-BFGS-B call on an 81-point grid. A noisy 100x100 UBV cube
    # (3497 negative pixels) therefore reconstructs in ~2.2 s, not minutes.
    factor = _nonnegative_factor(right_matrix)
    negative_indices = np.flatnonzero(negative)
    # `G^T c = b` keeps exactly the same quadratic objective, in the
    # numerically stable least-squares form that `nnls` expects. All right-hand
    # sides are solved in one LAPACK call; only NNLS itself stays per-pixel.
    transformed = cast(
        npt.NDArray[np.floating],
        solve(factor.T, right_hand_sides[:, negative_indices]),
    )
    for position, index in enumerate(negative_indices):
        solution, _ = nnls(factor, transformed[:, position])
        columns[:, index] = solution
    return cast(npt.NDArray[np.floating], columns.reshape(output_shape))


def reconstruct_from_photometry(
    photospectral_obj: PhotospectralObject,
    requested_wavelengths: npt.ArrayLike,
    # Reserved for future reconstruction modes:
    # reconstruct_mode: str = '',
) -> SpectralObject:
    """
    Reconstructs a SpectralObject from photospectral data on the wavelength array.

    The result is a plain `Spectrum`/`SpectralSet`/`SpectralCube`: the source
    photometry is not stored in it, so the caller keeps the input
    `PhotospectralObject` when it is needed later.

    Interpolation is not used because it is not a solution to the inverse ill-posed problem
    (i.e., looking at the spectrum through the filters does not give exactly the original photospectral_obj).

    The function uses the Tikhonov regularization method, with a combination of first-order
    and second-order differential operators for the Tikhonov matrix.
    That is, it tries to minimize height variations and curvature in the spectrum.

    After the unconstrained solution, every pixel that still has negative
    values is re-solved as a nonnegative least-squares problem. This applies
    to spectra, sets and cubes alike; a set or cube is processed pixel by
    pixel, because the quadratic program separates over the spatial axes.

    Confidence bands are computed only for 1D spectra reconstructed from at
    least two filters. Sets and cubes drop the uncertainty with an
    `UncertaintyLossWarning` because a full per-pixel reconstructed covariance
    would require `O(n² · pixels)` memory; a single-filter 1D reconstruction
    cannot constrain the spectral shape and also drops it.
    """
    br0 = photospectral_obj.spectral_dist
    filter_set = get_spectrometry(
        photospectral_obj.filter_set,
        requested_wavelengths,
        strictly=False,
    )
    nm1 = filter_set.wavelength_nm
    if Config.ignore_uncertainty_for_cubes and photospectral_obj.ndim == 3:
        cov0 = None
    else:
        cov0 = photospectral_obj.uncertainty
    cov1: Uncertainty | None = None
    if len(filter_set) == 1:
        # single-filter PhotospectralObject support
        target_shape = (nm1.size, *photospectral_obj.spatial_shape)
        br1 = np.broadcast_to(br0, target_shape).copy()
    else:
        filter_matrix = filter_set.matrix
        m_size = cast(int, filter_matrix.shape[0])
        right_matrix = _regularized_normal_matrix(filter_set)
        if photospectral_obj.ndim == 3:
            # scipy supports batch mode for 2d arrays, but not for 3D arrays
            br0 = br0.reshape(m_size, -1)
        left_vector = cast(
            npt.NDArray[np.floating],
            filter_matrix.T @ br0,
        )
        br1, right_matrix_inv = _solve_normal_equations(right_matrix, left_vector)
        br1 = _project_nonnegative(br1, left_vector, right_matrix)
        if photospectral_obj.ndim == 3:
            # Reshape the flattened pixels back to the spatial axes
            br1 = br1.reshape(-1, *photospectral_obj.spatial_shape)
        if photospectral_obj.ndim == 1 and cov0 is not None:
            # Measurement confidence band calculation
            # Confidence bands for spectral sets and cubes are not computed to save computational resources
            if right_matrix_inv is None:
                right_matrix_inv = np.linalg.inv(right_matrix)
            operator = right_matrix_inv @ filter_matrix.T
            cov1 = cov0.propagate(operator)
            #cov1 = filter_matrix.T @ cov0 @ filter_matrix + tikhonov_matrix # doesn't work
    if cov0 is not None and (photospectral_obj.ndim > 1 or len(filter_set) == 1):
        operation = (
            '`reconstruct_from_photometry()` for spectral sets and cubes'
            if photospectral_obj.ndim > 1
            else '`reconstruct_from_photometry()` for a single-filter spectrum'
        )
        uncertainty_loss_warning(operation, photospectral_obj.name)
    try:
        target_class = (Spectrum, SpectralSet, SpectralCube)[photospectral_obj.ndim - 1]
    except IndexError:
        raise UnsupportedDimensionError(photospectral_obj.ndim, photospectral_obj.name)
    return target_class._from_trusted_arrays(nm1, br1, cov1, photospectral_obj.name)  # pyright: ignore[reportPrivateUsage]
