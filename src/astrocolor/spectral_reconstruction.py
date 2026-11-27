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
    OptimizeResult,
    minimize,
)

from .auxiliary import smoothness_matrix
from .config import Config
from .errors import UnsupportedDimensionError, uncertainty_loss_warning
from .measurements import get_spectrometry
from .photospectral_objects import PhotospectralObject
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum
from .uncertainty import Uncertainty


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
        n_size = cast(int, filter_matrix.shape[1])
        order1_matrix = smoothness_matrix(n_size, order=1, step=filter_set.nm_step)
        order2_matrix = smoothness_matrix(n_size, order=2, step=filter_set.nm_step)
        # TODO: research on some known spectra to find which alpha and beta fit best
        alpha = 1/8
        beta = 5000
        tikhonov_matrix = alpha * (order1_matrix.T @ order1_matrix + beta * order2_matrix.T @ order2_matrix)
        right_matrix = filter_matrix.T @ filter_matrix + tikhonov_matrix
        if photospectral_obj.ndim == 3:
            # scipy supports batch mode for 2d arrays, but not for 3D arrays
            br0 = br0.reshape(m_size, -1)
        left_vector = filter_matrix.T @ br0
        # Perf: one inversion plus a BLAS product beats a generic LAPACK solve
        # for many right-hand sides: 3 filters x 10000 pixels: 0.177 -> 0.079 ms.
        right_matrix_inv: npt.NDArray[np.floating] | None = None
        if n_size <= 16 or (left_vector.ndim == 2 and left_vector.shape[1] >= 64):
            right_matrix_inv = np.linalg.inv(right_matrix)
            br1 = cast(npt.NDArray[np.floating], right_matrix_inv @ left_vector)
        else:
            br1 = cast(npt.NDArray[np.floating], solve(right_matrix, left_vector))
        if photospectral_obj.ndim == 3:
            # Reshape the flattened pixels back to the spatial axes
            br1 = br1.reshape(-1, *photospectral_obj.spatial_shape)
        if photospectral_obj.ndim == 1 and br1.min() < 0:
            # To avoid negative spectra, a lower bound is set and iterative
            # optimization is performed using quadratic programming methods.
            # The processing speed drops by a factor of about five,
            # so the use is blocked for spectral sets and cubes:
            # background noise near zero can be most of the pixels.
            # TODO: RECHECK! MAY CONTAIN ERRORS!
            def objective(vector: npt.NDArray[np.floating]):
                # Tikhonov-regularized quadratic objective: 0.5 * Y^T A Y - b^T Y
                return 0.5 * vector @ right_matrix @ vector - left_vector @ vector
            def gradient(vector: npt.NDArray[np.floating]):
                # Gradient of the objective
                return right_matrix @ vector - left_vector
            k_size = cast(int, right_matrix.shape[1])
            bounds = ((0, None) for _ in range(k_size))
            result = cast(OptimizeResult, minimize(  # pyright: ignore[reportCallIssue]
                fun=objective,
                x0=np.maximum(br1, 0),  # pyright: ignore[reportArgumentType]
                jac=gradient,
                bounds=bounds,  # pyright: ignore[reportArgumentType]
                method='L-BFGS-B',
            ))
            if not result.success:  # pyright: ignore[reportAny]
                raise ValueError(f'Optimization failed: {result.message}')  # pyright: ignore[reportAny]
            br1 = cast(npt.NDArray[np.floating], result.x)
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
