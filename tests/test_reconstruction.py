from typing import cast

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.optimize import nnls

import astrocolor as ac
from astrocolor.spectral_reconstruction import (
    _nonnegative_factor,  # pyright: ignore[reportPrivateUsage]
    _project_nonnegative,  # pyright: ignore[reportPrivateUsage]
    _regularized_normal_matrix,  # pyright: ignore[reportPrivateUsage]
)


def _reconstruct_1d(
    filter_set: ac.FilterSet,
    values: list[float],
    nm: np.ndarray,
) -> ac.Spectrum:
    """ Reconstructs one photometric column through the public 1D path. """
    return cast(
        ac.Spectrum,
        ac.reconstruct_from_photometry(ac.Photospectrum(filter_set, values), nm),
    )


def test_flat_photospectrum_reconstruction(ubv_filterset: ac.FilterSet):
    photospectrum = ac.Photospectrum(ubv_filterset, (1, 1, 1), name='test photospectrum')
    reconstructed = ac.get_spectrometry(photospectrum, ac.visible_range, strictly=True)
    np.testing.assert_allclose(reconstructed.spectral_dist, np.ones(ac.visible_range.size))


PhotospectralClass = type[ac.Photospectrum | ac.PhotospectralSet | ac.PhotospectralCube]
PlainClass = type[ac.Spectrum | ac.SpectralSet | ac.SpectralCube]

@pytest.mark.parametrize(
    'photospectral_type, plain_type, shape',
    [
        (ac.Photospectrum, ac.Spectrum, (3,)),
        (ac.PhotospectralSet, ac.SpectralSet, (3, 2)),
        (ac.PhotospectralCube, ac.SpectralCube, (3, 2, 2)),
    ],
)
def test_reconstruction_returns_plain_spectral_object(
    ubv_filterset: ac.FilterSet,
    photospectral_type: PhotospectralClass,
    plain_type: PlainClass,
    shape: tuple[int, ...],
):
    photometry = photospectral_type(ubv_filterset, np.ones(shape))
    nm = ubv_filterset.wavelength_nm

    reconstructed = ac.get_spectrometry(photometry, nm, strictly=True)

    assert isinstance(reconstructed, plain_type)
    assert not hasattr(reconstructed, 'photospectral_obj')
    assert reconstructed.spectral_dist.shape == (nm.size, *shape[1:])


def test_reconstruct_from_photometry_returns_plain_spectrum(ubv_filterset: ac.FilterSet):
    photometry = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], name='test photospectrum')
    nm = ubv_filterset.wavelength_nm

    direct = ac.reconstruct_from_photometry(photometry, nm)
    measured = ac.get_spectrometry(photometry, nm)

    assert isinstance(direct, ac.Spectrum)
    assert isinstance(measured, ac.Spectrum)
    np.testing.assert_allclose(direct.spectral_dist, measured.spectral_dist)
    np.testing.assert_array_equal(direct.wavelength_nm, measured.wavelength_nm)


def test_1d_reconstruction_propagates_uncertainty(ubv_filterset: ac.FilterSet):
    photometry = ac.Photospectrum(
        ubv_filterset,
        [0.5, 1.0, 2.0],
        uncertainty=[0.05, 0.1, 0.2],
    )

    reconstructed = ac.get_spectrometry(photometry, ubv_filterset.wavelength_nm, strictly=True)

    assert reconstructed.uncertainty is not None
    assert reconstructed.standard_deviation is not None
    assert reconstructed.standard_deviation.shape == reconstructed.spectral_dist.shape


def test_flatten_reconstructed_cube(ubv_filterset: ac.FilterSet):
    nm = ubv_filterset.wavelength_nm
    cube = ac.PhotospectralCube(ubv_filterset, np.ones((3, 2, 2)), name='test cube')
    reconstructed = ac.get_spectrometry(cube, nm, strictly=True)
    assert isinstance(reconstructed, ac.SpectralCube)

    flattened = reconstructed.flatten()
    assert isinstance(flattened, ac.SpectralSet)
    assert flattened.spectral_dist.shape == (nm.size, 4)
    np.testing.assert_allclose(flattened.spectral_dist, reconstructed.spectral_dist.reshape(nm.size, 4))


VALID_COLUMN = [0.5, 1.0, 2.0]
NEGATIVE_COLUMN = [0.5, 1.0, -1.0]


def test_2d_reconstruction_clips_only_negative_pixels(ubv_filterset: ac.FilterSet):
    nm = ubv_filterset.wavelength_nm
    photometry = ac.PhotospectralSet(
        ubv_filterset,
        np.array([VALID_COLUMN, NEGATIVE_COLUMN]).T,
    )

    reconstructed = ac.reconstruct_from_photometry(photometry, nm)

    assert reconstructed.spectral_dist.shape == (nm.size, 2)
    assert np.all(reconstructed.spectral_dist >= 0)
    # The affected column is solved independently, exactly like the 1D case;
    # the already nonnegative column is left at its unconstrained solution.
    for index, values in enumerate((VALID_COLUMN, NEGATIVE_COLUMN)):
        expected = _reconstruct_1d(ubv_filterset, values, nm)
        np.testing.assert_allclose(
            reconstructed.spectral_dist[:, index],
            expected.spectral_dist,
            atol=1e-10,
        )


def test_3d_reconstruction_clips_negative_pixels(ubv_filterset: ac.FilterSet):
    nm = ubv_filterset.wavelength_nm
    pixels = (
        VALID_COLUMN,
        NEGATIVE_COLUMN,
        [-1.0, 0.5, 2.0],
        [1.0, 1.0, 1.0],
    )
    cube = ac.PhotospectralCube(
        ubv_filterset,
        np.array(pixels).T.reshape(3, 2, 2),
    )

    reconstructed = ac.reconstruct_from_photometry(cube, nm)

    assert isinstance(reconstructed, ac.SpectralCube)
    assert reconstructed.spectral_dist.shape == (nm.size, 2, 2)
    assert np.all(reconstructed.spectral_dist >= 0)
    flattened = reconstructed.spectral_dist.reshape(nm.size, -1)
    for index, values in enumerate(pixels):
        expected = _reconstruct_1d(ubv_filterset, values, nm)
        np.testing.assert_allclose(
            flattened[:, index],
            expected.spectral_dist,
            atol=1e-10,
        )


def test_all_nonnegative_pixels_skip_the_projection(ubv_filterset: ac.FilterSet):
    nm = ubv_filterset.wavelength_nm
    columns = (VALID_COLUMN, [0.7, 1.1, 1.5], [1.0, 1.0, 1.0])
    photometry = ac.PhotospectralSet(ubv_filterset, np.array(columns).T)

    reconstructed = ac.reconstruct_from_photometry(photometry, nm)

    assert np.all(reconstructed.spectral_dist >= 0)
    for index, values in enumerate(columns):
        expected = _reconstruct_1d(ubv_filterset, values, nm)
        np.testing.assert_allclose(
            reconstructed.spectral_dist[:, index],
            expected.spectral_dist,
            atol=1e-10,
        )


def test_regularized_normal_matrix_is_symmetric_positive_definite(
    ubv_filterset: ac.FilterSet,
):
    nm = ubv_filterset.wavelength_nm
    filter_set = ac.get_spectrometry(ubv_filterset, nm, strictly=False)
    right_matrix = _regularized_normal_matrix(filter_set)

    np.testing.assert_allclose(right_matrix, right_matrix.T)
    # Cholesky and the NNLS reformulation rely on A being positive definite.
    assert np.all(np.linalg.eigvalsh(right_matrix) > 0)


def test_nonnegative_nnls_matches_unconstrained_solve(ubv_filterset: ac.FilterSet):
    """
    If the unconstrained optimum is feasible, the constrained problem must
    return exactly the same point.

    Production skips NNLS when nothing is negative; this test checks the
    analytic claim directly, so the shortcut is not merely an implementation
    assumption.
    """
    nm = ubv_filterset.wavelength_nm
    filter_set = ac.get_spectrometry(ubv_filterset, nm, strictly=False)
    right_matrix = _regularized_normal_matrix(filter_set)
    left_vector = filter_set.matrix.T @ np.array(VALID_COLUMN)

    unconstrained = solve(right_matrix, left_vector)
    assert np.all(unconstrained >= 0)

    factor = _nonnegative_factor(right_matrix)
    transformed = solve(factor.T, left_vector)
    constrained, _ = nnls(factor, transformed)

    np.testing.assert_allclose(constrained, unconstrained, atol=1e-9)


def test_constrained_projection_satisfies_kkt_without_mutating_input(
    ubv_filterset: ac.FilterSet,
):
    nm = ubv_filterset.wavelength_nm
    filter_set = ac.get_spectrometry(ubv_filterset, nm, strictly=False)
    right_matrix = _regularized_normal_matrix(filter_set)
    left_vector = filter_set.matrix.T @ np.array(NEGATIVE_COLUMN)

    unconstrained = solve(right_matrix, left_vector)
    original = unconstrained.copy()
    values = _project_nonnegative(unconstrained, left_vector, right_matrix)

    # The unconstrained solution is negative, so the projection must actually
    # solve the QP; it must not modify the caller's array as a side effect.
    assert np.any(unconstrained < 0)
    np.testing.assert_array_equal(unconstrained, original)

    gradient = right_matrix @ values - left_vector
    active = values > 1e-10
    assert np.all(values >= 0)
    # KKT: zero gradient on positive coordinates, nonnegative gradient on zeros.
    np.testing.assert_allclose(gradient[active], 0.0, atol=1e-8)
    assert np.all(gradient[~active] >= -1e-8)
