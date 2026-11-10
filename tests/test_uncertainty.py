import numpy as np
import pytest

import astrocolor as ac
from astrocolor.uncertainty import Uncertainty


def test_standard_deviation_input_stays_compact():
    nm = np.arange(500, 1001, 5)
    std = np.full(nm.size, 0.1)
    obj = ac.Spectrum(nm, np.ones(nm.size), std)

    assert obj.uncertainty is not None
    assert not obj.uncertainty.is_covariance
    assert obj.uncertainty.nbytes == std.nbytes
    # Explicit access still works, but must not promote the stored representation.
    assert obj.covariance_matrix is not None
    assert obj.covariance_matrix.shape == (nm.size, nm.size)
    assert not obj.uncertainty.is_covariance


def test_arithmetic_on_standard_deviations_stays_compact():
    nm = np.arange(500, 601, 5)
    values = np.linspace(1., 2., nm.size)
    std = np.full(nm.size, 0.1)
    obj = ac.Spectrum(nm, values, std)

    for result in (
        obj + obj,
        obj - obj,
        obj * obj,
        obj / obj,
        obj * 2,
        obj / 2,
        obj.normalized(),
        obj.determine_at_trusted_wavelengths(np.arange(490, 611, 5)),
    ):
        assert result.uncertainty is not None
        assert not result.uncertainty.is_covariance

    added_std = (obj + obj).standard_deviation
    product_std = (obj * obj).standard_deviation
    scaled_std = (obj * 2).standard_deviation
    assert added_std is not None
    assert product_std is not None
    assert scaled_std is not None
    np.testing.assert_allclose(added_std**2, 2 * std**2)
    np.testing.assert_allclose(product_std**2, 2 * (values * std)**2)
    np.testing.assert_allclose(scaled_std**2, 4 * std**2)


def test_linear_interpolation_promotes_to_covariance():
    nm = np.array([500., 512., 524., 536., 548., 560., 572., 584., 596.])
    std = np.linspace(0.1, 0.2, nm.size)
    obj = ac.Spectrum(nm, np.ones(nm.size), std)

    assert obj.uncertainty is not None
    assert obj.uncertainty.is_covariance
    assert obj.uncertainty.nbytes > std.nbytes


def test_full_covariance_input_roundtrip():
    nm = np.arange(500, 521, 5)
    covariance = np.diag(np.full(nm.size, 0.01))
    covariance[0, 1] = covariance[1, 0] = 0.002
    obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

    assert obj.uncertainty is not None
    assert obj.uncertainty.is_covariance
    stored = obj.covariance_matrix
    assert stored is not None
    np.testing.assert_allclose(stored, covariance)


def test_shared_covariance_is_not_expanded_over_space():
    nm = np.arange(500, 521, 5)
    covariance = np.eye(nm.size)
    obj = ac.SpectralSet(nm, np.ones((nm.size, 4)), covariance)

    assert obj.uncertainty is not None
    assert obj.uncertainty.is_covariance
    assert obj.uncertainty.nbytes == covariance.nbytes
    assert obj.covariance_matrix is not None
    assert obj.covariance_matrix.shape == (nm.size, nm.size, 4)


def test_filter_set_photometry_promotes_cross_filter_covariance(ubv_filterset: ac.FilterSet):
    nm = np.arange(500, 601, 5)
    std = np.full(nm.size, 0.1)
    spectrum = ac.Spectrum(nm, np.ones(nm.size), std)

    photospectrum = ac.get_photometry(spectrum, ubv_filterset)

    assert photospectrum.uncertainty is not None
    assert photospectrum.uncertainty.is_covariance
    assert photospectrum.covariance_matrix is not None
    assert photospectrum.covariance_matrix.shape == (
        len(ubv_filterset), len(ubv_filterset)
    )
    # Measuring a single band must not allocate the spectral covariance.
    assert spectrum.uncertainty is not None
    assert not spectrum.uncertainty.is_covariance


def test_invalid_uncertainty_is_rejected():
    nm = np.arange(500, 521, 5)
    with pytest.raises(ValueError):
        _ = ac.Spectrum(nm, np.ones(nm.size), -np.ones(nm.size))
    with pytest.raises(ValueError):
        _ = Uncertainty.from_covariance(
            np.full((nm.size, nm.size), np.nan), (nm.size,)
        )


def test_photospectral_standard_deviation_is_compact(ubv_filterset: ac.FilterSet):
    std = np.array([0.05, 0.1, 0.2])
    obj = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], std)

    assert obj.uncertainty is not None
    assert not obj.uncertainty.is_covariance
    assert obj.uncertainty.nbytes == std.nbytes
    result = obj + obj
    assert result.uncertainty is not None
    assert not result.uncertainty.is_covariance


def test_cube_uncertainty_is_ignored_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac.Config, 'ignore_uncertainty_for_cubes', True)
    nm = np.arange(500, 521, 5)
    cube = ac.SpectralCube(nm, np.ones((nm.size, 2, 2)), np.ones((nm.size, 2, 2)))

    assert cube.uncertainty is None
    assert cube.covariance_matrix is None


def test_shared_covariance_survives_scalar_arithmetic():
    nm = np.arange(500, 521, 5)
    covariance = np.eye(nm.size) * 0.01
    obj = ac.SpectralSet(nm, np.ones((nm.size, 4)), covariance)

    for result in (obj * 2, obj / 2, obj + obj, obj - obj):
        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        assert result.uncertainty.nbytes == covariance.nbytes

    scaled = (obj * 2).covariance_matrix
    added = (obj + obj).covariance_matrix
    assert scaled is not None
    assert added is not None
    np.testing.assert_allclose(scaled[..., 0], covariance * 4)
    np.testing.assert_allclose(added[..., 0], covariance * 2)


def test_shared_covariance_plus_diagonal_becomes_spatial():
    nm = np.arange(500, 521, 5)
    shared = np.eye(nm.size) * 0.01
    std = np.full((nm.size, 2), 0.2)
    left = ac.SpectralSet(nm, np.ones((nm.size, 2)), shared)
    right = ac.SpectralSet(nm, np.ones((nm.size, 2)), std)

    result = left + right

    covariance = result.covariance_matrix
    assert covariance is not None
    assert covariance.shape == (nm.size, nm.size, 2)
    expected = shared[:, :, np.newaxis] + np.diag(std[:, 0]**2)[:, :, np.newaxis]
    np.testing.assert_allclose(covariance[:, :, 0], expected[:, :, 0])
    np.testing.assert_allclose(covariance[:, :, 1], shared + np.diag(std[:, 1]**2))


def test_repr_uses_uncertainty_summary_for_spatial_objects():
    nm = np.arange(500, 601, 5)
    obj = ac.SpectralSet(nm, np.ones((nm.size, 2)), np.full((nm.size, 2), 0.1))

    text = repr(obj)

    assert 'uncertainty = Uncertainty(standard deviation' in text
    assert 'standard_deviation =' not in text


def test_repr_keeps_standard_deviation_values_for_spectrum():
    nm = np.arange(500, 601, 5)
    obj = ac.Spectrum(nm, np.ones(nm.size), np.full(nm.size, 0.1))

    text = repr(obj)

    assert 'standard_deviation = [0.100, 0.100, ..., 0.100]' in text


def test_ignore_correlations_defaults_to_false():
    assert ac.Config.ignore_correlations is False


def test_ignore_correlations_collapses_covariance_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    nm = np.arange(500, 601, 5)
    covariance = np.full((nm.size, nm.size), 0.01)
    obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

    assert obj.uncertainty is not None
    assert not obj.uncertainty.is_covariance
    assert obj.standard_deviation is not None
    np.testing.assert_allclose(obj.standard_deviation, np.full(nm.size, 0.1))
    stored = obj.covariance_matrix
    assert stored is not None
    np.testing.assert_allclose(stored, np.eye(nm.size) * 0.01)


def test_ignore_correlations_collapses_shared_covariance(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    nm = np.arange(500, 601, 5)
    covariance = np.full((nm.size, nm.size), 0.01)
    obj = ac.SpectralSet(nm, np.ones((nm.size, 2)), covariance)

    assert obj.uncertainty is not None
    assert not obj.uncertainty.is_covariance
    assert obj.standard_deviation is not None
    assert obj.standard_deviation.shape == (nm.size, 2)


def test_ignore_correlations_keeps_propagation_compact(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    nm = np.arange(500, 531, 5)
    std = np.full(nm.size, 0.1)
    matrix = np.array([[1., 1., 0., 0., 0., 0., 0.],
                       [0., 1., 1., 0., 0., 0., 0.]])
    uncertainty = Uncertainty.from_standard_deviation(std)

    result = uncertainty.propagate(matrix)

    assert not result.is_covariance
    np.testing.assert_allclose(result.variance, matrix**2 @ std**2)


def test_ignore_correlations_collapses_existing_object_on_next_operation(
    monkeypatch: pytest.MonkeyPatch,
):
    nm = np.arange(500, 551, 5)
    covariance = np.zeros((nm.size, nm.size))
    covariance[0, 1] = covariance[1, 0] = 0.002
    np.fill_diagonal(covariance, 0.01)
    obj = ac.Spectrum(nm, np.ones(nm.size), covariance)
    assert obj.uncertainty is not None
    assert obj.uncertainty.is_covariance

    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    result = obj * 2

    assert result.uncertainty is not None
    assert not result.uncertainty.is_covariance


def test_ignore_correlations_collapses_color_covariance(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    covariance = np.array([
        [0.01, 0.002, 0.001],
        [0.002, 0.02, 0.003],
        [0.001, 0.003, 0.03],
    ])
    color = ac.ColorPoint(np.array([0.2, 0.3, 0.4]), covariance)

    assert color.uncertainty is not None
    assert not color.uncertainty.is_covariance
    stored = color.covariance_matrix
    assert stored is not None
    np.testing.assert_allclose(stored, np.diag(np.diag(covariance)))


def test_ignore_correlations_keeps_color_image_upscale_compact(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    covariance = np.ones((3, 3, 2, 3)) * 0.01
    image = ac.ColorImage(np.ones((3, 2, 3)), covariance)

    upscaled = image.upscale(2)

    assert upscaled.uncertainty is not None
    assert not upscaled.uncertainty.is_covariance
    assert upscaled.covariance_matrix is not None
    assert upscaled.covariance_matrix.shape == (3, 3, 4, 6)


def test_ignore_correlations_still_validates_covariance(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
    nm = np.arange(500, 521, 5)
    covariance = np.eye(nm.size)
    covariance[0, 0] = np.nan

    with pytest.raises(ValueError):
        _ = ac.Spectrum(nm, np.ones(nm.size), covariance)
