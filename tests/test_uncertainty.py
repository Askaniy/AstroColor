import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.errors import (
    InconsistentDerivativeShapeError,
    NegativeStandardDeviationError,
    NegativeVarianceError,
    NonFiniteCovarianceError,
    UncertaintyLossWarning,
)
from astrocolor.uncertainty import Uncertainty, combine_covariance_terms


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


@pytest.mark.parametrize(
    'name, expected_message',
    [
        (None, 'Covariance matrix contains negative variances.'),
        ('broken object', 'Covariance matrix contains negative variances for "broken object".'),
    ],
)
def test_negative_covariance_raises_typed_error(name: str | None, expected_message: str):
    nm = np.arange(500, 521, 5)
    covariance = np.eye(nm.size)
    covariance[0, 0] = -1.

    with pytest.raises(NegativeVarianceError) as exc_info:
        _ = ac.Spectrum(nm, np.ones(nm.size), covariance, name=name)

    message = str(exc_info.value)
    assert message == expected_message
    assert '"None"' not in message


@pytest.mark.parametrize(
    'name, expected_message',
    [
        (None, 'Standard deviations contain negative values.'),
        ('broken object', 'Standard deviations contain negative values for "broken object".'),
    ],
)
def test_negative_standard_deviation_raises_typed_error(
    name: str | None,
    expected_message: str,
):
    nm = np.arange(500, 521, 5)
    std = np.ones(nm.size)
    std[0] = -1.

    with pytest.raises(NegativeStandardDeviationError) as exc_info:
        _ = ac.Spectrum(nm, np.ones(nm.size), std, name=name)

    message = str(exc_info.value)
    assert message == expected_message
    assert '"None"' not in message


@pytest.mark.parametrize(
    'name, expected_message',
    [
        (None, 'Covariance matrix contains non-finite values.'),
        ('broken object', 'Covariance matrix contains non-finite values for "broken object".'),
    ],
)
def test_non_finite_covariance_raises_typed_error(
    name: str | None,
    expected_message: str,
):
    nm = np.arange(500, 521, 5)
    covariance = np.eye(nm.size)
    covariance[0, 0] = np.nan

    with pytest.raises(NonFiniteCovarianceError) as exc_info:
        _ = ac.Spectrum(nm, np.ones(nm.size), covariance, name=name)

    message = str(exc_info.value)
    assert message == expected_message
    assert '"None"' not in message


def test_scale_scales_standard_deviation_and_keeps_compact():
    std = np.array([1., 2., 3.])
    uncertainty = Uncertainty.from_standard_deviation(std, std.shape)

    scaled = uncertainty.scale(2.5)

    assert not scaled.is_covariance
    assert scaled.nbytes == uncertainty.nbytes
    np.testing.assert_allclose(scaled.standard_deviation, std * 2.5)
    np.testing.assert_allclose(uncertainty.standard_deviation, std)


def test_scale_scales_shared_covariance_without_expanding_it():
    covariance = np.array([
        [0.04, 0.01, 0.0],
        [0.01, 0.09, 0.02],
        [0.0, 0.02, 0.16],
    ])
    uncertainty = Uncertainty.from_covariance(covariance, (3,))

    scaled = uncertainty.scale(2.)

    assert scaled.is_covariance
    assert scaled.covariance_input().shape == (3, 3)
    np.testing.assert_allclose(scaled.covariance_matrix, covariance * 4.)
    np.testing.assert_allclose(uncertainty.covariance_matrix, covariance)


def test_apply_pointwise_rejects_wrong_derivative_shape():
    uncertainty = Uncertainty.from_standard_deviation(np.ones(3), (3,))

    with pytest.raises(
        InconsistentDerivativeShapeError,
        match='Derivative shape',
    ) as exc_info:
        _ = uncertainty.apply_pointwise(np.ones((3, 2)), name='test color')

    assert 'test color' in str(exc_info.value)


def test_scale_rejects_non_finite_factor():
    std = np.array([1., 2., 3.])
    uncertainty = Uncertainty.from_standard_deviation(std, std.shape)

    for factor in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError, match='Scale factor must be finite'):
            _ = uncertainty.scale(factor)


def test_scale_rejects_factor_whose_square_overflows():
    std = np.array([1., 2., 3.])
    uncertainty = Uncertainty.from_standard_deviation(std, std.shape)

    with pytest.raises(ValueError, match='square is not finite'):
        _ = uncertainty.scale(1e300)


def test_combine_covariance_terms_handles_large_scalar_weights():
    result = combine_covariance_terms(1., 1e200, None, 1.)

    assert result is not None
    assert np.isinf(result)


class TestMeanSpectrumUncertainty:
    """ Spatial aggregation must not lose the spectral uncertainty. """

    def test_mean_spectrum_preserves_spectral_uncertainty(self):
        nm = np.arange(500, 551, 5)
        std = np.linspace(0.1, 0.5, nm.size)
        obj = ac.Spectrum(nm, np.ones(nm.size), std)

        result = obj.mean_spectrum()

        assert result.uncertainty is not None
        result_std = result.standard_deviation
        assert result_std is not None
        np.testing.assert_allclose(result_std, std)
        np.testing.assert_array_equal(result.spectral_dist, obj.spectral_dist)

    def test_mean_spectrum_preserves_1d_covariance(self):
        nm = np.arange(500, 531, 5)
        covariance = np.eye(nm.size) * 0.1
        covariance[0, 1] = covariance[1, 0] = 0.02
        obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        result = obj.mean_spectrum()

        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        result_covariance = result.covariance_matrix
        assert result_covariance is not None
        np.testing.assert_allclose(result_covariance, covariance)

    def test_mean_spectrum_averages_independent_pixels(self):
        nm = np.arange(500, 551, 5)
        std = np.linspace(0.1, 0.5, nm.size)[:, np.newaxis] * np.array([1., 2., 3.])[np.newaxis, :]
        obj = ac.SpectralSet(nm, np.ones((nm.size, 3)), std)

        result = obj.mean_spectrum()

        assert result.uncertainty is not None
        assert not result.uncertainty.is_covariance
        expected_variance = np.mean(std**2, axis=1) / 3
        np.testing.assert_allclose(result.uncertainty.variance, expected_variance)

    def test_mean_spectrum_preserves_full_spectral_covariance(self):
        nm = np.arange(500, 531, 5)
        n = nm.size
        rng = np.random.default_rng(0)
        base = rng.normal(size=(n, n))
        covariance = np.stack(
            [
                (base @ base.T) / n + np.eye(n) * (0.1 + 0.01 * pixel)
                for pixel in range(4)
            ],
            axis=2,
        )
        obj = ac.SpectralSet(nm, np.ones((n, 4)), covariance)

        result = obj.mean_spectrum()

        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        result_covariance = result.covariance_matrix
        assert result_covariance is not None
        np.testing.assert_allclose(result_covariance, covariance.mean(axis=2) / 4)

    def test_mean_spectrum_divides_shared_covariance_by_pixel_count(self):
        nm = np.arange(500, 531, 5)
        rng = np.random.default_rng(1)
        base = rng.normal(size=(nm.size, nm.size))
        covariance = (base @ base.T) / nm.size + np.eye(nm.size) * 0.1
        obj = ac.SpectralSet(nm, np.ones((nm.size, 4)), covariance)

        result = obj.mean_spectrum()

        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        result_covariance = result.covariance_matrix
        assert result_covariance is not None
        np.testing.assert_allclose(result_covariance, covariance / 4)

    def test_mean_spectrum_averages_cube_variance(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ac.Config, 'ignore_uncertainty_for_cubes', False)
        nm = np.arange(500, 531, 5)
        std = np.full((nm.size, 2, 3), 0.1)
        cube = ac.SpectralCube(nm, np.ones((nm.size, 2, 3)), std)

        result = cube.mean_spectrum()

        assert result.uncertainty is not None
        assert not result.uncertainty.is_covariance
        np.testing.assert_allclose(result.uncertainty.variance, np.mean(std**2, axis=(1, 2)) / 6)

    def test_mean_spectrum_compacts_covariance_when_correlations_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        nm = np.arange(500, 531, 5)
        rng = np.random.default_rng(2)
        base = rng.normal(size=(nm.size, nm.size))
        covariance = np.stack(
            [
                (base @ base.T) / nm.size + np.eye(nm.size) * (0.1 + 0.01 * pixel)
                for pixel in range(4)
            ],
            axis=2,
        )
        obj = ac.SpectralSet(nm, np.ones((nm.size, 4)), covariance)
        assert obj.uncertainty is not None
        assert obj.uncertainty.is_covariance

        monkeypatch.setattr(ac.Config, 'ignore_correlations', True)
        result = obj.mean_spectrum()

        assert result.uncertainty is not None
        assert not result.uncertainty.is_covariance
        diagonal: npt.NDArray[np.floating] = np.diagonal(
            covariance, axis1=0, axis2=1
        ).T
        expected = diagonal.mean(axis=1) / 4
        np.testing.assert_allclose(result.uncertainty.variance, expected)

    def test_mean_spatial_rejects_empty_spatial_axis(self):
        uncertainty = Uncertainty.from_standard_deviation(
            np.ones((5, 0)), (5, 0)
        )

        with pytest.raises(ValueError, match='empty spatial'):
            _ = uncertainty.mean_spatial()


class TestMedianSpectrumUncertainty:
    """ The median is nonlinear, so only one spatial element can preserve uncertainty. """

    def test_median_spectrum_preserves_1d_uncertainty(self):
        nm = np.arange(500, 551, 5)
        std = np.linspace(0.1, 0.5, nm.size)
        obj = ac.Spectrum(nm, np.ones(nm.size), std)

        result = obj.median_spectrum()

        assert result.uncertainty is not None
        result_std = result.standard_deviation
        assert result_std is not None
        np.testing.assert_allclose(result_std, std)

    def test_median_spectrum_preserves_1d_covariance(self):
        nm = np.arange(500, 531, 5)
        covariance = np.eye(nm.size) * 0.1
        covariance[0, 1] = covariance[1, 0] = 0.02
        obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        result = obj.median_spectrum()

        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        result_covariance = result.covariance_matrix
        assert result_covariance is not None
        np.testing.assert_allclose(result_covariance, covariance)

    def test_median_spectrum_preserves_single_pixel_set_uncertainty(self):
        nm = np.arange(500, 531, 5)
        std = np.full((nm.size, 1), 0.1)
        obj = ac.SpectralSet(nm, np.ones((nm.size, 1)), std)

        result = obj.median_spectrum()

        assert result.uncertainty is not None
        result_std = result.standard_deviation
        assert result_std is not None
        np.testing.assert_allclose(result_std, std[:, 0])

    def test_median_spectrum_reduces_single_pixel_covariance_shape(self):
        nm = np.arange(500, 531, 5)
        covariance = np.eye(nm.size) * 0.1
        obj = ac.SpectralSet(
            nm, np.ones((nm.size, 1)), covariance[:, :, np.newaxis]
        )

        result = obj.median_spectrum()

        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        result_covariance = result.covariance_matrix
        assert result_covariance is not None
        np.testing.assert_allclose(result_covariance, covariance)

    def test_median_spectrum_preserves_single_pixel_cube_uncertainty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(ac.Config, 'ignore_uncertainty_for_cubes', False)
        nm = np.arange(500, 531, 5)
        std = np.full((nm.size, 1, 1), 0.1)
        obj = ac.SpectralCube(nm, np.ones((nm.size, 1, 1)), std)

        result = obj.median_spectrum()

        assert result.uncertainty is not None
        result_std = result.standard_deviation
        assert result_std is not None
        np.testing.assert_allclose(result_std, std[:, 0, 0])

    def test_median_spectrum_warns_and_drops_spatial_uncertainty(self):
        nm = np.arange(500, 551, 5)
        std = np.full((nm.size, 3), 0.1)
        obj = ac.SpectralSet(nm, np.ones((nm.size, 3)), std)

        with pytest.warns(UncertaintyLossWarning, match='median_spectrum'):
            result = obj.median_spectrum()

        assert result.uncertainty is None


class TestSpectralOperationUncertaintyLoss:
    """ 2D/3D confidence bands are not reconstructed: the loss must be explicit. """

    def test_reconstruction_warns_for_set_uncertainty(self, ubv_filterset: ac.FilterSet):
        photometry = ac.PhotospectralSet(
            ubv_filterset, np.ones((3, 2)), np.full((3, 2), 0.1)
        )

        with pytest.warns(UncertaintyLossWarning, match='reconstruct_from_photometry'):
            reconstructed = ac.get_spectrometry(
                photometry, ubv_filterset.wavelength_nm, strictly=True
            )

        assert reconstructed.uncertainty is None

    def test_reconstruction_warns_for_single_filter_spectrum(self):
        filter_set = ac.FilterSet.get('Generic/Bessell.V')
        photometry = ac.Photospectrum(filter_set, [1.0], [0.1])

        with pytest.warns(UncertaintyLossWarning, match='single-filter'):
            reconstructed = ac.get_spectrometry(
                photometry, filter_set.wavelength_nm, strictly=True
            )

        assert reconstructed.uncertainty is None

    def test_reconstruction_supports_single_filter_set(self):
        filter_set = ac.FilterSet.get('Generic/Bessell.V')
        photometry = ac.PhotospectralSet(
            filter_set, np.ones((1, 2)), np.full((1, 2), 0.1)
        )

        with pytest.warns(UncertaintyLossWarning, match='reconstruct_from_photometry'):
            reconstructed = ac.get_spectrometry(
                photometry, filter_set.wavelength_nm, strictly=True
            )

        assert reconstructed.spectral_dist.shape == (filter_set.wavelength_nm.size, 2)
        assert reconstructed.uncertainty is None

    def test_spectral_operation_warns_for_2d_uncertainty(self, ubv_filterset: ac.FilterSet):
        nm = ubv_filterset.wavelength_nm
        # The photometry itself has no uncertainty: the only possible loss is
        # the uncertainty of the spectral operand of the product.
        photometry = ac.PhotospectralSet(ubv_filterset, np.ones((3, 2)))
        spectra = ac.SpectralSet(
            nm, np.ones((nm.size, 2)), np.full((nm.size, 2), 0.1)
        )

        with pytest.warns(UncertaintyLossWarning, match='spectral operation'):
            result = photometry * spectra

        assert result.uncertainty is None

    def test_spectral_operation_warns_once_for_reconstruction_loss(
        self, ubv_filterset: ac.FilterSet
    ):
        nm = ubv_filterset.wavelength_nm
        photometry = ac.PhotospectralSet(
            ubv_filterset, np.ones((3, 2)), np.full((3, 2), 0.1)
        )
        spectra = ac.SpectralSet(
            nm, np.ones((nm.size, 2)), np.full((nm.size, 2), 0.1)
        )

        with pytest.warns(UncertaintyLossWarning, match='spectral operation') as record:
            result = photometry * spectra

        loss_warnings = [
            warning for warning in record
            if isinstance(warning.message, UncertaintyLossWarning)
        ]
        assert len(loss_warnings) == 1
        assert result.uncertainty is None

    def test_spectral_operation_warns_once_for_two_photometry_operands(
        self, ubv_filterset: ac.FilterSet
    ):
        left = ac.PhotospectralSet(
            ubv_filterset, np.ones((3, 2)), np.full((3, 2), 0.1)
        )
        right = ac.PhotospectralSet(
            ubv_filterset, np.ones((3, 2)), np.full((3, 2), 0.1)
        )

        with pytest.warns(UncertaintyLossWarning, match='spectral operation') as record:
            result = left * right

        loss_warnings = [
            warning for warning in record
            if isinstance(warning.message, UncertaintyLossWarning)
        ]
        assert len(loss_warnings) == 1
        assert result.uncertainty is None

    def test_spectral_operation_warns_for_single_filter_operand(self):
        filter_set = ac.FilterSet.get('Generic/Bessell.V')
        nm = filter_set.wavelength_nm
        photometry = ac.Photospectrum(filter_set, [1.0], [0.1])
        spectrum = ac.Spectrum(nm, np.ones(nm.size), np.full(nm.size, 0.1))

        with pytest.warns(UncertaintyLossWarning, match='spectral operation') as record:
            result = photometry * spectrum

        loss_warnings = [
            warning for warning in record
            if isinstance(warning.message, UncertaintyLossWarning)
        ]
        assert len(loss_warnings) == 1
        assert result.uncertainty is not None
