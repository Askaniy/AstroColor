from typing import cast

import numpy as np
import pytest

import astrocolor as ac
from astrocolor.errors import NegativeStandardDeviationError


class TestGetSpectrometry:

    def test_filter(self, v_filter: ac.Filter):
        spectrum = ac.get_spectrometry(v_filter, ac.visible_range, strictly=True)
        assert spectrum.wavelength_nm.size == ac.visible_range.size
        spectrum = spectrum.zero_edged()
        np.testing.assert_equal(spectrum.wavelength_nm, v_filter.wavelength_nm)
        np.testing.assert_allclose(spectrum.spectral_dist, v_filter.spectral_dist)

    def test_filter_set(self, ubv_filterset: ac.FilterSet):
        spectral_set = ac.get_spectrometry(ubv_filterset, ac.visible_range, strictly=True)
        np.testing.assert_equal(spectral_set.wavelength_nm, ac.visible_range)
        for index, filter_id in enumerate(('U', 'B', 'V')):
            expected = ac.get_spectrometry(
                ac.Filter.get(f'Generic/Bessell.{filter_id}'),
                ac.visible_range,
                strictly=True,
            )
            np.testing.assert_allclose(
                spectral_set.spectral_dist[:, index],
                expected.spectral_dist,
            )

    def test_spectrum(self):
        u_filter = ac.Filter.get('Generic/Bessell.U')
        spectrum0 = ac.Spectrum(u_filter.wavelength_nm, u_filter.spectral_dist)
        spectrum1 = ac.get_spectrometry(spectrum0, spectrum0.wavelength_nm, strictly=True)
        np.testing.assert_equal(spectrum1.wavelength_nm, spectrum0.wavelength_nm)
        np.testing.assert_allclose(spectrum1.spectral_dist, spectrum0.spectral_dist)

    def test_spectral_set(self, ubv_filterset: ac.FilterSet):
        spectral_set0 = ac.SpectralSet(ubv_filterset.wavelength_nm, ubv_filterset.spectral_dist)
        spectral_set1 = ac.get_spectrometry(spectral_set0, spectral_set0.wavelength_nm, strictly=True)
        np.testing.assert_equal(spectral_set1.wavelength_nm, spectral_set0.wavelength_nm)
        np.testing.assert_allclose(spectral_set1.spectral_dist, spectral_set0.spectral_dist)

    def test_flat_spectrum(self):
        """ A flat spectrum should remain flat after extrapolation. """
        nm = np.arange(500, 701, 5)
        spectrum = ac.Spectrum(nm, np.ones_like(nm))
        np.testing.assert_equal(
            ac.get_spectrometry(spectrum, ac.visible_range, strictly=True).spectral_dist,
            np.ones(ac.visible_range.size),
        )


class TestGetPhotometry:

    def test_stub_and_possibility(self):
        assert isinstance(ac.get_photometry(ac.Spectrum.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.get_photometry(ac.Spectrum.stub(), ac.FilterSet.stub()), ac.Photospectrum)
        assert isinstance(ac.get_photometry(ac.SpectralSet.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.get_photometry(ac.SpectralSet.stub(), ac.FilterSet.stub()), ac.PhotospectralSet)
        assert isinstance(ac.get_photometry(ac.SpectralCube.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.get_photometry(ac.SpectralCube.stub(), ac.FilterSet.stub()), ac.PhotospectralCube)
        assert isinstance(ac.get_photometry(ac.Photospectrum.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.get_photometry(ac.Photospectrum.stub(), ac.FilterSet.stub()), ac.Photospectrum)
        assert isinstance(ac.get_photometry(ac.PhotospectralSet.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.get_photometry(ac.PhotospectralSet.stub(), ac.FilterSet.stub()), ac.PhotospectralSet)
        assert isinstance(ac.get_photometry(ac.PhotospectralCube.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.get_photometry(ac.PhotospectralCube.stub(), ac.FilterSet.stub()), ac.PhotospectralCube)

    def test_convolution_filter_integral(self, v_filter: ac.Filter, vega_calspec: ac.Spectrum):
        np.testing.assert_allclose(
            ac.get_photometry(vega_calspec, v_filter)[0], (vega_calspec * v_filter).integrate(), rtol=0.001
        )

    def test_convolution_filter_set_integral(self, ubv_filterset: ac.FilterSet, vega_calspec: ac.Spectrum):
        # FilterSet is normalized by 1 by design, you can't scale it to Vegan spectrum
        # SpectralSet could be normalized instead
        ubv_spectral_set = ac.SpectralSet(ubv_filterset.wavelength_nm, ubv_filterset.spectral_dist)
        np.testing.assert_allclose(
            ac.get_photometry(vega_calspec, ubv_filterset).spectral_dist,
            (vega_calspec * ubv_spectral_set).integrate(),
            rtol=0.001,
        )

    def test_vega_system_zero_points(
        self,
        v_filter: ac.Filter,
        ubv_filterset: ac.FilterSet,
        vega_calspec: ac.Spectrum,
    ):
        """ Zero-point calibration against Spanish Virtual Observatory """
        np.testing.assert_allclose(ac.get_photometry(vega_calspec, v_filter)[0], 3.623954e-11, rtol=0.001)
        np.testing.assert_allclose(
            ac.get_photometry(vega_calspec, ubv_filterset).spectral_dist,
            [4.094111e-11, 6.374534e-11, 3.626637e-11],
            rtol=0.001,
        )


class TestScaleToMatchValue:

    def test_requested_std_adds_rank_one_covariance(self, v_filter: ac.Filter):
        nm = v_filter.wavelength_nm
        spectral_dist = np.linspace(1., 2., nm.size)
        std = np.full(nm.size, 0.01)
        spectrum = ac.Spectrum(nm, spectral_dist, std)
        current_value, _ = ac.get_photometry(spectrum, v_filter)
        result = ac.scale_to_match_value(spectrum, v_filter, (0.5, 0.05))
        assert result.uncertainty is not None
        assert result.uncertainty.is_covariance
        covariance = result.covariance_matrix
        assert covariance is not None
        factor = 0.5 / current_value
        shape = spectral_dist / current_value
        expected = factor**2 * np.diag(std**2) + 0.05**2 * np.outer(shape, shape)
        np.testing.assert_allclose(covariance, expected)
        # The added scale term is a rank-one common mode.
        scale_term = np.asarray(covariance, dtype=np.float64) - factor**2 * np.diag(std**2)
        assert np.linalg.matrix_rank(scale_term) == 1
        diagonal = np.sqrt(np.diag(scale_term))
        np.testing.assert_allclose(
            scale_term / np.outer(diagonal, diagonal),
            np.ones_like(scale_term),
        )

    def test_value_without_std_keeps_compact_uncertainty(self, v_filter: ac.Filter):
        nm = v_filter.wavelength_nm
        std = np.full(nm.size, 0.01)
        spectrum = ac.Spectrum(nm, np.linspace(1., 2., nm.size), std)
        current_value, _ = ac.get_photometry(spectrum, v_filter)
        result = ac.scale_to_match_value(spectrum, v_filter, 0.5)
        assert result.uncertainty is not None
        assert not result.uncertainty.is_covariance
        covariance = result.covariance_matrix
        assert covariance is not None
        np.testing.assert_allclose(
            covariance,
            (0.5 / current_value)**2 * np.diag(std**2),
        )

    def test_std_is_a_common_mode_that_cancels_in_color_index(self, v_filter: ac.Filter):
        filter_set = ac.FilterSet.get(
            'Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R'
        )
        spectrum = ac.Spectrum(filter_set.wavelength_nm, np.ones(filter_set.wavelength_nm.size))
        photometry = ac.get_photometry(spectrum, filter_set)
        scaled = ac.scale_to_match_value(photometry, v_filter, (0.5, 0.05))
        assert scaled.uncertainty is not None
        covariance = scaled.covariance_matrix
        assert covariance is not None
        covariance = np.asarray(covariance, dtype=np.float64)
        assert np.linalg.matrix_rank(covariance) == 1
        flux = np.asarray(scaled.spectral_dist, dtype=np.float64)
        # The B-V magnitude difference is invariant under a common scaling.
        variance = cast(
            float,
            covariance[0, 0] / flux[0]**2
            + covariance[1, 1] / flux[1]**2
            - 2 * covariance[0, 1] / (flux[0] * flux[1]),
        )
        assert variance == pytest.approx(0., abs=1e-15)
        # Absolute brightness at the constraining filter keeps the requested error.
        _, photometric_variance = ac.get_photometry(scaled, v_filter)
        assert photometric_variance is not None
        assert photometric_variance == pytest.approx(0.05**2, rel=1e-6)

    def test_negative_requested_std_is_rejected(self, v_filter: ac.Filter):
        spectrum = ac.Spectrum(v_filter.wavelength_nm, np.ones(v_filter.wavelength_nm.size))
        with pytest.raises(NegativeStandardDeviationError):
            _ = ac.scale_to_match_value(spectrum, v_filter, (0.5, -0.05))

    def test_non_finite_requested_std_is_rejected(self, v_filter: ac.Filter):
        spectrum = ac.Spectrum(v_filter.wavelength_nm, np.ones(v_filter.wavelength_nm.size))
        with pytest.raises(ValueError, match='finite'):
            _ = ac.scale_to_match_value(spectrum, v_filter, (0.5, float('nan')))
