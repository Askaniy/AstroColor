import numpy as np

import astrocolor as ac


class TestGetSpectrometry:

    def test_filter(self, v_filter: ac.Filter):
        spectrum = ac.get_spectrometry(v_filter, ac.visible_range, strictly=True)
        assert spectrum.wavelength_nm.size == ac.visible_range.size
        spectrum.edges_to_zero()
        assert v_filter == spectrum

    def test_filter_set(self, ubv_filterset: ac.FilterSet):
        spectral_set = ac.get_spectrometry(ubv_filterset, ac.visible_range, strictly=True)
        assert spectral_set.wavelength_nm.size == ac.visible_range.size

    def test_spectrum(self):
        u_filter = ac.Filter.get('Generic/Bessell.U')
        spectrum0 = ac.Spectrum(u_filter.wavelength_nm, u_filter.spectral_dist)
        spectrum1 = ac.get_spectrometry(spectrum0, ac.visible_range, strictly=True)
        assert spectrum1.wavelength_nm.size == ac.visible_range.size

    def test_spectral_set(self, ubv_filterset: ac.FilterSet):
        spectral_set0 = ac.SpectralSet(ubv_filterset.wavelength_nm, ubv_filterset.spectral_dist)
        spectral_set1 = ac.get_spectrometry(spectral_set0, ac.visible_range, strictly=True)
        assert spectral_set1.wavelength_nm.size == ac.visible_range.size

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

    def test_convolution_filter_integral(self, v_filter: ac.Filter):
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_CALSPEC, v_filter)[0], (ac.vega_CALSPEC * v_filter).integrate(), rtol=0.01
        )

    def test_convolution_filter_set_integral(self, ubv_filterset: ac.FilterSet):
        # FilterSet is normalized by 1 by design, you can't scale it to Vegan spectrum
        # SpectralSet could be normalized instead
        ubv_spectral_set = ac.SpectralSet(ubv_filterset.wavelength_nm, ubv_filterset.spectral_dist)
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_CALSPEC, ubv_filterset).spectral_dist,
            (ac.vega_CALSPEC * ubv_spectral_set).integrate(),
            rtol=0.01,
        )

    def test_vega_system_zero_points(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
        """ Zero-point calibration against Spanish Virtual Observatory """
        # TODO: check the agreement percent
        np.testing.assert_allclose(ac.get_photometry(ac.vega_CALSPEC, v_filter)[0], 3.62708e-9, rtol=0.0025)
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_CALSPEC, ubv_filterset).spectral_dist,
            [4.089744e-9, 6.365467e-9, 3.623954e-9],
            rtol=0.035,
        )
