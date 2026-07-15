import numpy as np
import astrocolor as ac


# === Convolution Tests ===

class TestConvolution():
    """ Tests for observe() and spectral convolution. """

    # - stub / return-type tests

    def test_stub_and_convolution_possibility(self):
        assert isinstance(ac.observe(ac.Spectrum.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.observe(ac.Spectrum.stub(), ac.FilterSet.stub()), ac.Photospectrum)
        assert isinstance(ac.observe(ac.SpectralSet.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.observe(ac.SpectralSet.stub(), ac.FilterSet.stub()), ac.PhotospectralSet)
        assert isinstance(ac.observe(ac.SpectralCube.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.observe(ac.SpectralCube.stub(), ac.FilterSet.stub()), ac.PhotospectralCube)
        assert isinstance(ac.observe(ac.Photospectrum.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.observe(ac.Photospectrum.stub(), ac.FilterSet.stub()), ac.Photospectrum)
        assert isinstance(ac.observe(ac.PhotospectralSet.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.observe(ac.PhotospectralSet.stub(), ac.FilterSet.stub()), ac.PhotospectralSet)
        assert isinstance(ac.observe(ac.PhotospectralCube.stub(), ac.Filter.stub()), tuple)
        assert isinstance(ac.observe(ac.PhotospectralCube.stub(), ac.FilterSet.stub()), ac.PhotospectralCube)

    # - convolution correctness

    def test_convolution(self, filter_v, filterset_ubv):
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC, filter_v)[0], (ac.vega_CALSPEC * filter_v).integrate(), rtol=0.01
        )
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC, filterset_ubv).spectral_dist,
            (ac.vega_CALSPEC * filterset_ubv).integrate(),
            rtol=0.01,
        )

    # - zero-point calibration against Spanish Virtual Observatory

    def test_vega_system_zero_points(self, filter_v, filterset_ubv):
        # TODO: check the agreement percent
        np.testing.assert_allclose(ac.observe(ac.vega_CALSPEC, filter_v)[0], 3.62708e-11, rtol=0.0025)
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC, filterset_ubv).spectral_dist,
            [4.089744e-11, 6.365467e-11, 3.623954e-11],
            rtol=0.035,
        )
