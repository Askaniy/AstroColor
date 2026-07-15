import numpy as np
import astrocolor as ac


# === Arithmetic Operations Tests ===

class TestArithmetic():
    """ Tests for +, *, / operations between spectra and filters. """

    # - addition

    def test_addition_spectrum(self):
        np.testing.assert_allclose(
            (ac.vega_CALSPEC + ac.vega_CALSPEC).spectral_dist,
            (ac.vega_CALSPEC * 2).spectral_dist,
            rtol=0.01,
        )

    # - multiplication

    def test_multiplication_filter_spectrum_mean(self, filter_v, filterset_ubv):
        np.testing.assert_allclose(
            (filter_v * ac.vega_CALSPEC).mean_nm(), 544.601418, rtol=0.01
        )  # 544.543 in SVO Filter Profile Service
        np.testing.assert_allclose(
            (filterset_ubv * ac.vega_CALSPEC).mean_nm(),
            [366.764603, 435.741381, 544.601418],
            rtol=0.01,
        )

    def test_multiplication_observation(self, filter_v, filterset_ubv):
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC * 2, filter_v)[0],
            ac.observe(ac.vega_CALSPEC, filter_v)[0] * 2,
            rtol=0.01,
        )
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC * 2, filterset_ubv).spectral_dist,
            ac.observe(ac.vega_CALSPEC, filterset_ubv * 2).spectral_dist,
            rtol=0.01,
        )

    # - division

    def test_division_filter_spectrum_mean(self, filter_v, filterset_ubv):
        np.testing.assert_allclose(
            (filter_v / ac.vega_CALSPEC).mean_nm(), 558.681024, rtol=0.01
        )
        np.testing.assert_allclose(
            (filterset_ubv / ac.vega_CALSPEC).mean_nm(),
            [356.283866, 447.589411, 558.681024],
            rtol=0.01,
        )

    def test_division_spectrum_wavelength(self, filterset_ubv):
        np.testing.assert_allclose(
            (ac.sun_CALSPEC / ac.sun_CALSPEC.wavelength_nm).mean_nm(), 670.9781529, rtol=0.01
        )
        np.testing.assert_allclose(
            (filterset_ubv / filterset_ubv.wavelength_nm).mean_nm(),
            [359.158258, 438.480057, 548.890305],
            rtol=0.01,
        )

    # - normalization

    def test_normalization(self, filter_v, filterset_ubv):
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC, (filter_v * 2).normalize())[0],
            ac.observe(ac.vega_CALSPEC, filter_v)[0],
            rtol=0.01,
        )
        np.testing.assert_allclose(
            ac.observe(ac.vega_CALSPEC, (filterset_ubv * 2).normalize()).spectral_dist,
            ac.observe(ac.vega_CALSPEC, filterset_ubv).spectral_dist,
            rtol=0.01,
        )
