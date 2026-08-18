import numpy as np

import astrocolor as ac

# === Arithmetic Operations Tests ===

# - addition

def test_addition_spectrum():
    np.testing.assert_allclose(
        (ac.vega_CALSPEC + ac.vega_CALSPEC).spectral_dist,
        (ac.vega_CALSPEC * 2).spectral_dist,
        rtol=0.01,
    )

# - multiplication

def test_multiplication_filter_spectrum_mean(v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
    np.testing.assert_allclose(
        (v_filter * ac.vega_CALSPEC).mean_nm(), 544.601418, rtol=0.01
    )  # 544.543 in SVO Filter Profile Service
    np.testing.assert_allclose(
        (ubv_filterset * ac.vega_CALSPEC).mean_nm(),
        [366.764603, 435.741381, 544.601418],
        rtol=0.01,
    )

def test_multiplication_observation(v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
    np.testing.assert_allclose(
        ac.get_photometry(ac.vega_CALSPEC * 2, v_filter)[0],
        ac.get_photometry(ac.vega_CALSPEC, v_filter)[0] * 2,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        ac.get_photometry(ac.vega_CALSPEC * 2, ubv_filterset).spectral_dist,
        ac.get_photometry(ac.vega_CALSPEC, ubv_filterset * 2).spectral_dist,
        rtol=0.01,
    )

# - division

def test_zero_division_error(v_filter: ac.Filter):
    np.testing.assert_equal((v_filter / 0).spectral_dist, np.full_like(v_filter.spectral_dist, np.inf))

def test_zero_division_by_zero_error():
    np.testing.assert_equal((ac.SpectralSet.stub() / 0).spectral_dist, [[0.]])

def test_division_filter_spectrum_mean(v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
    np.testing.assert_allclose(
        (v_filter / ac.vega_CALSPEC).mean_nm(), 558.681024, rtol=0.01
    )
    np.testing.assert_allclose(
        (ubv_filterset / ac.vega_CALSPEC).mean_nm(),
        [356.283866, 447.589411, 558.681024],
        rtol=0.01,
    )

def test_division_spectrum_wavelength(ubv_filterset: ac.FilterSet):
    np.testing.assert_allclose(
        (ac.sun_CALSPEC / ac.sun_CALSPEC.wavelength_nm).mean_nm(), 670.9781529, rtol=0.01
    )
    np.testing.assert_allclose(
        (ubv_filterset / ubv_filterset.wavelength_nm).mean_nm(),
        [359.158258, 438.480057, 548.890305],
        rtol=0.01,
    )

# - normalization

def test_normalization(v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
    np.testing.assert_allclose(
        ac.get_photometry(ac.vega_CALSPEC, (v_filter * 2).normalized())[0],
        ac.get_photometry(ac.vega_CALSPEC, v_filter)[0],
        rtol=0.01,
    )
    np.testing.assert_allclose(
        ac.get_photometry(ac.vega_CALSPEC, (ubv_filterset * 2).normalized()).spectral_dist,
        ac.get_photometry(ac.vega_CALSPEC, ubv_filterset).spectral_dist,
        rtol=0.01,
    )
