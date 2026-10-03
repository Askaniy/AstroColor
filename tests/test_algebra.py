import numpy as np
import pytest

import astrocolor as ac
from astrocolor.algebra import mul_error, mul_value

# === Arithmetic Operations Tests ===


class TestAddiction:

    def test_spectrum(self):
        np.testing.assert_allclose(
            (ac.vega_CALSPEC + ac.vega_CALSPEC).spectral_dist,
            (ac.vega_CALSPEC * 2).spectral_dist,
            rtol=0.01,
        )


class TestMultiplication:

    def test_0D_mul_value(self):
        assert mul_value(2, 2) == 4

    def test_1D_mul_value(self):
        np.testing.assert_allclose(mul_value([3, 3], 3), [9, 9])

    def test_0D_mul_error_both_have_errors(self):
        x = 10
        dx = 2
        y = 20
        dy = 5
        expected = (y * dx)**2 + (x * dy)**2
        result = mul_error(x, dx**2, y, dy**2)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_mul_error_only_x_has_error(self):
        x = 10
        dx = 2
        y = 20
        expected = (y * dx)**2
        result = mul_error(x, dx**2, y, None)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_mul_error_only_y_has_error(self):
        x = 10
        y = 20
        dy = 5
        expected = (x * dy)**2
        result = mul_error(x, None, y, dy**2)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_0D_mul_error_no_errors(self):
        result = mul_error(10, None, 20, None)
        assert result is None

    def test_1D_1D_mul_error_no_errors(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = mul_error(x, None, y, None)
        assert result is None

    def test_0D_1D_mul_error_no_errors(self):
        result = mul_error(3, None, np.array([1, 2, 3]), None)
        assert result is None

    def test_scalar_times_array_with_error(self):
        """ Multiplication of scalar by array with error """
        scalar = 3
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])  # diagonal covariance matrix
        result = mul_error(scalar, None, array, cov_array)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_array_times_scalar_with_error(self):
        """ Multiplication of array with error by scalar """
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])
        scalar = 5
        result = mul_error(array, cov_array, scalar, None)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_array_times_scalar_both_have_errors(self):
        """ Multiplication of array and scalar, both with errors """
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])
        scalar = 5
        scalar_var = 0.5
        result = mul_error(array, cov_array, scalar, scalar_var)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array + np.diag(array**2 * scalar_var)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_multiplication_with_correlation(self):
        """ Multiplication of two 1D arrays with correlation """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_x = np.array([[0.1, 0.02, 0.01],
                        [0.02, 0.2, 0.03],
                        [0.01, 0.03, 0.3]])
        cov_y = np.array([[0.4, 0.05, 0.02],
                        [0.05, 0.5, 0.06],
                        [0.02, 0.06, 0.6]])
        result = mul_error(x, cov_x, y, cov_y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        diag_y = np.diag(y)
        diag_x = np.diag(x)
        expected = diag_y @ cov_x @ diag_y + diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_only_x_has_error(self):
        """ Multiplication of arrays, only x has error """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_x = np.diag([0.1, 0.2, 0.3])
        result = mul_error(x, cov_x, y, None)
        assert isinstance(result, np.ndarray)
        diag_y = np.diag(y)
        expected = diag_y @ cov_x @ diag_y
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_only_y_has_error(self):
        """ Multiplication of arrays, only y has error """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_y = np.diag([0.4, 0.5, 0.6])
        result = mul_error(x, None, y, cov_y)
        assert isinstance(result, np.ndarray)
        diag_x = np.diag(x)
        expected = diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_incompatible_sizes_raises_error(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        with pytest.raises(ValueError):
            _ = mul_error(x, np.diag([0.1, 0.2, 0.3]), y, np.diag([0.4, 0.5]))

    def test_vector_covariance_input(self):
        """ Check that variance vector can be passed instead of matrix """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        var_x = np.array([0.1, 0.2, 0.3])
        var_y = np.array([0.4, 0.5, 0.6])
        result = mul_error(x, var_x, y, var_y)
        assert isinstance(result, np.ndarray)
        cov_x = np.diag(var_x)
        cov_y = np.diag(var_y)
        diag_y = np.diag(y)
        diag_x = np.diag(x)
        expected = diag_y @ cov_x @ diag_y + diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_multiplication_filter_spectrum_mean(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            (v_filter * ac.vega_CALSPEC).mean_nm(), 544.601418, rtol=0.01
        )  # 544.543 in SVO Filter Profile Service
        np.testing.assert_allclose(
            (ubv_filterset * ac.vega_CALSPEC).mean_nm(),
            [366.764603, 435.741381, 544.601418],
            rtol=0.01,
        )

    def test_multiplication_observation(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
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


class TestDivision:

    def test_zero_division_error(self, v_filter: ac.Filter):
        np.testing.assert_equal((v_filter / 0).spectral_dist, np.full_like(v_filter.spectral_dist, np.inf))

    def test_zero_division_by_zero_error(self):
        np.testing.assert_equal((ac.SpectralSet.stub() / 0).spectral_dist, [[0.]])

    def test_filter_spectrum_mean(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            (v_filter / ac.vega_CALSPEC).mean_nm(), 558.681024, rtol=0.01
        )
        np.testing.assert_allclose(
            (ubv_filterset / ac.vega_CALSPEC).mean_nm(),
            [356.283866, 447.589411, 558.681024],
            rtol=0.01,
        )

    def test_spectrum_wavelength(self, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            (ac.sun_CALSPEC / ac.sun_CALSPEC.wavelength_nm).mean_nm(), 670.9781529, rtol=0.01
        )
        np.testing.assert_allclose(
            (ubv_filterset / ubv_filterset.wavelength_nm).mean_nm(),
            [359.158258, 438.480057, 548.890305],
            rtol=0.01,
        )


class TestNormalization:

    def test_filter(self, v_filter: ac.Filter):
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_CALSPEC, (v_filter * 2).normalized())[0],
            ac.get_photometry(ac.vega_CALSPEC, v_filter)[0],
            rtol=0.01,
        )

    def test_filter_set(self, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_CALSPEC, (ubv_filterset * 2).normalized()).spectral_dist,
            ac.get_photometry(ac.vega_CALSPEC, ubv_filterset).spectral_dist,
            rtol=0.01,
        )
