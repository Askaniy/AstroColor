import numpy as np
import pytest
import astrocolor as ac
from astrocolor.auxiliary import (
    repr_generator,
    repr_generator_1D,
    repr_generator_2D,
    linear_interp,
    spectral_binning,
    parse_value_std
)

np.random.seed(42)


# === Auxiliary Utilities Tests ===


class TestParsing():

    def test_sd_parsing_scalar(self):
        np.testing.assert_equal(parse_value_std(0.202), (0.202, None))

    def test_sd_parsing_two_values(self):
        np.testing.assert_equal(
            parse_value_std([0.202, 0.0665]), (0.202, 0.0665)
        )

    def test_sd_parsing_three_positive(self):
        np.testing.assert_equal(
            parse_value_std([0.202, 0.084, 0.049]), (0.202, 0.0665)
        )

    def test_sd_parsing_mixed_sign(self):
        np.testing.assert_equal(
            parse_value_std([0.202, +0.084, -0.049]), (0.202, 0.0665)
        )


class TestSpectralBinning():
    """ Tests for the spectral binning function """

    def test_spectral_binning(self):
        nm0_len = 100
        nm0 = np.sort(
            np.linspace(402, 650, nm0_len) + np.random.normal(0, 5, nm0_len)
        )
        br0 = nm0 / 100
        step = 5  # nm
        nm1 = np.arange(400, 700, step)
        nm0_diff = np.diff(nm0)
        br1, std1 = spectral_binning(nm0, br0, None, nm1, step, nm0_diff)
        np.testing.assert_allclose(br1, nm1 / 100, rtol=0.1)


class TestLinearInterp():
    """ Tests for the linear interpolation function. """

    # Basic 1D interpolation (no extrapolation)
    def test_basic_1d_interpolation(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x1 = np.array([0.5, 1.5, 2.5])
        y1 = linear_interp(x0, y0, x1)
        expected = np.array([0.5, 2.5, 6.5])
        np.testing.assert_allclose(y1, expected)
        # Compare with numpy's built-in interp for the same region
        np.testing.assert_allclose(y1, np.interp(x1, x0, y0))

    # Extrapolation mode='nearest' (constant)
    def test_extrapolation_nearest(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_ext = np.array([-1.0, 0.5, 3.5])
        y_const = linear_interp(x0, y0, x_ext, extrap_mode='nearest')
        np.testing.assert_allclose(y_const[0], y0[0])   # left extrapolation
        np.testing.assert_allclose(y_const[1], 0.5)      # interior point
        np.testing.assert_allclose(y_const[2], y0[-1])   # right extrapolation

    # Extrapolation mode='linear'
    def test_extrapolation_linear(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_ext = np.array([-1.0, 0.5, 3.5])
        y_lin = linear_interp(x0, y0, x_ext, extrap_mode='linear')
        slope_first = (y0[1] - y0[0]) / (x0[1] - x0[0])
        slope_last = (y0[-1] - y0[-2]) / (x0[-1] - x0[-2])
        np.testing.assert_allclose(y_lin[0], y0[0] + slope_first * (-1.0 - x0[0]))
        np.testing.assert_allclose(y_lin[2], y0[-1] + slope_last * (3.5 - x0[-1]))

    # Multidimensional y0: 2D array with nearest extrapolation
    def test_multidim_2d_nearest(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        y0_2d = np.column_stack([y0, y0 * 2, y0 * 3])
        x_multi = np.array([0.5, 1.5, -0.5, 3.5])
        y_multi_const = linear_interp(x0, y0_2d, x_multi, extrap_mode='nearest')
        np.testing.assert_allclose(y_multi_const[2], y0_2d[0])   # left ext
        np.testing.assert_allclose(y_multi_const[3], y0_2d[-1])  # right ext
        np.testing.assert_allclose(y_multi_const[0, 0], 0.5)     # interior

    # Multidimensional y0: 3D array with linear extrapolation
    def test_multidim_3d_linear(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0_3d = np.ones((4, 2, 3)) * np.arange(4)[:, None, None]
        x_multi = np.array([0.5, 1.5, -0.5, 3.5])
        y_3d_lin = linear_interp(x0, y0_3d, x_multi, extrap_mode='linear')
        np.testing.assert_allclose(y_3d_lin[2], -0.5)   # left ext
        np.testing.assert_allclose(y_3d_lin[3], 3.5)    # right ext

    # Edge cases
    def test_edge_two_point_x0(self):
        x0 = np.array([0.0, 2.0])
        y0 = np.array([10.0, 20.0])
        x1 = np.array([0.0, 1.0, 2.0, -1.0, 3.0])
        y_lin = linear_interp(x0, y0, x1, extrap_mode='linear')
        slope = (20 - 10) / (2 - 0)
        np.testing.assert_allclose(y_lin[0], 10.0)
        np.testing.assert_allclose(y_lin[1], 15.0)
        np.testing.assert_allclose(y_lin[3], 10.0 + slope * (-1.0 - 0.0))
        np.testing.assert_allclose(y_lin[4], 20.0 + slope * (3.0 - 2.0))

    def test_edge_all_points_outside(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_ext = np.array([-0.5, 3.5])
        slope_first = (y0[1] - y0[0]) / (x0[1] - x0[0])
        slope_last = (y0[-1] - y0[-2]) / (x0[-1] - x0[-2])
        y_out = linear_interp(x0, y0, x_ext, extrap_mode='linear')
        assert y_out.shape == (2,)
        np.testing.assert_allclose(y_out[0], y0[0] + slope_first * (-0.5 - x0[0]))
        np.testing.assert_allclose(y_out[1], y0[-1] + slope_last * (3.5 - x0[-1]))

    def test_edge_empty_x1(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        y_empty = linear_interp(x0, y0, np.array([]))
        assert y_empty.size == 0

    def test_edge_single_interior_point(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_single = np.array([0.5])
        y_single = linear_interp(x0, y0, x_single)
        assert y_single.shape == (1,)
        np.testing.assert_allclose(y_single[0], 0.5)

    def test_edge_exact_grid_points(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_exact = np.array([0.0, 2.0])
        y_exact = linear_interp(x0, y0, x_exact)
        np.testing.assert_allclose(y_exact, [0.0, 4.0])

    def test_edge_multidim_all_extrapolation(self):
        x0 = np.array([10.0, 20.0, 30.0])
        y0 = np.ones((3, 4, 5)) * np.arange(3)[:, None, None]
        x_ext = np.array([5.0, 35.0])
        slope_left = (y0[1] - y0[0]) / (20 - 10)
        slope_right = (y0[2] - y0[1]) / (30 - 20)
        y_lin = linear_interp(x0, y0, x_ext, extrap_mode='linear')
        np.testing.assert_allclose(y_lin[0], y0[0] + slope_left * (5 - 10))
        np.testing.assert_allclose(y_lin[1], y0[2] + slope_right * (35 - 30))

    def test_edge_high_dim_singular(self):
        x0 = np.linspace(0, 4, 5)
        y0 = np.random.rand(5, 1, 2, 1, 3)
        x_test = np.array([-0.5, 2.5, 4.5])
        y_const = linear_interp(x0, y0, x_test, extrap_mode='nearest')
        assert y_const.shape == (3, 1, 2, 1, 3)
        np.testing.assert_allclose(y_const[0], y0[0])
        np.testing.assert_allclose(y_const[2], y0[-1])

    # Existing test: full-range interpolation with both modes
    def test_full_range_interpolation(self):
        x0 = np.array([3, 5, 8, 9, 11])
        y0 = np.array([3, 4, 7, 0, 2])
        x1 = np.arange(0, 15, 0.5)
        y_nearest = linear_interp(x0, y0, x1, extrap_mode='nearest')
        y_linear = linear_interp(x0, y0, x1, extrap_mode='linear')
        expected_nearest = [
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.25, 3.5, 3.75,
            4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 3.5, 0.0, 0.5,
            1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]
        expected_linear = [
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
            4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 3.5, 0.0, 0.5,
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
        ]
        np.testing.assert_allclose(y_nearest, expected_nearest, rtol=1e-5)
        np.testing.assert_allclose(y_linear, expected_linear, rtol=1e-5)


class TestExtrapolation():

    def test_extrapolation_flat_spectrum(self):
        """ A flat spectrum should remain flat after extrapolation. """
        nm = np.arange(500, 701, 5)
        spectrum = ac.Spectrum(nm, np.ones_like(nm))
        np.testing.assert_equal(
            spectrum.determine_at_wavelengths(ac.visible_range, strictly=True).spectral_dist,
            np.ones(ac.visible_range.size),
        )

    def test_extrapolation_filter_set(self, ubv_filterset):
        extrapolated = ubv_filterset._determine_at_trusted_wavelengths(ac.visible_range)
        assert extrapolated.wavelength_nm.size == ac.visible_range.size

    def test_extrapolation_flat_photospectrum(self, ubv_filterset):
        """ A photospectrum with uniform magnitudes should remain flat after extrapolation. """
        photospectrum = ac.Photospectrum(ubv_filterset, (1, 1, 1), name='test photospectrum')
        np.testing.assert_allclose(
            photospectrum.determine_at_wavelengths(ac.visible_range, strictly=True).spectral_dist,
            np.ones(ac.visible_range.size),
        )


class TestReprGenerator1D():
    """ Tests for the repr_generator_1D function. """

    def test_int(self):
        arr = np.array([42])
        result = repr_generator_1D(arr)
        assert result == '[42]'

    def test_float(self):
        arr = np.array([42.])
        assert repr_generator_1D(arr) == '[42.000]'

    def test_two_elements(self):
        arr = np.array([0.5, 3.7])
        result = repr_generator_1D(arr)
        expected = '[0.500, 3.700]'
        assert result == expected

    def test_three_elements(self):
        arr = np.array([1.23456, -0.98765, 0.0])
        result = repr_generator_1D(arr)
        assert len(result.split(', ')) == 3
        assert result.startswith('[') and result.endswith(']')

    def test_four_elements(self):
        arr = np.array([42.7, -99.3, 55.1, 8])
        result = repr_generator_1D(arr)
        expected = '[42.700, -99.300, ..., 8.000]'
        assert result == expected

    def test_int_many_elements(self):
        arr = np.array([1, -2, 3, 4, -5])
        result = repr_generator_1D(arr)
        expected = '[1, -2, ..., -5]'
        assert result == expected

    def test_dimensional_error(self):
        arr_2d = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match='must be 1D'):
            repr_generator_1D(arr_2d)


class TestReprGenerator2D():
    """ Tests for the repr_generator_2D function. """

    def test_single_row(self):
        arr = np.array([[1, 2, 3]])
        result = repr_generator_2D(arr)
        assert result.replace('\n', '') == '[[1, 2, 3]]'

    def test_two_rows(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = repr_generator_2D(arr)
        lines = result.strip('[]\n').split('\n')
        assert len(lines) == 2
        # Each line should be a valid 1D representation
        for i in range(2):
            np.testing.assert_allclose(
                [float(x) for x in lines[i].strip().replace('[', '').replace(']', '').split(',')],
                arr[i, :]
            )

    def test_three_rows(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = repr_generator_2D(arr)
        lines = result.strip('[]\n').split('\n')
        assert len(lines) == 3

    def test_four_rows(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = repr_generator_2D(arr)
        expected = '[\n\t[1.000, 2.000]\n\t[3.000, 4.000]\n\t...\n\t[7.000, 8.000]\n]'
        assert result == expected

    def test_dimensional_error(self):
        arr_1d = np.array([1, 2, 3])
        with pytest.raises(ValueError, match='must be 2D'):
            repr_generator_2D(arr_1d)


class TestReprGenerator():
    """ Tests for the main repr_generator dispatching function. """

    def test_dispatches_to_1d(self):
        arr = np.array([0.5, 1.5])
        result = repr_generator(arr)
        assert '0' in result and '.' not in result.split('[')[-1].split(',')[0] or '.500' in result

    def test_dispatches_to_2d(self):
        arr = np.array([[0.5, 1.5], [2.5, 3.5]])
        result = repr_generator(arr)
        assert '\n' in result

    def test_other_dimensions_returns_generic(self):
        arr_3d = np.ones((2, 3, 4))
        result = repr_generator(arr_3d)
        assert '[3-dimensional array]' == result

    def test_empty_1d_array(self):
        arr = np.array([])
        result = repr_generator(arr)
        assert result == '[]'

    def test_single_element_2d(self):
        arr = np.array([[42]])
        result = repr_generator(arr)
        expected = '[\n[42]\n]'
        assert result == expected
