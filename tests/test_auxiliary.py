import numpy as np
import pytest

from astrocolor.auxiliary import (
    color_indices_parser,
    get_extremal_grid_endpoints,
    linear_interp,
    mag2irradiance,
    parse_value_std,
    repr_generator,
    repr_generator_1D,
    repr_generator_2D,
    spectral_binning,
    uniform_grid,
)
from astrocolor.core import nm_step, wavelength_nm_dtype

np.random.seed(42)


# === Auxiliary Utilities Tests ===


class TestWavelengthGrids:

    def test_grid_reversibility(self):
        old_grid = uniform_grid(400, 600, nm_step, wavelength_nm_dtype)
        nm_min, nm_max = get_extremal_grid_endpoints(old_grid)
        new_grid = uniform_grid(nm_min, nm_max, nm_step, wavelength_nm_dtype)
        np.testing.assert_equal(old_grid, new_grid)


class TestParsing:

    def test_sd_scalar(self):
        np.testing.assert_equal(parse_value_std(0.202), (0.202, None))

    def test_sd_two_values(self):
        np.testing.assert_equal(
            parse_value_std([0.202, 0.0665]), (0.202, 0.0665)
        )

    def test_sd_three_positive(self):
        np.testing.assert_equal(
            parse_value_std([0.202, 0.084, 0.049]), (0.202, 0.0665)
        )

    def test_sd_mixed_sign(self):
        np.testing.assert_equal(
            parse_value_std([0.202, +0.084, -0.049]), (0.202, 0.0665)
        )


class TestColorIndices:

    def test_1_index(self):
        # (120347) Salacia by Stansberry et al. 2012
        input = {'V-I': 0.87}
        # expected = {'V': 1., 'I': 1.-0.87}
        expected = (
            ('V', 'I'),
            mag2irradiance(np.array((0., -0.87))),
            None
        )
        output = color_indices_parser(input)
        np.testing.assert_equal(output, expected)

    def test_3_indices(self):
        # (120347) Salacia from MBOSS
        input = {'B-V': 0.664, 'V-R': 0.403, 'R-I': 0.433}
        # expected = {'B': 1., 'V': 1.-0.664, 'R': 1.-0.664-0.403, 'I': 1.-0.664-0.403-0.433}
        expected = (
            ('B', 'V', 'R', 'I'),
            mag2irradiance(np.array((0., -0.664, -0.664-0.403, -0.664-0.403-0.433))),
            None
        )
        output = color_indices_parser(input)
        np.testing.assert_equal(output, expected)

    def test_1_index_std(self):
        # (120347) Salacia by Stansberry et al. 2012
        input = {'V-I': [0.87, 0.01]}
        # expected = {'V': [1., 0.], 'I': [1.-0.87, 0.]}
        expected = (
            ('V', 'I'),
            mag2irradiance(np.array((0., -0.87))),
            (0.008823506076353184, 0.005885815833609801) # not real values! output reference of working state
        )
        output = color_indices_parser(input)
        np.testing.assert_equal(output, expected)

    def test_3_indices_std(self):
        # (120347) Salacia from MBOSS
        input = {'B-V': [0.664, 0.098], 'V-R': [0.403, 0.061], 'R-I': [0.433, 0.118]}
        # expected = {'B': [1., 0.], 'V': [1.-0.664, 0.], 'R': [1.-0.664-0.403, 0.], 'I': [1.-0.664-0.403-0.433, 0.]}
        expected = (
            ('B', 'V', 'R', 'I'),
            mag2irradiance(np.array((0., -0.664, -0.664-0.403, -0.664-0.403-0.433))),
            (0.08926846095326758, 0.024610219649774723, 0.14580865605424098, 0.37416765317580036)  # not real values! output reference of working state
        )
        output = color_indices_parser(input)
        np.testing.assert_equal(output, expected)


# TODO: add test for the spectrum generator from the slope
# (120347) Salacia by Pinilla-Alonso et al. 2008
# input = {'start': 520, 'stop': 860, 'percent_per_100nm': [12.6, 2.0]}


class TestSpectralBinning:

    def test_spectral_binning(self):
        nm0_len = 100
        nm0 = np.sort(
            np.linspace(402, 650, nm0_len) + np.random.normal(0, 5, nm0_len)
        )
        br0 = nm0 / 100
        step = 5  # nm
        nm1 = np.arange(400, 700, step)
        nm0_diff = np.diff(nm0)
        br1, _ = spectral_binning(nm0, br0, None, nm1, step, nm0_diff)
        np.testing.assert_allclose(br1, nm1 / 100, rtol=0.1)


class TestLinearInterp:

    # Basic 1D interpolation (no extrapolation)
    def test_basic_1d_interpolation(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x1 = np.array([0.5, 1.5, 2.5])
        y1 = linear_interp(x0, y0, x1)
        expected = np.array([0.5, 2.5, 6.5])
        np.testing.assert_allclose(y1, expected)
        # Compare with np.interp() for the same region
        np.testing.assert_allclose(y1, np.interp(x1, x0, y0))

    # Extrapolation mode='nearest' (constant)
    def test_extrapolation_nearest(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_ext = np.array([-1.0, 0.5, 3.5])
        y_const = linear_interp(x0, y0, x_ext, extrap_mode='nearest')
        np.testing.assert_allclose(y_const[0], y0[0])    # left extrapolation  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_const[1], 0.5)      # interior point  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_const[2], y0[-1])   # right extrapolation  # pyright: ignore[reportAny]

    # Extrapolation mode='linear'
    def test_extrapolation_linear(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_ext = np.array([-1.0, 0.5, 3.5])
        y_lin = linear_interp(x0, y0, x_ext, extrap_mode='linear')
        slope_first = (y0[1] - y0[0]) / (x0[1] - x0[0])     # pyright: ignore[reportAny]
        slope_last = (y0[-1] - y0[-2]) / (x0[-1] - x0[-2])  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_lin[0], y0[0] + slope_first * (-1.0 - x0[0]))  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_lin[2], y0[-1] + slope_last * (3.5 - x0[-1]))  # pyright: ignore[reportAny]

    # Multidimensional y0: 2D array with nearest extrapolation
    def test_multidim_2d_nearest(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        y0_2d = np.column_stack([y0, y0 * 2, y0 * 3])
        x_multi = np.array([0.5, 1.5, -0.5, 3.5])
        y_multi_const = linear_interp(x0, y0_2d, x_multi, extrap_mode='nearest')
        np.testing.assert_allclose(y_multi_const[2], y0_2d[0])   # left ext  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_multi_const[3], y0_2d[-1])  # right ext  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_multi_const[0, 0], 0.5)     # interior  # pyright: ignore[reportAny]

    # Multidimensional y0: 3D array with linear extrapolation
    def test_multidim_3d_linear(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0_3d = np.ones((4, 2, 3)) * np.arange(4)[:, None, None]
        x_multi = np.array([0.5, 1.5, -0.5, 3.5])
        y_3d_lin = linear_interp(x0, y0_3d, x_multi, extrap_mode='linear')
        np.testing.assert_allclose(y_3d_lin[2], -0.5)   # left ext  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_3d_lin[3], 3.5)    # right ext  # pyright: ignore[reportAny]

    # Edge cases
    def test_edge_two_point_x0(self):
        x0 = np.array([0.0, 2.0])
        y0 = np.array([10.0, 20.0])
        x1 = np.array([0.0, 1.0, 2.0, -1.0, 3.0])
        y_lin = linear_interp(x0, y0, x1, extrap_mode='linear')
        slope = (20 - 10) / (2 - 0)
        np.testing.assert_allclose(y_lin[0], 10.0)  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_lin[1], 15.0)  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_lin[3], 10.0 + slope * (-1.0 - 0.0))  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_lin[4], 20.0 + slope * (3.0 - 2.0))   # pyright: ignore[reportAny]

    def test_edge_all_points_outside(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_ext = np.array([-0.5, 3.5])
        slope_first = (y0[1] - y0[0]) / (x0[1] - x0[0])     # pyright: ignore[reportAny]
        slope_last = (y0[-1] - y0[-2]) / (x0[-1] - x0[-2])  # pyright: ignore[reportAny]
        y_out = linear_interp(x0, y0, x_ext, extrap_mode='linear')
        assert y_out.shape == (2,)
        np.testing.assert_allclose(y_out[0], y0[0] + slope_first * (-0.5 - x0[0]))  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_out[1], y0[-1] + slope_last * (3.5 - x0[-1]))  # pyright: ignore[reportAny]

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
        np.testing.assert_allclose(y_single[0], 0.5)  # pyright: ignore[reportAny]

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
        slope_left = (y0[1] - y0[0]) / (20 - 10)  # pyright: ignore[reportAny]
        slope_right = (y0[2] - y0[1]) / (30 - 20)  # pyright: ignore[reportAny]
        y_lin = linear_interp(x0, y0, x_ext, extrap_mode='linear')
        np.testing.assert_allclose(y_lin[0], y0[0] + slope_left * (5 - 10))  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_lin[1], y0[2] + slope_right * (35 - 30))  # pyright: ignore[reportAny]

    def test_edge_high_dim_singular(self):
        x0 = np.linspace(0, 4, 5)
        y0 = np.random.rand(5, 1, 2, 1, 3)
        x_test = np.array([-0.5, 2.5, 4.5])
        y_const = linear_interp(x0, y0, x_test, extrap_mode='nearest')
        assert y_const.shape == (3, 1, 2, 1, 3)
        np.testing.assert_allclose(y_const[0], y0[0])  # pyright: ignore[reportAny]
        np.testing.assert_allclose(y_const[2], y0[-1])  # pyright: ignore[reportAny]

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


class TestReprGenerator1D:
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
            _ = repr_generator_1D(arr_2d)


class TestReprGenerator2D:
    """ Tests for the repr_generator_2D function. """

    def test_single_row(self):
        arr = np.array([[1, 2, 3]])
        result = repr_generator_2D(arr)
        assert result == '[\n\t[1, 2, 3]\n]'

    def test_two_rows(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = repr_generator_2D(arr)
        expected = '[\n\t[1.000, 2.000],\n\t[3.000, 4.000]\n]'
        assert result == expected

    def test_three_rows(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = repr_generator_2D(arr)
        expected = '[\n\t[1.000, 2.000],\n\t[3.000, 4.000],\n\t[5.000, 6.000]\n]'
        assert result == expected

    def test_four_rows(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = repr_generator_2D(arr)
        expected = '[\n\t[1.000, 2.000],\n\t[3.000, 4.000],\n\t...\n\t[7.000, 8.000]\n]'
        assert result == expected

    def test_dimensional_error(self):
        arr_1d = np.array([1, 2, 3])
        with pytest.raises(ValueError, match='must be 2D'):
            _ = repr_generator_2D(arr_1d)


class TestReprGenerator:
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
        expected = '[\n\t[42]\n]'
        assert result == expected
