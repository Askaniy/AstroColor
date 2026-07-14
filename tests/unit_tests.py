import unittest
import numpy as np
from astrocolor.auxiliary import (
    linear_interp,
    spectral_binning,
    parse_value_std
)
from astrocolor import (
    Spectrum,
    SpectralSet,
    SpectralCube,
    Photospectrum,
    PhotospectralSet,
    PhotospectralCube,
    ReconstructedSpectrum,
    ReconstructedSpectralSet,
    ReconstructedSpectralCube,
    spectral_reconstruction,
    Filter,
    FilterSet,
    observe,
    scale_spectrum,
    ColorSystem,
    ColorPoint,
    ColorLine,
    ColorImage,
    visible_range,
    sun_CALSPEC,
    vega_CALSPEC,
    BlackBodyModel
)

np.random.seed(42)



# === Filter & FilterSet Statistics Tests ===

class TestFilterStatistics(unittest.TestCase):
    """ Tests for mean_nm() and std_of_nm() on filters, spectra, and their sets. """

    def setUp(self):
        self.v = Filter.get('Generic_Bessell.V')
        self.ubv = FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V')

    # - mean_nm() tests

    def test_mean_nm_spectra(self):
        np.testing.assert_allclose(sun_CALSPEC.mean_nm(), 857.052056, rtol=0.01)
        np.testing.assert_allclose(vega_CALSPEC.mean_nm(), 510.428463, rtol=0.01)

    def test_mean_nm_filter(self):
        # 551.210 in SVO Filter Profile Service — our value is ~551.204
        np.testing.assert_allclose(self.v.mean_nm(), 551.204273, rtol=0.01)

    def test_mean_nm_filter_set(self):
        np.testing.assert_allclose(
            self.ubv.mean_nm(), [360.507105, 441.301389, 551.204273], rtol=0.01
        )

    # - std_of_nm() tests

    def test_std_of_nm_spectra(self):
        np.testing.assert_allclose(sun_CALSPEC.std_of_nm(), 468.978657, rtol=0.01)
        np.testing.assert_allclose(vega_CALSPEC.std_of_nm(), 353.430263, rtol=0.01)

    def test_std_of_nm_filter(self):
        np.testing.assert_allclose(self.v.std_of_nm(), 36.354015, rtol=0.01)

    def test_std_of_nm_filter_set(self):
        np.testing.assert_allclose(
            self.ubv.std_of_nm(), [21.932217, 35.816641, 36.354015], rtol=0.01
        )


# === Convolution Tests ===

class TestObservation(unittest.TestCase):
    """ Tests for observe() and spectral convolution. """

    def setUp(self):
        self.v = Filter.get('Generic_Bessell.V')
        self.ubv = FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V')

    # - stub / return-type tests

    def test_stub_and_convolution_possibility(self):
        self.assertIsInstance(observe(Spectrum.stub(), Filter.stub()), tuple)
        self.assertIsInstance(observe(Spectrum.stub(), FilterSet.stub()), Photospectrum)
        self.assertIsInstance(observe(SpectralSet.stub(), Filter.stub()), tuple)
        self.assertIsInstance(observe(SpectralSet.stub(), FilterSet.stub()), PhotospectralSet)
        self.assertIsInstance(observe(SpectralCube.stub(), Filter.stub()), tuple)
        self.assertIsInstance(observe(SpectralCube.stub(), FilterSet.stub()), PhotospectralCube)
        self.assertIsInstance(observe(Photospectrum.stub(), Filter.stub()), tuple)
        self.assertIsInstance(observe(Photospectrum.stub(), FilterSet.stub()), Photospectrum)
        self.assertIsInstance(observe(PhotospectralSet.stub(), Filter.stub()), tuple)
        self.assertIsInstance(observe(PhotospectralSet.stub(), FilterSet.stub()), PhotospectralSet)
        self.assertIsInstance(observe(PhotospectralCube.stub(), Filter.stub()), tuple)
        self.assertIsInstance(observe(PhotospectralCube.stub(), FilterSet.stub()), PhotospectralCube)

    # - convolution correctness

    def test_convolution(self):
        np.testing.assert_allclose(
            observe(vega_CALSPEC, self.v)[0], (vega_CALSPEC * self.v).integrate(), rtol=0.01
        )
        np.testing.assert_allclose(
            observe(vega_CALSPEC, self.ubv).spectral_dist,
            (vega_CALSPEC * self.ubv).integrate(),
            rtol=0.01,
        )

    # - zero-point calibration against Spanish Virtual Observatory

    def test_vega_system_zero_points(self):
        # TODO: check the agreement percent
        np.testing.assert_allclose(observe(vega_CALSPEC, self.v)[0], 3.62708e-11, rtol=0.0025)
        np.testing.assert_allclose(
            observe(vega_CALSPEC, self.ubv).spectral_dist,
            [4.089744e-11, 6.365467e-11, 3.623954e-11],
            rtol=0.035,
        )


# === Arithmetic Operations Tests ===

class TestArithmetic(unittest.TestCase):
    """ Tests for +, *, / operations between spectra and filters. """

    def setUp(self):
        self.v = Filter.get('Generic_Bessell.V')
        self.ubv = FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V')

    # - addition

    def test_addition_spectrum(self):
        np.testing.assert_allclose(
            (vega_CALSPEC + vega_CALSPEC).spectral_dist,
            (vega_CALSPEC * 2).spectral_dist,
            rtol=0.01,
        )

    # - multiplication

    def test_multiplication_filter_spectrum_mean(self):
        np.testing.assert_allclose(
            (self.v * vega_CALSPEC).mean_nm(), 544.601418, rtol=0.01
        )  # 544.543 in SVO Filter Profile Service
        np.testing.assert_allclose(
            (self.ubv * vega_CALSPEC).mean_nm(),
            [366.764603, 435.741381, 544.601418],
            rtol=0.01,
        )

    def test_multiplication_observation(self):
        np.testing.assert_allclose(
            observe(vega_CALSPEC * 2, self.v)[0],
            observe(vega_CALSPEC, self.v)[0] * 2,
            rtol=0.01,
        )
        np.testing.assert_allclose(
            observe(vega_CALSPEC * 2, self.ubv).spectral_dist,
            observe(vega_CALSPEC, self.ubv * 2).spectral_dist,
            rtol=0.01,
        )

    # - division

    def test_division_filter_spectrum_mean(self):
        np.testing.assert_allclose(
            (self.v / vega_CALSPEC).mean_nm(), 558.681024, rtol=0.01
        )
        np.testing.assert_allclose(
            (self.ubv / vega_CALSPEC).mean_nm(),
            [356.283866, 447.589411, 558.681024],
            rtol=0.01,
        )

    def test_division_spectrum_wavelength(self):
        np.testing.assert_allclose(
            (sun_CALSPEC / sun_CALSPEC.wavelength_nm).mean_nm(), 670.9781529, rtol=0.01
        )
        np.testing.assert_allclose(
            (self.ubv / self.ubv.wavelength_nm).mean_nm(),
            [359.158258, 438.480057, 548.890305],
            rtol=0.01,
        )

    # - normalization

    def test_normalization(self):
        np.testing.assert_allclose(
            observe(vega_CALSPEC, (self.v * 2).normalize())[0],
            observe(vega_CALSPEC, self.v)[0],
            rtol=0.01,
        )
        np.testing.assert_allclose(
            observe(vega_CALSPEC, (self.ubv * 2).normalize()).spectral_dist,
            observe(vega_CALSPEC, self.ubv).spectral_dist,
            rtol=0.01,
        )


# === Spectrum Creation & Properties Tests ===

class TestSpectrumCreation(unittest.TestCase):

    def test_spectrum_from_nm_float(self):
        spectrum = Spectrum.monochromatic(555.5)
        np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(spectrum.mean_nm(), 555.5, rtol=1e-10)

    def test_spectrum_from_nm_integer(self):
        spectrum = Spectrum.monochromatic(555)
        np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(spectrum.mean_nm(), 555, rtol=1e-10)

    def test_filter_edges(self):
        v = Filter.get('Generic_Bessell.V')
        self.assertEqual(v.spectral_dist[0], 0.)
        self.assertEqual(v.spectral_dist[-1], 0.)

    def test_filter_edges_extrapolated(self):
        v = Filter.get('Generic_Bessell.V')
        extrapolated_v = v.determine_at_wavelengths(visible_range)
        self.assertEqual(extrapolated_v.spectral_dist[0], 0.)
        self.assertEqual(extrapolated_v.spectral_dist[-1], 0.)

    def test_filter_system_getitem(self):
        ubv = FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V')
        v = Filter.get('Generic_Bessell.V')
        np.testing.assert_equal(ubv[2].mean_nm(), v.mean_nm())


# === Extrapolation Tests ===

class TestExtrapolation(unittest.TestCase):

    def test_extrapolation_flat_spectrum(self):
        """ A flat spectrum should remain flat after extrapolation. """
        nm = np.arange(500, 701, 5)
        spectrum = Spectrum(nm, np.ones_like(nm))
        np.testing.assert_equal(
            spectrum.determine_at_wavelengths(visible_range, strictly=True).spectral_dist,
            np.ones(visible_range.size),
        )

    def test_extrapolation_flat_photospectrum(self):
        """ A photospectrum with uniform magnitudes should remain flat after extrapolation. """
        ubv = FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V')
        photospectrum = Photospectrum(ubv, (1, 1, 1), name='test photospectrum')
        np.testing.assert_allclose(
            photospectrum.determine_at_wavelengths(visible_range, strictly=True).spectral_dist,
            np.ones(visible_range.size),
        )


# === Color System Tests ===

class TestColorSystem(unittest.TestCase):

    def test_color_system_reversibility(self):
        """ Converting a color point between systems should be reversible to machine precision. """
        srgb = ColorSystem('sRGB')
        xyz = ColorSystem('CIE 1931 XYZ')
        color0 = ColorPoint.from_spectral_data(sun_CALSPEC)  # in XYZ
        color1 = color0.to_color_system(srgb)
        color2 = color1.to_color_system(xyz)
        np.testing.assert_allclose(color0.to_array(), color2.to_array(), rtol=1e-13)

    def test_adaptation_white_point(self):
        """ CIE 1931 RGB should have Illuminant E white point """
        rgb = ColorSystem('CIE 1931 RGB')
        rgb_ = ColorSystem('CIE 1931 RGB', adaptation_white_point='Illuminant E')
        color0 = ColorPoint.from_spectral_data(sun_CALSPEC)  # in XYZ
        color1 = color0.to_color_system(rgb)
        color2 = color1.to_color_system(rgb_)
        np.testing.assert_allclose(color1.to_array(), color2.to_array(), rtol=1e-13)


# === Auxiliary Utilities Tests ===

class TestParsing(unittest.TestCase):

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

class TestSpectralBinning(unittest.TestCase):
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

class TestLinearInterp(unittest.TestCase):
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
        self.assertEqual(y_out.shape, (2,))
        np.testing.assert_allclose(y_out[0], y0[0] + slope_first * (-0.5 - x0[0]))
        np.testing.assert_allclose(y_out[1], y0[-1] + slope_last * (3.5 - x0[-1]))

    def test_edge_empty_x1(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        y_empty = linear_interp(x0, y0, np.array([]))
        self.assertEqual(y_empty.size, 0)

    def test_edge_single_interior_point(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0])
        y0 = np.array([0.0, 1.0, 4.0, 9.0])
        x_single = np.array([0.5])
        y_single = linear_interp(x0, y0, x_single)
        self.assertEqual(y_single.shape, (1,))
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
        self.assertEqual(y_const.shape, (3, 1, 2, 1, 3))
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


if __name__ == '__main__':
    unittest.main()
