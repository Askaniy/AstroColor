from typing import cast

import numpy as np
import pytest

import astrocolor as ac
from astrocolor.errors import NonFiniteValuesWarning

# === Spectral Objects Creation & Properties Tests ===

class TestMonochromaticSpectrum:

    def test_spectrum_int(self):
        spectrum = ac.Spectrum.monochromatic(555)
        np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(spectrum.mean_nm(), 555, rtol=1e-10)
        np.testing.assert_allclose(spectrum.spectral_dist, [0., 0.2, 0.], rtol=1e-10)

    def test_spectrum_float(self):
        spectrum = ac.Spectrum.monochromatic(555.5)
        np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(spectrum.mean_nm(), 555.5, rtol=1e-10)
        np.testing.assert_allclose(spectrum.spectral_dist, [0., 0.18, 0.02, 0.], rtol=1e-10)

    def test_spectral_cube_int(self):
        cube = ac.SpectralCube.monochromatic(555)
        np.testing.assert_allclose(cube.integrate(), np.ones((1, 1)), rtol=1e-10)
        np.testing.assert_allclose(cube.mean_nm(), 555, rtol=1e-10)

    def test_spectral_cube_float(self):
        cube = ac.SpectralCube.monochromatic(555.5)
        np.testing.assert_allclose(cube.integrate(), np.ones((1, 1)), rtol=1e-10)
        np.testing.assert_allclose(cube.mean_nm(), 555.5, rtol=1e-10)

    def test_intensity_int(self):
        spectrum = ac.Spectrum.monochromatic(555, intensity=np.pi)
        cube = ac.SpectralCube.monochromatic(555, intensity=[[np.pi]])
        np.testing.assert_allclose(spectrum.integrate(), np.pi, rtol=1e-10)
        np.testing.assert_allclose(cube.integrate(), [[np.pi]], rtol=1e-10)

    def test_intensity_float(self):
        spectrum = ac.Spectrum.monochromatic(555.5, intensity=np.pi)
        cube = ac.SpectralCube.monochromatic(555.5, intensity=[[np.pi]])
        np.testing.assert_allclose(spectrum.integrate(), np.pi, rtol=1e-10)
        np.testing.assert_allclose(cube.integrate(), [[np.pi]], rtol=1e-10)

    def test_emission_spectrum_1_line(self):
        nm = [555.5]
        sd = [1.]
        spectrum = ac.Spectrum(nm, sd, is_emission_spectrum=True)
        assert spectrum.spectral_dist[-1] == 0.
        assert spectrum.wavelength_nm[0] == 550
        assert spectrum.wavelength_nm[-1] == 565
        np.testing.assert_allclose(spectrum.integrate(), 1., rtol=1e-10)
        np.testing.assert_allclose(spectrum.spectral_dist, [0., 0.18, 0.02, 0.], rtol=1e-10)

    def test_emission_spectrum_2_lines(self):
        nm = [501, 602]
        sd = [2, 3]
        spectrum = ac.Spectrum(nm, sd, is_emission_spectrum=True)
        assert spectrum.spectral_dist[0] == 0.
        assert spectrum.spectral_dist[-1] == 0.
        assert spectrum.wavelength_nm[0] == 495
        assert spectrum.wavelength_nm[-1] == 610
        np.testing.assert_allclose(spectrum.integrate(), sum(sd), rtol=1e-10)

    def test_emission_spectral_cube(self):
        nm = [501, 602]
        sd = [[[0.5]], [[0.5]]]
        cube = ac.SpectralCube(nm, sd, is_emission_spectrum=True)
        assert cube.wavelength_nm[0] == 495
        assert cube.wavelength_nm[-1] == 610
        np.testing.assert_allclose(cube.integrate(), np.ones((1, 1)), rtol=1e-10)


class TestSpectrum:

    def test_spectrum_integrate(self):
        # Nominal total solar irradiance = 1361 W/m², according to IAU 2015 Resolution B3
        # https://en.wikipedia.org/wiki/Solar_constant
        np.testing.assert_allclose(ac.sun_calspec.integrate(), 1326.2373, rtol=0.001)
        np.testing.assert_allclose(ac.sun_tsis1_hsrs.integrate(), 1361.4892, rtol=0.001)
        # Apparent bolometric magnitude zero point = 2.518021002e-8 W/m², according to IAU 2015 Resolution B2
        # https://arxiv.org/abs/1510.06262
        np.testing.assert_allclose(ac.vega_calspec.integrate(), 3.0417e-08, rtol=0.001)


class TestNonFiniteValues:

    def test_non_finite_spectral_dist_is_replaced_with_zero(self):
        nm = np.arange(500., 541., 5.)
        sd = np.arange(1., nm.size + 1.)
        sd[[2, 4, 6]] = [np.nan, np.inf, -np.inf]

        with pytest.warns(NonFiniteValuesWarning, match='spectral_dist'):
            spectrum = ac.Spectrum(nm, sd, name='test')

        np.testing.assert_array_equal(spectrum.spectral_dist[[2, 4, 6]], 0.)
        np.testing.assert_allclose(np.delete(spectrum.spectral_dist, [2, 4, 6]), np.delete(sd, [2, 4, 6]))

    def test_non_finite_wavelengths_are_removed(self):
        nm = np.array([500., np.nan, 510., np.inf, 520.])
        sd = np.array([1., 2., 3., 4., 5.])
        std = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with pytest.warns(NonFiniteValuesWarning, match='wavelength'):
            spectrum = ac.Spectrum(nm, sd, std, name='test')

        assert np.all(np.isfinite(spectrum.wavelength_nm))
        np.testing.assert_allclose(spectrum.spectral_dist[spectrum.wavelength_nm == 510], 3.)
        assert spectrum.standard_deviation is not None
        np.testing.assert_allclose(spectrum.standard_deviation[spectrum.wavelength_nm == 510], 0.3)

    def test_all_non_finite_wavelengths_create_a_stub(self):
        with pytest.warns(NonFiniteValuesWarning, match='wavelength'):
            spectrum = ac.Spectrum([np.nan, np.inf], [1., 2.], name='test')

        np.testing.assert_array_equal(spectrum.wavelength_nm, [555])
        np.testing.assert_array_equal(spectrum.spectral_dist, [0.])

    def test_non_finite_photospectral_dist_is_replaced_with_zero(self, ubv_filterset: ac.FilterSet):
        with pytest.warns(NonFiniteValuesWarning, match='spectral_dist'):
            spectrum = ac.Photospectrum(ubv_filterset, [1., np.nan, np.inf], name='test')

        np.testing.assert_array_equal(spectrum.spectral_dist, [1., 0., 0.])

    def test_non_finite_monochromatic_intensity_is_replaced_with_zero(self):
        with pytest.warns(NonFiniteValuesWarning, match='intensity'):
            spectrum = ac.Spectrum.monochromatic(555., intensity=np.inf)

        np.testing.assert_array_equal(spectrum.spectral_dist, [0., 0., 0.])

    @pytest.mark.parametrize('non_finite', [np.nan, np.inf, -np.inf])
    def test_non_finite_standard_deviation_is_replaced_with_zero(self, non_finite: float):
        nm = np.arange(500., 601., 5.)
        std = np.full(nm.size, 0.1)
        std[3] = non_finite

        with pytest.warns(NonFiniteValuesWarning, match='standard deviation'):
            spectrum = ac.Spectrum(nm, np.ones(nm.size), std, name='test')

        assert spectrum.standard_deviation is not None
        np.testing.assert_allclose(cast(float, spectrum.standard_deviation[3]), 0.)
        np.testing.assert_allclose(np.delete(spectrum.standard_deviation, 3), 0.1)
        assert spectrum.covariance_matrix is not None
        assert np.all(np.isfinite(spectrum.covariance_matrix))
