import numpy as np

import astrocolor as ac

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
