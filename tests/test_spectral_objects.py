import numpy as np

import astrocolor as ac

# === Spectral Objects Creation & Properties Tests ===

def test_spectrum_from_nm_float():
    spectrum = ac.Spectrum.monochromatic(555.5)
    np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
    np.testing.assert_allclose(spectrum.mean_nm(), 555.5, rtol=1e-10)

def test_spectrum_from_nm_integer():
    spectrum = ac.Spectrum.monochromatic(555)
    np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
    np.testing.assert_allclose(spectrum.mean_nm(), 555, rtol=1e-10)
