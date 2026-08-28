import numpy as np

import astrocolor as ac

# === ReadMe Examples Tests ===

def test_photometry():

    # Block 1
    spectrum = ac.Spectrum(
        wavelength_nm=[400, 500, 600, 700],
        spectral_dist=[1, 2, 3, 4]
    )
    bessell_V = ac.Filter.get('Generic/Bessell.V')
    flux_value, flux_error = ac.get_photometry(spectrum, bessell_V)
    assert flux_value == 2.5167967171270256
    assert flux_error is None

    # Block 2
    bessell_BVR = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
    photospectrum_BVR = ac.Photospectrum(
        filter_set=bessell_BVR,
        spectral_dist=[1, 2, 3]
    )
    sloan_gr = ac.FilterSet.get('SLOAN/SDSS.g', 'SLOAN/SDSS.r')
    photospectrum = ac.get_photometry(photospectrum_BVR, sloan_gr)
    np.testing.assert_allclose(photospectrum.spectral_dist, [1.23743753, 2.71275724])

def test_color():
    color_xyz = ac.ColorPoint.from_spectral_data(ac.sun_CALSPEC)
    color_system = ac.ColorSystem('sRGB', 'Illuminant E') # recommended
    color_rgb = color_xyz.to_color_system(color_system)
    color_rgb.gamma_correction = True
    color_rgb.maximize_brightness = True
    color_html = color_rgb.to_html()
    assert color_html == '#effeff'

def test_models():
    bb_7000K = ac.get_spectrometry(ac.BlackBodyModel(7000), [400, 450])
    np.testing.assert_equal(bb_7000K.wavelength_nm, [400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450])
    np.testing.assert_allclose(
        bb_7000K.spectral_dist,
        [
            68731.39230746, 68848.04077054, 68911.83231492, 68925.20601463,
            68890.57885667, 68810.33686959, 68686.82746675, 68522.35290187,
            68319.16473884, 68079.45924271, 67805.37360451
        ]
    )
