import numpy as np
import pytest

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
    assert flux_value == pytest.approx(2.5167967171270256)
    assert flux_error is None

    # Block 2
    bessell_BVR = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
    photospectrum_BVR = ac.Photospectrum(
        filter_set=bessell_BVR,
        spectral_dist=[1, 2, 3]
    )
    sloan_gr = ac.FilterSet.get('SLOAN/SDSS.g', 'SLOAN/SDSS.r')
    photospectrum = ac.get_photometry(photospectrum_BVR, sloan_gr)
    np.testing.assert_allclose(photospectrum.spectral_dist, [1.236852, 2.713022], rtol=0.001)

    # Direct spectral reconstruction (uses photospectrum_BVR from the previous block)
    reconstructed = ac.get_spectrometry(photospectrum_BVR, requested_wavelengths=[400, 700])
    assert isinstance(reconstructed, ac.ReconstructedSpectrum)
    assert 400 in reconstructed.wavelength_nm
    assert 700 in reconstructed.wavelength_nm

def test_color():
    color = ac.get_color(ac.sun_tsis1_hsrs)
    color_html = color.to_html()
    assert color_html == '#ecfeff'

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

def test_celestial_objects():
    mercury = ac.CelestialObject(
        name='Mercury',
        filter_set='Generic/Bessell',
        filters=['U', 'B', 'V', 'R'],
        sd_geometric=[0.087, 0.105, 0.142, 0.172],
        spherical_albedo=['Generic/Bessell.V', 0.07],
    )
    geometric, estimated = mercury.get_spectral_data('geometric')
    assert isinstance(geometric, ac.Photospectrum)
    assert estimated is False
    flux, _ = ac.get_photometry(geometric, ac.Filter.get('Generic/Bessell.V'))
    assert flux == pytest.approx(0.142, abs=0.001)

def test_photometric_models():
    hg = ac.photometric_models.HG({'G': 0.2})
    phase_integral = hg.phase_integral
    assert phase_integral is not None
    q, q_std = phase_integral
    assert q == pytest.approx(0.4268)
    assert q_std is None
    spherical, estimated = hg.estimate_spherical_albedo((0.14, 0.01))
    assert spherical == pytest.approx(0.14 * q)
    assert estimated is False

def test_color_output():
    color_obj = ac.get_color(ac.sun_tsis1_hsrs)
    linear_rgb = color_obj.spectral_dist  # internal linear sRGB/E
    display_rgb = color_obj.render()      # sRGB/E + gamma + maximize
    assert linear_rgb.shape == (3,)
    assert display_rgb.shape == (3,)

    display_settings = ac.DisplaySettings(
        color_system=ac.ColorSystem('sRGB', 'Sun'),
        maximize_brightness=False,
        scale_factor=0.5
    )
    display_rgb = color_obj.render(display_settings)
    assert display_rgb.shape == (3,)

def test_custom_color_standards():
    custom_white = ac.WhitePoint.from_xy('Custom D50', 0.34567, 0.35850)
    spectral_white = ac.WhitePoint.from_spectrum('Sun-like', ac.sun_tsis1_hsrs)
    assert spectral_white.same_chromaticity(ac.ColorSystem.supported_white_points['Sun'])
    custom_space = ac.ColorSpace(
        name='Custom sRGB',
        primaries=ac.ColorSystem.supported_color_spaces['sRGB'].primaries,
        white_point=custom_white,
        transfer_function=ac.transfer_functions.srgb,
    )
    custom_system = ac.ColorSystem(custom_space)
    assert custom_system.internal_white_point is custom_white

def test_gamma_corrected_conversion():
    srgb_d65 = ac.ColorSystem('sRGB')
    srgb_e = ac.ColorSystem('sRGB', 'Illuminant E')
    encoded_color = ac.ColorPoint(
        [0.5, 0.25, 0.1],
        color_system=srgb_d65,
        gamma_corrected=True,
    )
    linear_rgb = encoded_color.to_color_system(srgb_e).spectral_dist
    display_rgb = srgb_e.encode(linear_rgb)
    expected = srgb_e.inv_matrix @ srgb_d65.matrix @ srgb_d65.decode([0.5, 0.25, 0.1])
    np.testing.assert_allclose(linear_rgb, expected)
    np.testing.assert_allclose(display_rgb, srgb_e.encode(expected))
