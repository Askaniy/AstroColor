from pathlib import Path

import numpy as np
import pytest

import astrocolor as ac
from astrocolor import standard_spectra


@pytest.mark.parametrize(
    ('name', 'source', 'expected_name'),
    (
        ('Sun', 'CALSPEC', 'Sun [CALSPEC]'),
        ('Sun', 'TSIS-1 HSRS', 'Sun [TSIS-1 HSRS]'),
        ('Vega', 'CALSPEC', 'Vega [CALSPEC]'),
    ),
)
def test_all_sources_are_available(name: str, source: str, expected_name: str):
    spectrum = ac.get_standard_spectrum(name, source)

    assert isinstance(spectrum, ac.Spectrum)
    assert spectrum.name == expected_name
    assert np.all(np.diff(spectrum.wavelength_nm) > 0)
    assert np.all(np.isfinite(spectrum.spectral_dist))


def test_default_sources():
    assert ac.get_standard_spectrum('Sun').name == 'Sun [TSIS-1 HSRS]'
    assert ac.get_standard_spectrum('Vega').name == 'Vega [CALSPEC]'


def test_names_and_sources_are_case_and_space_insensitive():
    spectrum = ac.get_standard_spectrum('  sun  ', ' tsis-1   hsrs ')

    assert spectrum.name == 'Sun [TSIS-1 HSRS]'


def test_getter_returns_independent_copies():
    first = ac.get_standard_spectrum('Sun', 'CALSPEC')
    second = ac.get_standard_spectrum('Sun', 'CALSPEC')
    original_nm = second.wavelength_nm.copy()
    original_sd = second.spectral_dist.copy()

    assert first is not second
    assert not np.shares_memory(first.wavelength_nm, second.wavelength_nm)
    assert not np.shares_memory(first.spectral_dist, second.spectral_dist)

    first.wavelength_nm[:] = 0
    first.spectral_dist[:] = 0.
    np.testing.assert_array_equal(second.wavelength_nm, original_nm)
    np.testing.assert_array_equal(second.spectral_dist, original_sd)
    np.testing.assert_array_equal(
        ac.get_standard_spectrum('Sun', 'CALSPEC').wavelength_nm,
        original_nm,
    )
    np.testing.assert_array_equal(
        ac.get_standard_spectrum('Sun', 'CALSPEC').spectral_dist,
        original_sd,
    )


def test_unknown_name_reports_available_spectra():
    with pytest.raises(ValueError, match=r'Unknown standard spectrum .*Sun.*Vega'):
        _ = ac.get_standard_spectrum('Moon')


def test_unknown_source_reports_available_sources():
    with pytest.raises(ValueError, match=r'Unknown source .*Sun.*CALSPEC.*TSIS-1 HSRS'):
        _ = ac.get_standard_spectrum('Sun', 'JWST')


def test_missing_file_names_source_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(standard_spectra, '_DATA_FOLDER', tmp_path)
    standard_spectra.clear_cache()
    path = tmp_path / 'Sun_CALSPEC.npz'

    with pytest.raises(FileNotFoundError) as error:
        _ = ac.get_standard_spectrum('Sun', 'CALSPEC')

    message = str(error.value)
    assert 'Sun' in message
    assert 'CALSPEC' in message
    assert str(path) in message


def test_templates_are_cached_and_cache_can_be_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    standard_spectra.clear_cache()
    first = ac.get_standard_spectrum('Sun', 'CALSPEC')

    # The cached template is served even when the data folder is unavailable.
    monkeypatch.setattr(standard_spectra, '_DATA_FOLDER', tmp_path)
    cached = ac.get_standard_spectrum('Sun', 'CALSPEC')
    np.testing.assert_array_equal(cached.spectral_dist, first.spectral_dist)

    standard_spectra.clear_cache()
    with pytest.raises(FileNotFoundError):
        _ = ac.get_standard_spectrum('Sun', 'CALSPEC')


def test_mutating_a_returned_spectrum_does_not_affect_white_points_or_normalization():
    white_point = ac.ColorSystem.supported_white_points['Sun']
    spectrum = ac.get_standard_spectrum('Sun', 'TSIS-1 HSRS')
    spectrum.spectral_dist[:] = 0.

    assert ac.ColorSystem.supported_white_points['Sun'] is white_point
    assert white_point.same_chromaticity(
        ac.WhitePoint.from_spectrum(
            'Sun-like',
            ac.get_standard_spectrum('Sun', 'TSIS-1 HSRS'),
        )
    )

    nm = np.arange(500, 601, 5.)
    sd = np.ones_like(nm)
    obj = ac.CelestialObject(
        name='test',
        wavelength_nm=nm,
        spectral_dist=sd,
        is_reflecting_sunlight=True,
    )
    expected = ac.Spectrum(nm, sd) / ac.scale_to_match_value(
        ac.get_standard_spectrum('Sun', 'TSIS-1 HSRS'),
        ac.Filter.get('Generic/Bessell.V'),
    )
    assert obj.spectral_data_unscaled is not None
    np.testing.assert_allclose(
        obj.spectral_data_unscaled.spectral_dist,
        expected.spectral_dist,
    )
