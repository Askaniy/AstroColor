import os
import urllib.error as urle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astrocolor import Filter
from astrocolor.config import Config
from astrocolor.errors import FilterNetworkError
from astrocolor.filter_loader import (
    fetch_from_fps_raw,
    get_parameter,
    get_profile,
)

# === FilterObjects Statistics Tests ===

# Local file loading and SVO FPS fetching.
# All network-dependent tests use unittest.mock to avoid real HTTP requests.

# A narrow filter for testing
SAMPLE_XML_PATH = Path(__file__).parent/'Galileo.SSI.7270A.xml'
EXPECTED_N_POINTS = 7


class TestGetProfile:
    """ Tests for get_profile() function """

    def test_shape(self):
        """ Parsing the sample XML should return (7,) arrays for both columns """
        xml_content = ET.fromstring(SAMPLE_XML_PATH.read_bytes())
        nm, sd = get_profile(xml_content, 'NonExistent/Filter')
        assert isinstance(nm, np.ndarray), 'Wavelength must be a numpy array'
        assert isinstance(sd, np.ndarray), 'Transmission must be a numpy array'
        assert len(nm) == EXPECTED_N_POINTS, f'Expected {EXPECTED_N_POINTS} points, got {len(nm)}'
        assert len(sd) == EXPECTED_N_POINTS

    def test_values(self):
        """ First data point should match the XML """
        xml_content = ET.fromstring(SAMPLE_XML_PATH.read_bytes())
        nm, sd = get_profile(xml_content, 'NonExistent/Filter')
        np.testing.assert_equal(cast(float, nm[0]), 7200.0, f'First wavelength: {nm[0]} != 7200.0')
        np.testing.assert_equal(cast(float, sd[0]), 0.002494, f'First transmission: {sd[0]} != 0.002494')

    def test_invalid_xml(self):
        """ Invalid XML should raise FilterNetworkError """
        with pytest.raises(FilterNetworkError, match='No RESOURCE type="results"'):
            xml_content = ET.fromstring(b'<invalid>xml</invalid>')
            _ = get_profile(xml_content, 'NonExistent/Filter')

    def test_no_data(self):
        """ VOTable without TABLEDATA should raise FilterNetworkError. """
        xml_string = b'''<?xml version="1.0"?>
<VOTABLE version="1.1">
  <RESOURCE type="results">
    <TABLE>
      <FIELD name="Wavelength" datatype="double"/>
      <FIELD name="Transmission" datatype="double"/>
    </TABLE>
  </RESOURCE>
</VOTABLE>'''
        with pytest.raises(FilterNetworkError, match='No TABLEDATA'):
            xml_content = ET.fromstring(xml_string)
            _ = get_profile(xml_content, 'NonExistent/Filter')


class TestGetParams:
    """ Tests for get_info_from_xml() function. """

    def test_on_sample(self):
        """ Sample XML has DetectorType=1 (Photon counter). """
        xml_content = ET.fromstring(SAMPLE_XML_PATH.read_bytes())
        wavelength_unit = get_parameter(xml_content, 'WavelengthUnit', default='Angstrom')
        detector_type = get_parameter(xml_content, 'DetectorType', default='0')
        assert wavelength_unit == 'Angstrom', f'Expected "Angstrom", got "{wavelength_unit}"'
        assert detector_type == '1', f'Expected Photon counter ("1"), got "{detector_type}"'

    def test_default(self):
        """ If DetectorType is not present, default should be 0 (Energy counter) """
        xml_string = b'''<?xml version="1.0"?>
<VOTABLE version="1.1">
  <RESOURCE type="results">
    <TABLE>
      <PARAM name="filterID" value="Test/Test"/>
      <FIELD name="Wavelength" datatype="double"/>
      <FIELD name="Transmission" datatype="double"/>
      <DATA><TABLEDATA></TABLEDATA></DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>'''
        xml_content = ET.fromstring(xml_string)
        wavelength_unit = get_parameter(xml_content, 'WavelengthUnit', default='Angstrom')
        detector_type = get_parameter(xml_content, 'DetectorType', default='0')
        assert wavelength_unit == 'Angstrom', f'Expected default value "Angstrom", got "{wavelength_unit}"'
        assert detector_type == '0', f'Expected default value "0" (Energy counter), got "{detector_type}"'


class TestFetchFPS:
    """
    Tests for the fetch_from_fps_raw() functions. All use mocking to avoid real HTTP requests.

    Note: This class directly calls fetch_from_fps_raw(), which bypasses _cached_get entirely,
    so Config.allow_internet_access has no effect here — we must patch urlopen at the module level.
    """

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_raw_success(self, mock_urlopen: MagicMock):
        """ Successful raw fetch should return the XML bytes content """
        # Mock the response to simulate SVO FPS returning our sample XML
        mock_response = MagicMock()
        mock_response.read.return_value = SAMPLE_XML_PATH.read_bytes()  # pyright: ignore[reportAny]
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)  # pyright: ignore[reportAny]
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=None)  # pyright: ignore[reportAny]
        xml_content = fetch_from_fps_raw('Galileo/SSI.7270A')
        assert isinstance(xml_content, ET.Element), 'Should return an XML element.'
        # Verify it contains the expected data by parsing it
        nm, _sd = get_profile(xml_content, 'NonExistent/Filter')
        assert len(nm) == EXPECTED_N_POINTS
        np.testing.assert_equal(cast(float, nm[0]), 7200.0)

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_http_error(self, mock_urlopen: MagicMock):
        """ HTTP error should raise FilterNetworkError """
        from email.message import Message as EmailMessage
        http_err = urle.HTTPError(
            'https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php',
            404, 'Not Found', EmailMessage(), None
        )
        mock_urlopen.side_effect = http_err
        with pytest.raises(FilterNetworkError) as exc_info:
            _ = fetch_from_fps_raw('NonExistent/Filter')
        assert 'HTTP error' in str(exc_info.value)

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_url_error(self, mock_urlopen: MagicMock):
        """ URL error should raise FilterNetworkError """
        url_err = urle.URLError('Connection refused')
        mock_urlopen.side_effect = url_err
        with pytest.raises(FilterNetworkError) as exc_info:
            _ = fetch_from_fps_raw('NonExistent/Filter')

        assert 'Request failed' in str(exc_info.value)

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_bad_status(self, mock_urlopen: MagicMock):
        """ SVO FPS returning bad status should raise FilterNetworkError """
        xml_string = b'''<?xml version="1.0"?>
<VOTABLE version="1.1">
  <INFO name="QUERY_STATUS" value="ERROR"/>
</VOTABLE>'''
        mock_response = MagicMock()
        mock_response.read.return_value = xml_string  # pyright: ignore[reportAny]
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)  # pyright: ignore[reportAny]
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=None)  # pyright: ignore[reportAny]
        with pytest.raises(FilterNetworkError) as exc_info:
            _ = fetch_from_fps_raw('NonExistent/Filter')
        assert 'status' in str(exc_info.value).lower() or 'ERROR' in str(exc_info.value)


# === Config Filter Folder Tests ===

class TestCustomFiltersFolder:
    """ Tests for the get_custom_filters_folder() method. """

    def test_returns_none_when_unset(self):
        """ When no custom path is set, it should return None. """
        # Reset directly to bypass the setter's guard (which silently ignores non-Path values)
        Config._custom_filters_path = None  # pyright: ignore[reportPrivateUsage]
        assert Config.get_custom_filters_folder() is None

    def test_returns_path_from_path(self, tmp_filters_folder: Path):
        """ When a Path is set, it should return that path. """
        Config.set_custom_filters_folder(tmp_filters_folder)
        result = Config.get_custom_filters_folder()
        assert isinstance(result, Path)
        assert result == tmp_filters_folder

    def test_returns_path_from_str(self, tmp_filters_folder: Path):
        """ When a str is set, it should return the Path instance. """
        Config.set_custom_filters_folder(str(tmp_filters_folder))
        result = Config.get_custom_filters_folder()
        assert isinstance(result, Path)
        assert result == tmp_filters_folder

    def test_raises_for_nonexistent_directory(self):
        """ When a nonexistent directory path is set, it should raise FileNotFoundError. """
        with pytest.raises(FileNotFoundError, match='Custom filters folder does not exist'):
            Config.set_custom_filters_folder('/some/string/path')


class TestCachedFiltersFolder:
    """ Tests for the get_cached_filters_folder() method. """

    @patch.object(Config, '_default_cached_filters_folder', return_value=Path('/fake/cache/astrocolor/filters'))
    def test_returns_default_path(self, mock_default: MagicMock):  # pyright: ignore[reportUnusedParameter]
        """ Simulate an environment without creating real directories on disk. """
        Config._cached_filters_path = None  # pyright: ignore[reportPrivateUsage]
        result = Config.get_cached_filters_folder()
        assert isinstance(result, Path)
        expected_default = Path('/fake/cache/astrocolor/filters')
        assert str(result) == str(expected_default)

    def test_returns_custom_path_from_path(self, tmp_filters_folder: Path):
        """ When a custom Path is set, it should return that path. """
        # Reset directly to bypass the setter's guard and conftest fixture
        Config._cached_filters_path = None  # pyright: ignore[reportPrivateUsage]
        Config.set_cached_filters_folder(tmp_filters_folder)
        result = Config.get_cached_filters_folder()
        assert isinstance(result, Path)
        assert result == tmp_filters_folder

    def test_returns_path_from_str(self, tmp_filters_folder: Path):
        """ When a str is set, it should return the Path instance. """
        Config.set_cached_filters_folder(str(tmp_filters_folder))
        result = Config.get_cached_filters_folder()
        assert isinstance(result, Path)
        assert result == tmp_filters_folder

    @patch('astrocolor.config.platform.system', return_value='Linux')
    def test_restores_default_when_set_none(self, mock_system: MagicMock):  # pyright: ignore[reportUnusedParameter]
        """ Setting None should restore the platform-specific default path. """
        # Reset directly to bypass the setter's guard and conftest fixture
        Config._cached_filters_path = None  # pyright: ignore[reportPrivateUsage]
        Config.set_cached_filters_folder(None)
        result = Config.get_cached_filters_folder()
        assert isinstance(result, Path)
        xdg_cache = os.environ.get('XDG_CACHE_HOME') or (Path.home() / '.cache')
        expected_default = Path(xdg_cache) / 'astrocolor' / 'filters'
        assert str(result) == str(expected_default)

    def test_raises_for_nonexistent_directory(self):
        """ When a nonexistent directory path is set, it should raise FileNotFoundError. """
        with pytest.raises(FileNotFoundError, match='Cached filters folder does not exist'):
            Config.set_cached_filters_folder('/some/string/path')


class TestBundledFiltersFolder:
    """ Tests for the get_bundled_filters_folder() method. """

    def test_returns_bundled_filters_directory(self):
        """ Should always return a path inside the AstroColor package directory. """
        result = Config.get_bundled_filters_folder()
        assert isinstance(result, Path)
        # The bundled folder should be under library_folder (the astrocolor package dir).
        script_dir = Config.library_folder
        assert str(result).startswith(str(script_dir))
        assert 'filters' in result.parts


def _write_profile(folder: Path, group: str, name: str, start_nm: float) -> None:
    """ Write a small synthetic filter profile for lookup tests. """
    group_folder = folder / group
    group_folder.mkdir(parents=True, exist_ok=True)
    wavelengths = np.arange(start_nm, start_nm + 50, 10.)
    transmissions = np.array([0., 1., 1., 1., 0.])
    np.savetxt(group_folder / f'{name}.txt', np.column_stack([wavelengths, transmissions]))


class TestFilterCacheConfig:
    """ Tests that cached filters respect the current lookup configuration. """

    def test_cached_folder_change_invalidates_cache(self, tmp_path: Path):
        """ A profile reloaded after changing the cache folder must not come from the old cache. """
        folder_a = tmp_path / 'cached_a'
        folder_b = tmp_path / 'cached_b'
        _write_profile(folder_a, 'CacheTest', 'Switch', 400.)
        _write_profile(folder_b, 'CacheTest', 'Switch', 500.)
        original = Config.get_cached_filters_folder()
        try:
            Config.set_cached_filters_folder(folder_a)
            first = Filter.get('CacheTest/Switch')
            Config.set_cached_filters_folder(folder_b)
            second = Filter.get('CacheTest/Switch')
        finally:
            Config.set_cached_filters_folder(original)
        np.testing.assert_allclose(first.mean_nm(), 420.)
        np.testing.assert_allclose(second.mean_nm(), 520.)

    def test_custom_folder_change_invalidates_cache(self, tmp_path: Path):
        """ The custom filters folder is part of the same cache key. """
        folder_a = tmp_path / 'custom_a'
        folder_b = tmp_path / 'custom_b'
        _write_profile(folder_a, 'CacheTest', 'CustomSwitch', 600.)
        _write_profile(folder_b, 'CacheTest', 'CustomSwitch', 700.)
        original = Config.get_custom_filters_folder()
        try:
            Config.set_custom_filters_folder(folder_a)
            first = Filter.get('CacheTest/CustomSwitch')
            Config.set_custom_filters_folder(folder_b)
            second = Filter.get('CacheTest/CustomSwitch')
        finally:
            Config.set_custom_filters_folder(original)
        np.testing.assert_allclose(first.mean_nm(), 620.)
        np.testing.assert_allclose(second.mean_nm(), 720.)

    def test_internet_access_change_invalidates_cache(self, tmp_path: Path):
        """ Enabling network access must not reuse a profile cached while offline. """
        folder = tmp_path / 'cache'
        _write_profile(folder, 'CacheTest', 'Network', 800.)
        original_cached = Config.get_cached_filters_folder()
        original_internet = Config.allow_internet_access
        try:
            Config.set_cached_filters_folder(folder)
            Config.allow_internet_access = False
            first = Filter.get('CacheTest/Network')
            (folder / 'CacheTest' / 'Network.txt').unlink()
            Config.allow_internet_access = True
            xml_content = ET.fromstring(b'''<VOTABLE>
                <RESOURCE type="results"><TABLE>
                    <PARAM name="WavelengthUnit" value="Nanometer"/>
                    <PARAM name="DetectorType" value="0"/>
                </TABLE></RESOURCE>
            </VOTABLE>''')
            with (
                patch('astrocolor.filter_objects.fetch_from_fps_raw', return_value=xml_content),
                patch(
                    'astrocolor.filter_objects.get_profile',
                    return_value=(
                        np.arange(900., 950., 10.),
                        np.array([0., 1., 1., 1., 0.]),
                    ),
                ),
            ):
                second = Filter.get('CacheTest/Network')
        finally:
            Config.set_cached_filters_folder(original_cached)
            Config.allow_internet_access = original_internet
        np.testing.assert_allclose(first.mean_nm(), 820.)
        np.testing.assert_allclose(second.mean_nm(), 920.)


class TestFilterLookupPriority:
    """ Tests for the filter lookup precedence. """

    def test_custom_profile_overrides_cached_download(self, tmp_path: Path):
        """ A custom profile must win over an already cached SVO FPS download. """
        cached_folder = tmp_path / 'cached'
        custom_folder = tmp_path / 'custom'
        _write_profile(cached_folder, 'PriorityTest', 'Priority', 800.)
        _write_profile(custom_folder, 'PriorityTest', 'Priority', 900.)
        original_cached = Config.get_cached_filters_folder()
        original_custom = Config.get_custom_filters_folder()
        try:
            Config.set_cached_filters_folder(cached_folder)
            downloaded = Filter.get('PriorityTest/Priority')
            Config.set_custom_filters_folder(custom_folder)
            corrected = Filter.get('PriorityTest/Priority')
        finally:
            Config.set_cached_filters_folder(original_cached)
            Config.set_custom_filters_folder(original_custom)
        np.testing.assert_allclose(downloaded.mean_nm(), 820.)
        np.testing.assert_allclose(corrected.mean_nm(), 920.)

    def test_bundled_profile_overrides_custom(self, tmp_path: Path):
        """ Bundled profiles stay authoritative even when a custom folder is set. """
        custom_folder = tmp_path / 'custom'
        _write_profile(custom_folder, 'Generic', 'Bessell.V', 800.)
        bundled = Filter.get('Generic/Bessell.V')
        original_custom = Config.get_custom_filters_folder()
        try:
            Config.set_custom_filters_folder(custom_folder)
            selected = Filter.get('Generic/Bessell.V')
        finally:
            Config.set_custom_filters_folder(original_custom)
        np.testing.assert_array_equal(selected.wavelength_nm, bundled.wavelength_nm)
        np.testing.assert_array_equal(selected.spectral_dist, bundled.spectral_dist)
