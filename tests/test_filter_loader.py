import urllib.error as urle
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astrocolor.errors import FilterNetworkError
from astrocolor.filter_loader import (
    fetch_from_fps_raw,
    get_parameter,
    get_profile,
)

# === FilterObjects Statistics Tests ===

# Local file loading, SVO FPS fetching, caching.
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
        np.testing.assert_equal(nm[0], 7200.0, f'First wavelength: {nm[0]} != 7200.0')
        np.testing.assert_equal(sd[0], 0.002494, f'First transmission: {sd[0]} != 0.002494')

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
    """ Tests for the fetch_from_fps_raw() functions. All use mocking. """

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_raw_success(self, mock_urlopen):
        """ Successful raw fetch should return the XML bytes content """
        # Mock the response to simulate SVO FPS returning our sample XML
        mock_response = MagicMock()
        mock_response.read.return_value = SAMPLE_XML_PATH.read_bytes()
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=None)
        xml_content = fetch_from_fps_raw('Galileo/SSI.7270A')
        assert isinstance(xml_content, ET.Element), 'Should return an XML element.'
        # Verify it contains the expected data by parsing it
        nm, _sd = get_profile(xml_content, 'NonExistent/Filter')
        assert len(nm) == EXPECTED_N_POINTS
        np.testing.assert_equal(nm[0], 7200.0)

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_http_error(self, mock_urlopen):
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
    def test_fetch_url_error(self, mock_urlopen):
        """ URL error should raise FilterNetworkError """
        url_err = urle.URLError('Connection refused')
        mock_urlopen.side_effect = url_err
        with pytest.raises(FilterNetworkError) as exc_info:
            _ = fetch_from_fps_raw('NonExistent/Filter')

        assert 'Request failed' in str(exc_info.value)

    @patch('astrocolor.filter_loader.urlopen')
    def test_fetch_bad_status(self, mock_urlopen):
        """ SVO FPS returning bad status should raise FilterNetworkError """
        xml_string = b'''<?xml version="1.0"?>
<VOTABLE version="1.1">
  <INFO name="QUERY_STATUS" value="ERROR"/>
</VOTABLE>'''
        mock_response = MagicMock()
        mock_response.read.return_value = xml_string
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=None)
        with pytest.raises(FilterNetworkError) as exc_info:
            _ = fetch_from_fps_raw('NonExistent/Filter')
        assert 'status' in str(exc_info.value).lower() or 'ERROR' in str(exc_info.value)
