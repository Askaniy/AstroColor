"""
SVO Filter Profile Service client and VOTable parser.

The module downloads filter profiles and extracts wavelengths, transmissions
and detector metadata; `Filter` uses it to build cached filter objects.
"""

import ssl
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
import numpy as np
import numpy.typing as npt

from .config import Config
from .errors import FilterNetworkError

_TIMEOUT_SECONDS = 30
ssl_ctx = ssl.create_default_context(cafile=certifi.where())


def _read_url(url: str) -> bytes:
    with urlopen(Request(url), timeout=_TIMEOUT_SECONDS, context=ssl_ctx) as response:  # pyright: ignore[reportAny]
        return response.read()  # pyright: ignore[reportAny]


def fetch_text(url: str) -> str | None:
    """ Fetch a text response, returning `None` on network or decoding errors. """
    try:
        payload = _read_url(url)
    except (OSError, ValueError):
        return None
    return payload.decode('utf-8', errors='replace')


def fetch_xml(url: str) -> ET.Element | None:
    """ Fetch and parse an XML response, returning `None` on any error. """
    text = fetch_text(url)
    if text is None:
        return None
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return None


def query_status_ok(root: ET.Element) -> bool:
    """ Check the VOTable `QUERY_STATUS`; a missing status is treated as success. """
    for element in root.iter('INFO'):
        if element.get('name', '').lower() == 'query_status':
            return element.get('value', '').upper() == 'OK'
    return True


def get_filter_folders() -> list[Path]:
    """ Folders that may contain filter profiles, in lookup priority order. """
    custom = Config.get_custom_filters_folder()
    return [
        Config.get_bundled_filters_folder(),
        *([custom] if custom is not None else []),
        Config.get_cached_filters_folder(),
    ]


def parse_local_filter_name(filename: str) -> tuple[str, str] | None:
    """
    Parse a local profile filename into ``(instrument, band)``.

    SVO-origin files are named ``Instrument.Band.ext``. Only text extensions
    understood by `Filter.get()` are accepted, so unrelated files in filter
    folders do not appear in the catalog.
    """
    stem, dot, extension = filename.rpartition('.')
    extension = extension.lower()
    if (
        not dot
        or not extension.startswith(('txt', 'dat', 'csv'))
        or any(character not in 'aupj' for character in extension[3:])
    ):
        return None
    instrument, dot, band = stem.partition('.')
    return (instrument, band) if dot and instrument and band else None


def fetch_from_fps_raw(filter_id: str) -> ET.Element:
    """
    Fetch raw XML content from SVO FPS for a given filter ID.
    Checks QUERY_STATUS and raises FilterNetworkError on failure.
    """
    url = f'{Config.svo_fps_url}?ID={filter_id}'

    try:
        xml_content = _read_url(url)
    except HTTPError as e:
        raise FilterNetworkError(filter_id, f'HTTP error: {e.code} {e.reason}')
    except URLError as e:
        raise FilterNetworkError(filter_id, f'Request failed: {e.reason}')

    # Check for query status OK in the response
    root = ET.fromstring(xml_content.decode('utf-8'))
    for elem in root.iter():
        if elem.tag == 'INFO' and elem.get('name', '').lower() == 'query_status' and elem.get('value', '') != 'OK':
            raise FilterNetworkError(filter_id, f'SVO FPS returned status: {elem.get("value")}')

    return root

def get_profile(xml_content: ET.Element, filter_id: str) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ Parse SVO FPS VOTable response and return (wavelength_angstrom, transmission) """

    # Find the first TABLE inside RESOURCE type="results"
    resource_elem = None
    for elem in xml_content:
        if elem.tag == 'RESOURCE' and elem.get('type') == 'results':
            resource_elem = elem
            break
    if resource_elem is None:
        raise FilterNetworkError(filter_id, 'No RESOURCE type="results" found in VOTable response')

    table_elem = None
    for elem in resource_elem:
        if elem.tag == 'TABLE':
            table_elem = elem
            break
    if table_elem is None:
        raise FilterNetworkError(filter_id, 'No TABLE element found inside RESOURCE type="results"')

    # Find DATA -> TABLEDATA section
    data_elem = None
    for elem in table_elem:
        if elem.tag == 'DATA':
            data_elem = elem
            break
    if data_elem is None or not any(c.tag == 'TABLEDATA' for c in data_elem):
        raise FilterNetworkError(filter_id, 'No TABLEDATA element found inside DATA section')

    tabledata_elem = next(c for c in data_elem if c.tag == 'TABLEDATA')

    wavelengths: list[float] = []
    transmissions: list[float] = []
    for tr in tabledata_elem:
        if tr.tag != 'TR':
            continue
        wl_text = tr[0].text
        tr_text = tr[1].text
        if wl_text is None or tr_text is None:
            continue
        try:
            wavelengths.append(float(wl_text))
            transmissions.append(float(tr_text))
        except ValueError:
            # Skip rows with non-numeric data
            pass

    if not wavelengths or not transmissions:
        raise FilterNetworkError(filter_id, 'No wavelength/transmission data found in VOTable response')

    return np.array(wavelengths, dtype=np.float64), np.array(transmissions, dtype=np.float64)

def get_parameter(root: ET.Element, target: str, default: str = '') -> str:
    """ Extract the value of a named PARAM element from an SVO FPS VOTable response """
    param = root.find(f".//PARAM[@name='{target}']")
    if param is not None:
        result = param.get('value')
        if result is not None:
            return result
    return default
