import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt

from .errors import FilterNetworkError

script_folder = Path(__file__).parent
filters_folder = script_folder/'filters'

SVO_FPS_URL: str = 'https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php'

allow_internet_access = True


def fetch_from_fps_raw(filter_id: str) -> ET.Element:
    """
    Fetch raw XML content from SVO FPS for a given filter ID.
    Checks QUERY_STATUS and raises FilterNetworkError on failure.
    """
    url = f'{SVO_FPS_URL}?ID={filter_id}'

    try:
        req = Request(url)
        with urlopen(req, timeout=30) as resp:
            xml_content = resp.read()
    except HTTPError as e:
        raise FilterNetworkError(filter_id, f'HTTP error: {e.code} {e.reason}')
    except URLError as e:
        raise FilterNetworkError(filter_id, f'Request failed: {e.reason}')

    # Check for query status OK in the response
    xml_str = xml_content.decode('utf-8') if isinstance(xml_content, bytes) else xml_content
    root = ET.fromstring(xml_str)
    for elem in root.iter():
        if elem.tag == 'INFO' and elem.get('name', '').lower() == 'query_status' and elem.get('value', '') != 'OK':
            raise FilterNetworkError(filter_id, f'SVO FPS returned status: {elem.get("value")}')

    return ET.fromstring(xml_content)

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
    """ Extract WavelengthUnit and DetectorType from SVO FPS VOTable PARAM elements """
    param = root.find(f".//PARAM[@name='{target}']")
    if param is not None:
        result = param.get('value')
        if result is not None:
            return result
    return default
