from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt

from .errors import FilterNetworkError, FilterNotFoundError

script_folder = Path(__file__).parent
filters_folder = script_folder / 'filters'

SVO_FPS_URL: str = 'https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php'

allow_internet_access = True
