import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.request import urlopen

import numpy as np

from astrocolor.errors import FilterNetworkError, FilterNotFoundError

# === FilterObjects Statistics Tests ===
# Local file loading, SVO FPS fetching, caching.
# All network-dependent tests use unittest.mock to avoid real HTTP requests.
