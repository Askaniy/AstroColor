"""
Public API for photometric conversions, spectral reconstruction and color.

The package re-exports its core user-facing data types and functions here;
specialized model families (`physical_models`, `photometric_models`) and
`transfer_functions` are available as submodules.
"""

from . import photometric_models, physical_models, transfer_functions
from .celestial_objects import CelestialObject
from .color_objects import (
    DEFAULT_SETTINGS,
    LINEAR_SETTINGS,
    ColorImage,
    ColorLine,
    ColorPoint,
    DisplaySettings,
    get_color,
)
from .color_standards import (
    Chromaticity,
    ColorSpace,
    ColorSystem,
    WhitePoint,
    default_color_system,
    visible_range,
    xyz_cmf,
    xyz_color_system,
)
from .config import Config
from .filter_catalog import get_bands, get_facilities, get_instruments
from .filter_objects import Filter, FilterSet
from .measurements import get_photometry, get_spectrometry, scale_to_match_value
from .photospectral_objects import PhotospectralCube, PhotospectralSet, Photospectrum
from .spectral_objects import SpectralCube, SpectralSet, Spectrum
from .spectral_reconstruction import reconstruct_from_photometry
from .standard_spectra import get_standard_spectrum
from .uncertainty import Uncertainty

# API namespace
__all__ = (
    'DEFAULT_SETTINGS',
    'LINEAR_SETTINGS',
    'CelestialObject',
    'Chromaticity',
    'ColorImage',
    'ColorLine',
    'ColorPoint',
    'ColorSpace',
    'ColorSystem',
    'Config',
    'DisplaySettings',
    'Filter',
    'FilterSet',
    'PhotospectralCube',
    'PhotospectralSet',
    'Photospectrum',
    'SpectralCube',
    'SpectralSet',
    'Spectrum',
    'Uncertainty',
    'WhitePoint',
    'default_color_system',
    'get_bands',
    'get_color',
    'get_facilities',
    'get_instruments',
    'get_photometry',
    'get_spectrometry',
    'get_standard_spectrum',
    'photometric_models',
    'physical_models',
    'reconstruct_from_photometry',
    'scale_to_match_value',
    'transfer_functions',
    'visible_range',
    'xyz_cmf',
    'xyz_color_system',
)
