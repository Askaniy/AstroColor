"""
Public API for photometric conversions, spectral reconstruction and color.

The package re-exports its core user-facing classes and functions here;
specialized APIs such as `transfer_functions` and `photometric_models` are
available as submodules.
"""

from . import photometric_models, transfer_functions
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
from .filter_objects import Filter, FilterSet
from .measurements import get_photometry, get_spectrometry, scale_to_match_value
from .photospectral_objects import PhotospectralCube, PhotospectralSet, Photospectrum
from .physical_models import BlackBodyModel
from .reconstructed_objects import (
    ReconstructedSpectralCube,
    ReconstructedSpectralSet,
    ReconstructedSpectrum,
    spectral_reconstruction,
)
from .reference_spectra import sun_calspec, sun_tsis1_hsrs, vega_calspec
from .spectral_objects import SpectralCube, SpectralSet, Spectrum
from .transfer_functions import TransferFunction
from .uncertainty import Uncertainty

# API namespace
__all__ = (
    'DEFAULT_SETTINGS',
    'LINEAR_SETTINGS',
    'BlackBodyModel',
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
    'ReconstructedSpectralCube',
    'ReconstructedSpectralSet',
    'ReconstructedSpectrum',
    'SpectralCube',
    'SpectralSet',
    'Spectrum',
    'TransferFunction',
    'Uncertainty',
    'WhitePoint',
    'default_color_system',
    'get_color',
    'get_photometry',
    'get_spectrometry',
    'photometric_models',
    'scale_to_match_value',
    'spectral_reconstruction',
    'sun_calspec',
    'sun_tsis1_hsrs',
    'transfer_functions',
    'vega_calspec',
    'visible_range',
    'xyz_cmf',
    'xyz_color_system',
)
