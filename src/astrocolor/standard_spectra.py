"""
Bundled standard spectra of the Sun and Vega.

The spectra are exposed through `get_standard_spectrum()`, which returns a
fresh `Spectrum` on every call. The loaded templates are cached privately, so
a caller cannot corrupt the standards shared by photometry and color
calculations.
"""

# CALSPEC: composite ultraviolet and optical absolute calibrated reference spectra of the HST standards
# Main link: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
# Database references: Bohlin, Gordon, & Tremblay (2014), Bohlin, Hubeny, & Rauch (2020), Bohlin et al. (2022), Bohlin et al. (2025).

# TSIS-1 HSRS: TSIS-1 Hybrid Solar Reference Spectrum (HSRS) - Full Spectrum Extension
# Main link: https://lasp.colorado.edu/lisird/data/tsis1_hsrs_binned_fs
# Related research: "Version 2 of the TSIS-1 Hybrid Solar Reference Spectrum and Extension to the Full Spectrum."
# Coddington, O. M., Richard, E. C., Harber, D., Pilewskie, P., Woods, T. N., Snow, M., et al. (2023).
# Earth and Space Science, 10, e2022EA002637. https://doi.org/10.1029/2022EA002637

from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from .spectral_objects import Spectrum

# - Sun [CALSPEC]
# Binned version of sun_reference_stis_002.fits
# Ultraviolet to near-infrared absolute flux distribution of the Sun to 2.7 microns. It combines absolute
# flux measurements from space and the ground with a model spectrum for the near-infrared (Colina, Bohlin, & Castelli 1996).
# - Sun [TSIS-1 HSRS]
# Time range: Dec 01, 2019 to Dec 06, 2019
# Spectral range: 115 nm to 200000 nm
# License: Creative Commons - Attribution 4.0 International
# - Vega [CALSPEC]
# Binned version of alpha_lyr_stis_012.fits
# The composite flux standard consists of a special Kurucz 9550K model from 900-1152A (Kurucz 2003),
# IUE data from 1152-1675A, STIS CCD fluxes from 1675-10200A (Bohlin & Gilliland 2004a), and the 9550K model longward of 10200A
_FILES: dict[tuple[str, str], str] = {
    ('Sun', 'CALSPEC'): 'Sun_CALSPEC.npz',
    ('Sun', 'TSIS-1 HSRS'): 'Sun_TSIS-1_HSRS.npz',
    ('Vega', 'CALSPEC'): 'Vega_CALSPEC.npz',
}

_DEFAULT_SOURCE: dict[str, str] = {
    'Sun': 'TSIS-1 HSRS',
    'Vega': 'CALSPEC',
}

_DATA_FOLDER = Path(__file__).parent / 'data'


def _normalize(value: str) -> str:
    """ Normalizes a user-facing name or source for case- and space-insensitive matching. """
    return ' '.join(value.split()).casefold()


_LOOKUP: dict[tuple[str, str], tuple[str, str]] = {
    (_normalize(name), _normalize(source)): (name, source)
    for name, source in _FILES
}
_CANONICAL_NAMES: dict[str, str] = {
    _normalize(name): name for name, _ in _FILES
}


def _resolve_standard_spectrum(name: str, source: str | None) -> tuple[str, str]:
    """ Finds a registry key, reporting the available choices on failure. """
    name_key = _normalize(name)
    canonical_name = _CANONICAL_NAMES.get(name_key)
    if canonical_name is None:
        available = ', '.join(
            f'{entry_name} [{entry_source}]' for entry_name, entry_source in _FILES
        )
        raise ValueError(
            f'Unknown standard spectrum {name!r}. Available: {available}.'
        )
    if source is None:
        source = _DEFAULT_SOURCE[canonical_name]
    try:
        return _LOOKUP[(name_key, _normalize(source))]
    except KeyError:
        available_sources = ', '.join(
            entry_source for entry_name, entry_source in _FILES
            if entry_name == canonical_name
        )
        raise ValueError(
            f'Unknown source {source!r} for standard spectrum {canonical_name!r}. '
            + f'Available sources: {available_sources}.'
        ) from None


@cache
def _load_template(name: str, source: str) -> Spectrum:
    """ Loads and caches a template for a canonical registry key. """
    path = _DATA_FOLDER / _FILES[(name, source)]
    try:
        with cast(np.lib.npyio.NpzFile, np.load(path)) as data:
            wavelength_nm = cast(npt.NDArray[np.integer], data['wavelength_nm'])
            spectral_dist = cast(npt.NDArray[np.floating], data['spectral_dist'])
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'Missing data file for standard spectrum {name!r} from source {source!r}: {path}'
        ) from error
    return Spectrum(
        wavelength_nm,
        spectral_dist,
        name=f'{name} [{source}]',
    )


def get_standard_spectrum(name: str, source: str | None = None) -> Spectrum:
    """
    Returns a copy of a bundled standard spectrum.

    The template is loaded and cached lazily; every call returns a new
    `Spectrum`, so changing it cannot affect other users of the standard.

    Args:
    - `name` (str): standard name, `"Sun"` or `"Vega"` (case-insensitive)
    - `source` (str | None): data source, `"CALSPEC"` or `"TSIS-1 HSRS"`
      (case-insensitive). When omitted, the default source is used:
      `"TSIS-1 HSRS"` for the Sun and `"CALSPEC"` for Vega.

    Raises:
    - ValueError: if the name or source is unknown
    - FileNotFoundError: if the bundled data file is missing

    Example:
    ```
    >>> sun = get_standard_spectrum('Sun', 'CALSPEC')
    >>> vega = get_standard_spectrum('Vega')
    ```
    """
    canonical_name, canonical_source = _resolve_standard_spectrum(name, source)
    template = _load_template(canonical_name, canonical_source)
    return Spectrum(
        template.wavelength_nm,
        template.spectral_dist,
        name=template.name,
    )


def clear_cache() -> None:
    """ Clears the cached standard-spectrum templates. Intended for tests. """
    _load_template.cache_clear()
