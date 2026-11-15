"""
Discovery of available filter profiles.

The catalog merges three local sources (bundled, custom and cached profiles)
with the Spanish Virtual Observatory Filter Profile Service (SVO FPS).  The
SVO part is cached on disk and refreshed only when the service reports a new
``Last update`` date; that date is checked at most once per day.

Names follow the SVO FPS convention: a filter ID has the form
``Facility/Instrument.Band``, so ``Generic/Bessell.V`` belongs to the
``Generic`` facility, the ``Bessell`` instrument and the ``V`` band.
"""

import json
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from html import unescape
from pathlib import Path
from threading import RLock
from typing import cast
from urllib.parse import quote, unquote, urlencode

from .config import Config
from .filter_loader import (
    fetch_text,
    fetch_xml,
    get_filter_folders,
    parse_local_filter_name,
    query_status_ok,
)

__all__ = ('get_bands', 'get_facilities', 'get_instruments')

_CATALOG_FILENAME = 'filter_catalog.json'
_CATALOG_SCHEMA_VERSION = 3

_LAST_UPDATE_PATTERN = re.compile(r'Last update:\s*(\d{4}-\d{2}-\d{2})')
_BROWSE_FACILITY_PATTERN = re.compile(r'[?&]gname=([^&#"]*)')
_BROWSE_INSTRUMENT_PATTERN = re.compile(r'[?&]gname2=([^&#"]*)')
_FILTER_ID_PATTERN = re.compile(r'[?&]id=([^&#"]*)')

type _LocalCatalog = dict[str, dict[str, set[str]]]
type _BandCatalog = dict[str, dict[str, set[str] | None]]


@dataclass
class _CatalogData:
    """ Persistent part of the SVO catalog cache. """

    last_update: str | None = None
    checked_at: float | None = None
    home_url: str | None = None
    fps_url: str | None = None
    facilities: list[str] = field(default_factory=list)
    facilities_fetched: bool = False
    catalog: _BandCatalog = field(default_factory=dict)


# ------------ SVO response parsing ------------

def _findall(pattern: re.Pattern[str], text: str) -> list[str]:
    """ Typed wrapper around `re.Pattern.findall` for string patterns. """
    return cast(list[str], pattern.findall(text))


def _decode_names(pattern: re.Pattern[str], text: str) -> set[str]:
    """ Extract and URL/HTML-decode the values captured by a pattern. """
    return {unescape(unquote(encoded)).strip() for encoded in _findall(pattern, text)}


def _parse_filter_ids(text: str) -> list[str]:
    """ Extract valid filter IDs from an SVO browse page. """
    found: set[str] = set()
    for filter_id in _decode_names(_FILTER_ID_PATTERN, text):
        if _split_filter_id(filter_id) is not None:
            found.add(filter_id)
    return sorted(found)


def _parse_browse_instruments(text: str) -> list[str]:
    """ Extract instrument names from a facility browse page. """
    return sorted(name for name in _decode_names(_BROWSE_INSTRUMENT_PATTERN, text) if name)


def _browse_url(facility: str, instrument: str | None = None) -> str:
    """ Build a human-readable SVO browse URL for a facility or instrument. """
    parameters: dict[str, str] = {
        'mode': 'browse',
        'gname': facility,
        'asttype': '',
    }
    if instrument is not None:
        parameters['gname2'] = instrument
    return f'{Config.svo_fps_home_url}?{urlencode(parameters, quote_via=quote)}'


# ------------ Network fetchers ------------

def _fetch_last_update() -> str | None:
    """ Read the `Last update: YYYY-MM-DD` footer of the SVO FPS web service. """
    text = fetch_text(f'{Config.svo_fps_home_url}?mode=voservice')
    if text is None:
        return None
    match = _LAST_UPDATE_PATTERN.search(text)
    return match.group(1) if match is not None else None


def _fetch_browse_facilities() -> set[str] | None:
    """ Fetch the facility list from the SVO browse index. """
    text = fetch_text(f'{Config.svo_fps_home_url}?mode=browse&asttype=')
    if text is None:
        return None
    return {facility for facility in _decode_names(_BROWSE_FACILITY_PATTERN, text) if facility}


def _fetch_server_facility_filters(facility: str) -> set[str] | None:
    """
    Fetch all filter IDs of a facility through the SVO FPS VOTable service.

    Returns `None` on a network/parsing failure and an empty set when the
    service answers successfully but has no rows (some facilities, notably
    ``Generic``, are only available through the browse pages).
    """
    query = urlencode({'Facility': facility, 'VERB': '0'}, quote_via=quote)
    root = fetch_xml(f'{Config.svo_fps_url}?{query}')
    if root is None:
        return None
    if not query_status_ok(root):
        return set()

    filter_ids: set[str] = set()
    for table in root.iter('TABLE'):
        field_names = [field.get('name') for field in table.findall('FIELD')]
        if 'filterID' not in field_names:
            continue
        index = field_names.index('filterID')
        for row in table.findall('DATA/TABLEDATA/TR'):
            cells = list(row)
            if index >= len(cells):
                continue
            text = cells[index].text
            if text is None:
                continue
            filter_id = text.strip()
            if _split_filter_id(filter_id) is not None:
                filter_ids.add(filter_id)
    return filter_ids


def _fetch_server_facility_instruments_html(facility: str) -> list[str] | None:
    """ Fallback instrument discovery through the SVO browse pages. """
    text = fetch_text(_browse_url(facility))
    if text is None:
        return None
    return _parse_browse_instruments(text)


def _fetch_server_instrument_filters_html(facility: str, instrument: str) -> set[str] | None:
    """ Fallback band discovery through the SVO browse pages. """
    text = fetch_text(_browse_url(facility, instrument))
    if text is None:
        return None
    return _ids_to_bands(_parse_filter_ids(text), facility).get(instrument, set())


# ------------ Filter ID helpers ------------

def _split_filter_id(filter_id: str) -> tuple[str, str, str] | None:
    """ Split ``Facility/Instrument.Band`` into its three components. """
    facility, slash, remainder = filter_id.partition('/')
    instrument, dot, band = remainder.partition('.')
    if not (facility and slash and instrument and dot and band):
        return None
    return facility, instrument, band


def _ids_to_bands(filter_ids: Iterable[str], facility: str) -> dict[str, set[str]]:
    """
    Group filter IDs of one facility by instrument.

    IDs whose facility prefix differs from the requested one are ignored.
    SVO aliases can return canonical IDs for an alias query, but those IDs
    are not valid for `Filter.get()`, so they are not exposed.
    """
    bands: dict[str, set[str]] = {}
    for filter_id in filter_ids:
        parts = _split_filter_id(filter_id)
        if parts is not None:
            filter_facility, instrument, band = parts
            if filter_facility == facility:
                bands.setdefault(instrument, set()).add(band)
    return bands


# ------------ Local catalog ------------

def _local_catalog() -> _LocalCatalog:
    """
    Scan bundled, custom and cached filter folders.

    The result maps facility -> instrument -> bands. Duplicate profiles in
    different roots are merged, because the catalog describes availability
    rather than the lookup precedence used by `Filter.get()`.
    """
    catalog: _LocalCatalog = {}
    for root in get_filter_folders():
        if not root.is_dir():
            continue
        try:
            facility_directories = sorted(root.iterdir())
        except OSError:
            continue
        for facility_directory in facility_directories:
            if facility_directory.name.startswith('.') or not facility_directory.is_dir():
                continue
            try:
                files = sorted(facility_directory.iterdir())
            except OSError:
                continue
            for path in files:
                if path.name.startswith('.') or not path.is_file():
                    continue
                parsed = parse_local_filter_name(path.name)
                if parsed is None:
                    continue
                instrument, band = parsed
                catalog.setdefault(facility_directory.name, {}).setdefault(instrument, set()).add(band)
    return catalog


# ------------ Persistent server catalog ------------

def _as_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [item for item in items if isinstance(item, str)]


def _as_bands(value: object) -> set[str] | None:
    if value is None:
        return None
    return set(_as_string_list(value))


def _as_catalog(value: object) -> _BandCatalog:
    if not isinstance(value, dict):
        return {}
    items = cast(dict[object, object], value)
    catalog: _BandCatalog = {}
    for facility, instruments in items.items():
        if not isinstance(facility, str) or not isinstance(instruments, dict):
            continue
        instrument_items = cast(dict[object, object], instruments)
        catalog[facility] = {
            instrument: _as_bands(bands)
            for instrument, bands in instrument_items.items()
            if isinstance(instrument, str)
        }
    return catalog


def _catalog_from_json(raw: object) -> _CatalogData:
    """ Validate a JSON object and convert it to `_CatalogData`. """
    if not isinstance(raw, dict):
        return _CatalogData()
    data = cast(dict[str, object], raw)
    if data.get('schema_version') != _CATALOG_SCHEMA_VERSION:
        return _CatalogData()
    checked_at = data.get('checked_at')
    return _CatalogData(
        last_update=_as_string(data.get('last_update')),
        checked_at=checked_at if isinstance(checked_at, (int, float)) else None,
        home_url=_as_string(data.get('home_url')),
        fps_url=_as_string(data.get('fps_url')),
        facilities=_as_string_list(data.get('facilities')),
        facilities_fetched=data.get('facilities_fetched') is True,
        catalog=_as_catalog(data.get('catalog')),
    )


def _catalog_to_json(data: _CatalogData) -> dict[str, object]:
    """ Convert the catalog cache to a JSON-serializable object. """
    payload: dict[str, object] = asdict(data)
    payload['schema_version'] = _CATALOG_SCHEMA_VERSION
    payload['catalog'] = {
        facility: {
            instrument: None if bands is None else sorted(bands)
            for instrument, bands in instruments.items()
        }
        for facility, instruments in data.catalog.items()
    }
    return payload


def _load_catalog(path: Path) -> _CatalogData:
    """ Read the persistent catalog cache; a missing or broken file is empty. """
    try:
        raw = cast(object, json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return _CatalogData()
    return _catalog_from_json(raw)


class _ServerCatalog:
    """ Cached view of the SVO FPS facility/instrument/band hierarchy. """

    def __init__(self, path: Path, home_url: str, fps_url: str) -> None:
        self.path: Path = path
        self.data: _CatalogData = _load_catalog(path)
        if (
            self.data.home_url not in (None, home_url)
            or self.data.fps_url not in (None, fps_url)
        ):
            self.data = _CatalogData()
        self.data.home_url = home_url
        self.data.fps_url = fps_url
        self._lock: RLock = RLock()
        self._attempts: dict[str, float] = {}

    # --- persistence ---

    def _save_locked(self) -> None:
        """ Write the cache atomically, ignoring read-only cache locations. """
        try:
            _ = self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f'{self.path.name}.tmp')
            _ = temporary.write_text(
                json.dumps(_catalog_to_json(self.data), ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
            _ = temporary.replace(self.path)
        except OSError:
            pass

    def _clear_server_data_locked(self) -> None:
        """ Drop everything that depends on the current SVO database version. """
        self.data.facilities = []
        self.data.facilities_fetched = False
        self.data.catalog = {}
        self._attempts.clear()

    # --- freshness ---

    def _attempt_allowed(self, key: str) -> bool:
        """ Rate-limit one lazy fetch; successful data is checked separately. """
        now = time.monotonic()
        attempted = self._attempts.get(key)
        if attempted is not None and now - attempted < Config.filter_catalog_ttl:
            return False
        self._attempts[key] = now
        return True

    def _refresh_if_needed_locked(self) -> None:
        """ Check the SVO update date at most once per day. """
        if not Config.allow_internet_access:
            return
        checked_at = self.data.checked_at
        if checked_at is not None and 0 <= time.time() - checked_at < Config.filter_catalog_ttl:
            return
        if not self._attempt_allowed('last_update'):
            return

        last_update = _fetch_last_update()
        if last_update is None:
            return
        self.data.checked_at = time.time()
        if last_update != self.data.last_update:
            self.data.last_update = last_update
            self._clear_server_data_locked()
        self._save_locked()

    # --- facilities ---

    def _fetch_facilities_locked(self) -> None:
        """ Fill the server facility list once per catalog version. """
        if self.data.facilities_fetched or not Config.allow_internet_access:
            return
        if not self._attempt_allowed('facilities'):
            return
        browse = _fetch_browse_facilities()
        if not browse:
            return
        self.data.facilities = sorted(browse)
        self.data.facilities_fetched = True
        self._save_locked()

    def facilities(self) -> set[str]:
        """ Return the cached server facility names. """
        with self._lock:
            self._refresh_if_needed_locked()
            self._fetch_facilities_locked()
            return set(self.data.facilities) | set(self.data.catalog)

    # --- facility details ---

    def _load_facility_locked(self, facility: str) -> None:
        """ Fetch the band or instrument list of a facility once. """
        if facility in self.data.catalog:
            return
        if not Config.allow_internet_access:
            return
        if self.data.facilities_fetched and facility not in self.data.facilities:
            return
        if not self._attempt_allowed(f'facility:{facility}'):
            return

        filter_ids = _fetch_server_facility_filters(facility)
        bands = _ids_to_bands(filter_ids or (), facility)
        if bands:
            facility_catalog: dict[str, set[str] | None] = {}
            facility_catalog.update(bands)
            self.data.catalog[facility] = facility_catalog
            self._save_locked()
            return

        # The VOTable search is empty for a few facilities (for example
        # "Generic"), so fall back to the browse pages for instruments.
        instruments = _fetch_server_facility_instruments_html(facility)
        if instruments is not None:
            self.data.catalog[facility] = {instrument: None for instrument in instruments}
            self._save_locked()

    def instruments(self, facility: str) -> set[str]:
        """ Return the cached server instruments of a facility. """
        with self._lock:
            self._refresh_if_needed_locked()
            self._load_facility_locked(facility)
            return set(self.data.catalog.get(facility, {}))

    def filters(self, facility: str, instrument: str) -> set[str]:
        """ Return the cached server bands of a facility/instrument pair. """
        with self._lock:
            self._refresh_if_needed_locked()
            self._load_facility_locked(facility)

            facility_catalog = self.data.catalog.get(facility)
            if facility_catalog is None:
                return set()
            bands = facility_catalog.get(instrument)
            if bands is not None:
                return set(bands)
            if instrument not in facility_catalog:
                return set()
            if not Config.allow_internet_access:
                return set()
            if not self._attempt_allowed(f'instrument:{facility}/{instrument}'):
                return set()

            fetched = _fetch_server_instrument_filters_html(facility, instrument)
            if fetched is None:
                return set()
            facility_catalog[instrument] = fetched
            self._save_locked()
            return set(fetched)


@lru_cache(maxsize=4)
def _server_catalog_for(path: Path, home_url: str, fps_url: str) -> _ServerCatalog:
    """ Return the persistent catalog for one cache path and SVO endpoint. """
    return _ServerCatalog(path, home_url, fps_url)


def _get_server_catalog() -> _ServerCatalog:
    """ Return the persistent catalog associated with the current cache folder. """
    return _server_catalog_for(
        Config.get_cached_filters_folder() / _CATALOG_FILENAME,
        Config.svo_fps_home_url,
        Config.svo_fps_url,
    )


def _clear_catalog_cache() -> None:  # pyright: ignore[reportUnusedFunction]
    """ Drop in-memory catalog objects; the next call reloads them from disk. """
    _server_catalog_for.cache_clear()


# ------------ Public API ------------

def get_facilities() -> list[str]:
    """
    Return all available filter facilities.

    The result combines bundled, custom and cached local profiles with the
    facilities advertised by the SVO browse index. The SVO list is cached on
    disk and its ``Last update`` date is checked at most once per day. Network
    failures degrade gracefully to the locally available facilities.

    The browse index uses canonical facility names, so a returned name can be
    used directly in ``Facility/Instrument.Band``.

    Returns:
    - Sorted list of facility names, for example ``['Generic', 'HST', 'SLOAN']``.
    """
    local = _local_catalog()
    server = _get_server_catalog().facilities()
    return sorted(set(local) | server)


def get_instruments(facility: str) -> list[str]:
    """
    Return all instruments available for a filter facility.

    Args:
    - `facility` (str): facility name, for example ``'HST'``.

    Returns:
    - Sorted list of instrument names; an empty list for an unknown facility.
    """
    if not facility:
        return []
    local = _local_catalog().get(facility, {})
    server = _get_server_catalog().instruments(facility)
    return sorted(set(local) | server)


def get_bands(facility: str, instrument: str) -> list[str]:
    """
    Return all bands available for a facility/instrument pair.

    Args:
    - `facility` (str): facility name, for example ``'Generic'``.
    - `instrument` (str): instrument name, for example ``'Bessell'``.

    Returns:
    - Sorted list of band names, for example ``['B', 'I', 'R', 'U', 'V']``.
      Combine a band with the facility and instrument to obtain the filter ID
      used by `Filter.get()`: ``f'{facility}/{instrument}.{band}'``.
      An empty list is returned for an unknown pair.
    """
    if not facility or not instrument:
        return []
    local = _local_catalog().get(facility, {}).get(instrument, set())
    server = _get_server_catalog().filters(facility, instrument)
    return sorted(local | server)
