import json
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

import astrocolor as ac
from astrocolor import filter_catalog
from astrocolor.config import Config
from astrocolor.filter_loader import parse_local_filter_name


@pytest.fixture
def catalog_env(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """ Isolate the catalog from the real cache and from the network. """
    cached = tmp_path / 'cached'
    custom = tmp_path / 'custom'
    _ = cached.mkdir()
    _ = custom.mkdir()
    with Config.override(
        cached_filters_folder=cached,
        custom_filters_folder=custom,
        allow_internet_access=False,
    ):
        filter_catalog._clear_catalog_cache()  # pyright: ignore[reportPrivateUsage]
        try:
            yield cached, custom
        finally:
            filter_catalog._clear_catalog_cache()  # pyright: ignore[reportPrivateUsage]


def _write_profile(
    folder: Path,
    facility: str,
    instrument: str,
    band: str,
    start_nm: float,
    extension: str = 'txt',
) -> None:
    """ Write a minimal SVO-named local profile. """
    facility_folder = folder / facility
    _ = facility_folder.mkdir(parents=True, exist_ok=True)
    _ = (facility_folder / f'{instrument}.{band}.{extension}').write_text(
        f'{start_nm} 0\n{start_nm + 10} 1\n',
        encoding='utf-8',
    )


def _cache_json(cached: Path) -> Path:
    return cached / 'filter_catalog.json'


def _set_checked_at(cached: Path, value: datetime) -> None:
    path = _cache_json(cached)
    data = cast(dict[str, object], json.loads(path.read_text(encoding='utf-8')))
    data['checked_at'] = value.timestamp()
    _ = path.write_text(json.dumps(data), encoding='utf-8')


@pytest.mark.parametrize(
    ('filename', 'expected'),
    [
        ('Bessell.V.datA', ('Bessell', 'V')),
        ('1931_2deg.x.txt', ('1931_2deg', 'x')),
        ('README.md', None),
        ('Bessell.V.fits', None),
        ('Bessell.V', None),
    ],
)
def test_parse_local_filter_name(filename: str, expected: tuple[str, str] | None) -> None:
    assert parse_local_filter_name(filename) == expected


def test_corrupt_catalog_cache_falls_back(catalog_env: tuple[Path, Path]) -> None:
    cached, _custom = catalog_env
    _ = _cache_json(cached).write_bytes(b'\xff\xfe{}')
    assert 'Generic' in ac.get_facilities()


def test_parse_votable_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    votable = ET.fromstring(b'''<?xml version="1.0"?>
<VOTABLE>
  <INFO name="QUERY_STATUS" value="OK"/>
  <RESOURCE type="results">
    <TABLE>
      <FIELD name="filterID" datatype="char" arraysize="*"/>
      <FIELD name="Instrument" datatype="char" arraysize="*"/>
      <DATA><TABLEDATA>
        <TR><TD>Fac/Cam.g</TD><TD></TD></TR>
        <TR><TD>Fac/Cam.r</TD><TD></TD></TR>
        <TR><TD>malformed-id</TD><TD></TD></TR>
      </TABLEDATA></DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>''')
    monkeypatch.setattr(filter_catalog, 'fetch_xml', MagicMock(return_value=votable))
    assert filter_catalog._fetch_server_facility_filters('Fac') == {'Fac/Cam.g', 'Fac/Cam.r'}  # pyright: ignore[reportPrivateUsage]


def test_parse_browse_html(monkeypatch: pytest.MonkeyPatch) -> None:
    html = '''
    <a href="index.php?mode=browse&gname=Generic&gname2=Bessell&asttype=">Bessell</a>
    <a href="index.php?mode=browse&gname=Generic&gname2=Cousins&asttype=">Cousins</a>
    <a href="index.php?id=Generic/Bessell.V&mode=browse">V</a>
    <a href="index.php?id=Generic/Bessell.B&mode=browse">B</a>
    <a href="index.php?id=Other/Cam.g&mode=browse">other</a>
    <a href="index.php?id=Other/Bessell.X&mode=browse">foreign Bessell</a>
    '''
    monkeypatch.setattr(filter_catalog, 'fetch_text', MagicMock(return_value=html))
    assert filter_catalog._fetch_server_facility_instruments_html('Generic') == ['Bessell', 'Cousins']  # pyright: ignore[reportPrivateUsage]
    assert filter_catalog._fetch_server_instrument_filters_html('Generic', 'Bessell') == {'B', 'V'}  # pyright: ignore[reportPrivateUsage]


def test_local_facilities_instruments_and_bands(catalog_env: tuple[Path, Path]) -> None:
    cached, custom = catalog_env
    _write_profile(cached, 'Shared', 'Cam', 'V', 400.)
    _write_profile(custom, 'Shared', 'Cam', 'B', 500.)
    _write_profile(custom, 'OnlyCustom', 'Spec', 'R', 600., extension='datA')

    facilities = ac.get_facilities()
    assert {'CIE', 'Generic', 'SLOAN', 'Shared', 'OnlyCustom'} <= set(facilities)

    assert ac.get_instruments('Shared') == ['Cam']
    assert ac.get_bands('Shared', 'Cam') == ['B', 'V']

    assert ac.get_instruments('OnlyCustom') == ['Spec']
    assert ac.get_bands('OnlyCustom', 'Spec') == ['R']

    assert ac.get_instruments('Generic') == ['Bessell']
    assert ac.get_bands('Generic', 'Bessell') == ['B', 'I', 'R', 'U', 'V']


def test_unknown_names_return_empty_lists(catalog_env: tuple[Path, Path]) -> None:
    _cached, _custom = catalog_env
    assert ac.get_instruments('NoSuchFacility') == []
    assert ac.get_bands('NoSuchFacility', 'NoSuchInstrument') == []
    assert ac.get_bands('Generic', 'NoSuchInstrument') == []
    assert ac.get_instruments('') == []
    assert ac.get_bands('', '') == []


def test_server_xml_catalog_is_fetched_once_and_cached(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached, _custom = catalog_env
    Config.allow_internet_access = True

    last_update = MagicMock(return_value='2026-10-19')
    browse = MagicMock(return_value={'Remote'})
    facility_filters = MagicMock(return_value=[
        'Remote/Cam.g',
        'Remote/Cam.r',
        'Remote/Other.x',
    ])
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', last_update)
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', browse)
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_filters', facility_filters)
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', MagicMock(return_value=None))

    facilities = ac.get_facilities()
    assert 'Remote' in facilities
    assert last_update.call_count == 1
    assert browse.call_count == 1

    assert ac.get_instruments('Remote') == ['Cam', 'Other']
    assert ac.get_bands('Remote', 'Cam') == ['g', 'r']
    assert ac.get_bands('Remote', 'Other') == ['x']
    assert facility_filters.call_count == 1

    # The same process must not repeat the update check or the facility fetch.
    _ = ac.get_facilities()
    _ = ac.get_instruments('Remote')
    _ = ac.get_bands('Remote', 'Cam')
    assert last_update.call_count == 1
    assert facility_filters.call_count == 1

    # A bundled-only facility must not trigger a server lookup after the
    # server facility list is known.
    assert ac.get_instruments('CIE') == ['1931_2deg']
    assert ac.get_bands('CIE', '1931_2deg') == ['x', 'y', 'z']
    assert facility_filters.call_count == 1
    assert _cache_json(cached).is_file()


def test_cached_server_catalog_works_offline(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached, _custom = catalog_env
    Config.allow_internet_access = True
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-10-19'))
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'Remote'}))
    monkeypatch.setattr(
        filter_catalog,
        '_fetch_server_facility_filters',
        MagicMock(return_value=['Remote/Cam.V', 'Remote/Cam.R']),
    )
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', MagicMock(return_value=None))

    assert ac.get_bands('Remote', 'Cam') == ['R', 'V']
    assert _cache_json(cached).is_file()

    # Reload the cache in a fresh in-memory catalog and disable the network.
    filter_catalog._clear_catalog_cache()  # pyright: ignore[reportPrivateUsage]
    Config.allow_internet_access = False
    network = MagicMock(side_effect=AssertionError('network must not be used'))
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', network)
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', network)
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_filters', network)
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', network)
    monkeypatch.setattr(filter_catalog, '_fetch_server_instrument_filters_html', network)

    assert 'Remote' in ac.get_facilities()
    assert ac.get_instruments('Remote') == ['Cam']
    assert ac.get_bands('Remote', 'Cam') == ['R', 'V']
    assert network.call_count == 0


def test_endpoint_change_does_not_reuse_server_cache(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-10-19'))
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'Remote'}))

    def facility_filters(_facility: str) -> list[str]:
        if Config.svo_fps_url.endswith('/one'):
            return ['Remote/Cam.V']
        return ['Remote/Cam.R']

    monkeypatch.setattr(
        filter_catalog,
        '_fetch_server_facility_filters',
        MagicMock(side_effect=facility_filters),
    )

    with Config.override(allow_internet_access=True):
        with Config.override(svo_fps_url='https://example.test/one'):
            assert ac.get_bands('Remote', 'Cam') == ['V']
        with Config.override(svo_fps_url='https://example.test/two'):
            assert ac.get_bands('Remote', 'Cam') == ['R']


def test_new_last_update_invalidates_server_catalog(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached, _custom = catalog_env
    Config.allow_internet_access = True
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-01-01'))
    browse = MagicMock(return_value={'OldRemote'})
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', browse)
    monkeypatch.setattr(
        filter_catalog,
        '_fetch_server_facility_filters',
        MagicMock(return_value=['OldRemote/Cam.V']),
    )
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', MagicMock(return_value=None))

    assert ac.get_bands('OldRemote', 'Cam') == ['V']

    # Pretend the daily check was done two days ago.
    _set_checked_at(cached, datetime.now(UTC) - timedelta(days=2))
    filter_catalog._clear_catalog_cache()  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-02-01'))
    browse.return_value = {'NewRemote'}
    monkeypatch.setattr(
        filter_catalog,
        '_fetch_server_facility_filters',
        MagicMock(return_value=['NewRemote/Cam.B']),
    )

    facilities = ac.get_facilities()
    assert 'NewRemote' in facilities
    assert 'OldRemote' not in facilities
    assert ac.get_bands('NewRemote', 'Cam') == ['B']


def test_server_facility_filters_are_parsed_without_instrument_column(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ SVO sometimes leaves the Instrument column empty; filter IDs still carry it. """
    _cached, _custom = catalog_env
    Config.allow_internet_access = True
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-10-19'))
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'NOAO'}))
    monkeypatch.setattr(
        filter_catalog,
        '_fetch_server_facility_filters',
        MagicMock(return_value=['NOAO/NEWFIRM.J1', 'NOAO/NEWFIRM.J2']),
    )
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', MagicMock(return_value=None))

    assert ac.get_instruments('NOAO') == ['NEWFIRM']
    assert ac.get_bands('NOAO', 'NEWFIRM') == ['J1', 'J2']


def test_foreign_facility_ids_are_ignored(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-10-19'))
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'Alias'}))
    facility_filters = MagicMock(return_value=['Canonical/Cam.V'])
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_filters', facility_filters)
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', MagicMock(return_value=['Cam']))
    instrument_filters = MagicMock(return_value=set())
    monkeypatch.setattr(filter_catalog, '_fetch_server_instrument_filters_html', instrument_filters)

    with Config.override(allow_internet_access=True):
        assert 'Alias' in ac.get_facilities()
        assert ac.get_instruments('Alias') == ['Cam']
        assert ac.get_bands('Alias', 'Cam') == []
    assert facility_filters.call_count == 1
    assert instrument_filters.call_count == 1


def test_server_html_fallback_for_facilities_without_votable_rows(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    Config.allow_internet_access = True
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-10-19'))
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'BrowseOnly'}))
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_filters', MagicMock(return_value=[]))
    monkeypatch.setattr(
        filter_catalog,
        '_fetch_server_facility_instruments_html',
        MagicMock(return_value=['Cam', 'Spec']),
    )
    instrument_filters = MagicMock(return_value={'a', 'b'})
    monkeypatch.setattr(filter_catalog, '_fetch_server_instrument_filters_html', instrument_filters)

    assert ac.get_instruments('BrowseOnly') == ['Cam', 'Spec']
    assert ac.get_bands('BrowseOnly', 'Cam') == ['a', 'b']
    assert instrument_filters.call_count == 1

    # An instrument absent from the facility page must not trigger a request.
    assert ac.get_bands('BrowseOnly', 'Missing') == []
    assert instrument_filters.call_count == 1


def test_browse_index_failure_returns_local_facilities(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    Config.allow_internet_access = True
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', MagicMock(return_value='2026-10-19'))
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value=None))
    facility_filters = MagicMock(side_effect=AssertionError('VOTable fallback must not be used'))
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_filters', facility_filters)

    facilities = ac.get_facilities()
    assert 'Generic' in facilities
    assert facility_filters.call_count == 0


def test_catalog_uses_configured_svo_urls(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    urls: list[str] = []

    def fake_text(url: str) -> str | None:
        urls.append(url)
        return None

    def fake_xml(url: str) -> ET.Element | None:
        urls.append(url)
        return None

    monkeypatch.setattr(filter_catalog, 'fetch_text', fake_text)
    monkeypatch.setattr(filter_catalog, 'fetch_xml', fake_xml)

    with Config.override(
        svo_fps_home_url='https://example.test/index.php',
        svo_fps_url='https://example.test/fps.php',
    ):
        _ = filter_catalog._fetch_last_update()  # pyright: ignore[reportPrivateUsage]
        _ = filter_catalog._fetch_browse_facilities()  # pyright: ignore[reportPrivateUsage]
        _ = filter_catalog._fetch_server_facility_filters('HST')  # pyright: ignore[reportPrivateUsage]

    assert urls[0].startswith('https://example.test/index.php?mode=voservice')
    assert any(url.startswith('https://example.test/index.php?mode=browse') for url in urls)
    assert any(url.startswith('https://example.test/fps.php?') for url in urls)


def test_catalog_ttl_is_configurable(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    Config.allow_internet_access = True
    last_update = MagicMock(return_value='2026-10-19')
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', last_update)
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'Remote'}))

    _ = ac.get_facilities()
    assert last_update.call_count == 1

    with Config.override(filter_catalog_ttl=0):
        _ = ac.get_facilities()
        _ = ac.get_facilities()
    assert last_update.call_count == 3


def test_update_date_is_checked_at_most_once_per_day(
    catalog_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cached, _custom = catalog_env
    Config.allow_internet_access = True
    last_update = MagicMock(return_value='2026-10-19')
    monkeypatch.setattr(filter_catalog, '_fetch_last_update', last_update)
    monkeypatch.setattr(filter_catalog, '_fetch_browse_facilities', MagicMock(return_value={'Remote'}))
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_filters', MagicMock(return_value=[]))
    monkeypatch.setattr(filter_catalog, '_fetch_server_facility_instruments_html', MagicMock(return_value=[]))

    _ = ac.get_facilities()
    _ = ac.get_facilities()
    assert last_update.call_count == 1
