"""
Shared fixtures and isolation for the benchmark suite.

Benchmarks are not collected by the normal test run (see ``testpaths`` in
``pyproject.toml``); run them explicitly with
``pytest benchmarks --benchmark-only``.
"""

from collections.abc import Generator

import pytest

from astrocolor import Filter, FilterSet
from astrocolor.color_standards import ColorSystem, default_color_system
from astrocolor.config import Config


@pytest.fixture(scope='session', autouse=True)
def _isolate_filters(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None]:
    """Keep benchmarks offline and away from the user's filter cache."""
    folder = tmp_path_factory.mktemp('astrocolor_benchmarks')
    with Config.override(
        allow_internet_access=False,
        cached_filters_folder=folder,
        custom_filters_folder=None,
        ignore_uncertainty_for_cubes=True,
    ):
        yield


@pytest.fixture(scope='session')
def v_filter() -> Filter:
    """Bessell V, a representative single filter profile."""
    return Filter.get('Generic/Bessell.V')


@pytest.fixture(scope='session')
def ubv_filter_set() -> FilterSet:
    """Bessell UBV, a representative three-filter set."""
    return FilterSet.get(
        'Generic/Bessell.U',
        'Generic/Bessell.B',
        'Generic/Bessell.V',
    )


@pytest.fixture(scope='session')
def srgb() -> ColorSystem:
    """The default sRGB color system."""
    return default_color_system
