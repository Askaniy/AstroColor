from pathlib import Path

import pytest

from astrocolor import Filter, FilterSet
from astrocolor.config import Config


@pytest.fixture(scope='session')
def v_filter():
    return Filter.get('Generic/Bessell.V')

@pytest.fixture(scope='session')
def ubv_filterset():
    return FilterSet.get(
        'Generic/Bessell.U',
        'Generic/Bessell.B',
        'Generic/Bessell.V'
    )

# Session-scoped temp folder for filter cache / custom paths.
# The directory is cleaned before each test run to avoid stale state from previous runs.
@pytest.fixture(scope='session')
def tmp_filters_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('astrocolor_tmp', numbered=False)


@pytest.fixture(autouse=True, scope='session')
def _clean_tmp_dir(tmp_filters_folder: Path):  # pyright: ignore[reportUnusedFunction]
    """
    Remove any leftover files from previous test runs so tests always start with a clean slate.

    This is important because session-scoped fixtures share the same directory across
    all tests — without cleanup, stale cache or custom filter files could cause flaky failures.
    """
    # Clean up before use (in case of interrupted previous run)
    if tmp_filters_folder.exists():
        for item in tmp_filters_folder.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()
    yield

@pytest.fixture(scope='session', autouse=True)
def _isolate_filters(tmp_filters_folder: Path):  # pyright: ignore[reportUnusedFunction]
    """
    Isolate tests from the network and real XDG cache directory.

    - Disables internet access so Filter.get() never tries to fetch from SVO FPS.
      Filters needed for the tests are bundled and loaded directly from disk
    - Redirects the cached filter path to a local tmp/ directory (custom filters
      remain None so config-level unit tests can verify getter/setter logic).
    - Restores original Config state after all tests complete.

    This fixture is session-scoped (runs once per test run) because:
    1. Creating temp folders repeatedly is unnecessary overhead.
    2. Tests that modify Config._cached_filters_path / _custom_filters_path expect the
       same folder to exist across multiple test classes in a single pytest invocation.
    """
    # Save original state
    orig_internet = Config.allow_internet_access
    orig_custom = Config._custom_filters_path  # pyright: ignore[reportPrivateUsage]
    orig_cached = Config._cached_filters_path  # pyright: ignore[reportPrivateUsage]
    # Apply isolation: cached path is set permanently to avoid creating real dirs on disk
    Config.allow_internet_access = False
    tmp_filters_folder.mkdir(exist_ok=True)
    Config.set_cached_filters_folder(tmp_filters_folder)
    try:
        yield
    finally:
        # --- Restore original state ----------------------------------------------
        Config.allow_internet_access = orig_internet
        Config._custom_filters_path = orig_custom  # pyright: ignore[reportPrivateUsage]
        Config._cached_filters_path = orig_cached  # pyright: ignore[reportPrivateUsage]
