"""
Global settings for paths, network access, SVO endpoints and uncertainty handling.

`Config` controls filter lookup locations, SVO FPS downloads, catalog refresh
frequency, cube batching and whether correlations or cube uncertainties are
processed. Use `Config.override()` for temporary changes.
"""

import os
import platform
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast


class Config:
    """ Centralized global settings for the AstroColor library. """

    # Processing spectral cubes to generate images requires a significant amount of RAM.
    # If the pixel limit is exceeded, the image will be split into batches and then reassembled.
    # For example, 11K image processing and 1 megapixel chunk requires ~10 Gb of free RAM.
    # Value to optimize based on the computer specifications and Task Manager readings.
    pixel_upper_limit: int = 1_000_000 # 1 megapixel

    # Performance is prioritized for image processing through (photo)spectral cubes,
    # and uncertainty processing is disabled by default.
    ignore_uncertainty_for_cubes: bool = True

    # Ignore correlations between spectral points. Covariance matrices are
    # reduced to their diagonal, and operations on existing objects produce
    # compact results without modifying the stored representation.
    # Explicit covariance_matrix access still materializes the matrix.
    ignore_correlations: bool = False

    # Fetch filters from Spanish Virtual Observatory Filter Profile Service (SVO FPS)
    # https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php
    allow_internet_access: bool = True

    # SVO FPS endpoints. They are configurable for mirrors and tests.
    svo_fps_home_url: str = 'https://svo2.cab.inta-csic.es/svo/theory/fps3/index.php'
    svo_fps_url: str = 'https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php'

    # How often the SVO catalog checks the server's "Last update" date, in seconds.
    # The default is one day. Set it to 0 to check on every call.
    filter_catalog_ttl: float = 24 * 60 * 60

    # === Bundled filters ===

    library_folder: Path = Path(__file__).parent

    @classmethod
    def get_bundled_filters_folder(cls) -> Path:
        """
        Directory containing filter profiles shipped with the library.
        This is a read-only path.
        """
        return cls.library_folder / 'filters'

    # === Cached filters ===

    _cached_filters_path: Path | None = None

    @classmethod
    def _default_cached_filters_folder(cls) -> Path:
        """
        Compute the default cached filters folder for the current platform.

        Defaults depend on the operating system:

        - **Linux**: `$XDG_CACHE_HOME/astrocolor/filters` (`~/.cache/astrocolor/filters` if XDG_CACHE_HOME is unset)
        - **macOS**: `~/Library/Caches/AstroColor/filters`
        - **Windows**: `%LOCALAPPDATA%/AstroColor/filters` (i.e. `AppData\\Local\\AstroColor\\filters`).

        This method does not create the directory; use `get_cached_filters_folder` for that.
        """
        match platform.system():
            case 'Windows':
                localappdata = os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local'
                default = Path(localappdata) / 'AstroColor'
            case 'Darwin':  # macOS
                default = Path.home() / 'Library' / 'Caches' / 'AstroColor'
            case _:  # Linux (XDG)
                xdg_cache = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
                default = Path(xdg_cache) / 'astrocolor'
        return default / 'filters'

    @classmethod
    def get_cached_filters_folder(cls) -> Path:
        """
        Directory for filter profiles downloaded from SVO FPS and saved locally.

        Defaults depend on the operating system (see `_default_cached_filters_folder`).

        The folder is created lazily on first access if it does not exist yet.
        """
        if isinstance(cls._cached_filters_path, Path):
            return cls._cached_filters_path
        default = cls._default_cached_filters_folder()
        try:
            default.mkdir(parents=True, exist_ok=True)
        except OSError:  # e.g. read-only filesystem in CI
            pass
        cls._cached_filters_path = default
        return cls._cached_filters_path

    @classmethod
    def set_cached_filters_folder(cls, value: str | Path | None) -> None:
        """
        Set the directory for cached filter profiles downloaded from SVO FPS.

        Defaults depend on the operating system (see `_default_cached_filters_folder`).

        Pass `None` to restore the platform-specific default.

        Raises:
        - FileNotFoundError: If *value* does not point to an existing directory.
        """
        if isinstance(value, str):
            value = Path(value)
        if isinstance(value, Path) and not value.is_dir():
            raise FileNotFoundError(
                f'Cached filters folder does not exist: {value}'
            )
        cls._cached_filters_path = value

    # === Custom filters ===

    _custom_filters_path: Path | None = None

    @classmethod
    def get_custom_filters_folder(cls) -> Path | None:
        """
        Directory for user-provided filter files.
        Returns `None` if the user has not specified a path.
        """
        return cls._custom_filters_path

    @classmethod
    def set_custom_filters_folder(cls, value: str | Path | None) -> None:
        """
        Set the directory for user-provided filter files.

        Profiles in this folder take precedence over cached and freshly
        downloaded SVO FPS profiles, but not over bundled filters.

        Pass `None` to disable custom filters (the default).

        Raises:
        - FileNotFoundError: If *value* does not point to an existing directory.
        """
        if isinstance(value, str):
            value = Path(value)
        if isinstance(value, Path) and not value.is_dir():
            raise FileNotFoundError(
                f'Custom filters folder does not exist: {value}'
            )
        cls._custom_filters_path = value

    # === Temporary overrides ===

    @classmethod
    @contextmanager
    def override(cls, **values: object) -> Generator[None, None, None]:
        """
        Temporarily override `Config` settings and restore them afterwards.

        Both plain class attributes (for example `allow_internet_access`) and
        settings with getter/setter pairs (for example `cached_filters_folder`)
        are supported:

        ```
        with Config.override(allow_internet_access=False):
            ...
        ```

        Args:
        - `**values`: setting names and their temporary values.

        Raises:
        - AttributeError: If a setting with the given name does not exist.
        - FileNotFoundError: If a folder setter rejects the given path.
        """
        previous: list[tuple[str, object, Callable[[object], None] | None]] = []
        try:
            for name, value in values.items():
                setter = getattr(cls, f'set_{name}', None)
                getter = getattr(cls, f'get_{name}', None)
                if callable(setter) and callable(getter):
                    setter_fn = cast(Callable[[object], None], setter)
                    if name.endswith('_filters_folder'):
                        attribute = f'_{name.removesuffix("_filters_folder")}_filters_path'
                        old_value = cast(object, getattr(cls, attribute, None))
                    else:
                        getter_fn = cast(Callable[[], object], getter)
                        old_value = getter_fn()
                    previous.append((name, old_value, setter_fn))
                    setter_fn(value)
                elif hasattr(cls, name):
                    previous.append((name, cast(object, getattr(cls, name)), None))
                    setattr(cls, name, value)
                else:
                    raise AttributeError(f'Config has no setting named "{name}"')
            yield
        finally:
            for name, old_value, setter_fn in reversed(previous):
                if setter_fn is None:
                    setattr(cls, name, old_value)
                else:
                    setter_fn(old_value)
