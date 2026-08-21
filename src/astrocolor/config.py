import os
import platform
from pathlib import Path


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

    # Fetch filters from Spanish Virtual Observatory Filter Profile Service (SVO FPS)
    # https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php
    allow_internet_access: bool = True

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
