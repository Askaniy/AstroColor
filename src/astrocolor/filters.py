from copy import deepcopy
import numpy as np
import numpy.typing as npt
from typing import Any, Final
from functools import lru_cache

from .spectral_objects import Spectrum, SpectralSet
from .data_manager import script_folder
from .errors import FilterNotFoundError


@lru_cache(maxsize=128)
def _cached_get(filter_id: str) -> 'Filter':
    """
    Returns a cached filter object.
    This has been separated out to create copies of the results and avoid mutations.
    """
    if not isinstance(filter_id, str):
        raise ValueError('Spanish Virtual Observatory filter ID must be a string')
    try:
        file_path = next((script_folder / 'filters').glob(f'{filter_id}.*'))
        nm, sd = np.loadtxt(file_path).T[:2]
    except (StopIteration, FileNotFoundError):
        raise FilterNotFoundError(filter_id)
    # TODO: remove the block below after SVO FPS support is ready!
    if str(file_path)[-1] == 'A':
        # temporal workaround: convert angstrom to nm
        nm /= 10
    return Filter(nm, sd, name=filter_id)


class Filter(Spectrum):
    """
    Stores a filter profile as a normalized, edge-zeroed :class:`Spectrum`.

    A `Filter` is created through the classmethod `get()` which supports an optional cache.
    Direct construction via `__init__()` always creates a new object (no caching).

    Attributes:
        wavelength_nm     – spectral axis in nanometers
        spectral_dist     – normalized transmission profile
        covariance_matrix – always `None`
        name              – human-readable identifier
    """

    # Class-level sentinel
    covariance_matrix: Final[None] = None

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        name: Any = None,
    ) -> None:
        """
        Creates a :class:`Filter` from arrays of wavelength and transmission profile.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): normalized transmission profile
        - `name` (Any): human-readable identifier
        """
        spectrum = Spectrum(wavelength_nm, spectral_dist, name=name)
        self.from_spectrum(spectrum)

    def from_spectrum(self, spectrum: Spectrum) -> 'Filter':
        """ Create a :class:`Filter` from an arbitrary :class:`Spectrum`. """
        spectrum = spectrum.edges_zeroed().normalize()
        self.wavelength_nm = spectrum.wavelength_nm
        self.spectral_dist = spectrum.spectral_dist
        self.name = spectrum.name
        return self

    @staticmethod
    def get(filter_id: str) -> 'Filter':
        """ Use Spanish Virtual Observatory filter ID to create a :class:`Filter` object. """
        return deepcopy(_cached_get(filter_id))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class FilterSet(SpectralSet):
    """
    Class to work with a set of filter profiles.

    Attributes:
        wavelength_nm     – combined spectral axis covering all filter ranges
        spectral_dist     – packed transmissions [len(nm), n_filters]
        covariance_matrix – always `None` for FilterSet
        name              – tuple of human-readable filter identifiers

    Example:
        >>> bvr = FilterSet.get('Generic_Bessell.B', 'Generic_Bessell.V', 'Generic_Bessell.R')
    """

    # Class-level sentinel
    covariance_matrix: Final[None] = None

    def __init__(self, *filters: Filter) -> None:
        """
        Creates a :class:`FilterSet` from set of :class:`Filter` objects.
        Combines spectral axes into a single interval and combines transmission data into a single array.
        """
        # Getting the wavelength info and filter names
        names = []
        nm_min = np.inf
        nm_max = 0.
        for profile in filters:
            names.append(profile.name)
            nm_min = min(nm_min, profile.wavelength_nm[0])
            nm_max = max(nm_max, profile.wavelength_nm[-1])
        # Naming
        self.name: tuple[Any, ...] = tuple(names)
        # Matrix packing
        self.wavelength_nm = self._grid(nm_min, nm_max)
        self.spectral_dist = np.zeros((len(self.wavelength_nm), len(filters)), dtype=self._spectral_dist_dtype)
        for i, profile in enumerate(filters):
            mask = (self.wavelength_nm >= profile.wavelength_nm[0]) & (self.wavelength_nm <= profile.wavelength_nm[-1])
            self.spectral_dist[mask, i] = profile.spectral_dist

    @staticmethod
    def get(*filter_ids: str) -> 'FilterSet':
        """
        Use Spanish Virtual Observatory filter IDs to create a :class:`FilterSet` object.
        """
        filters = []
        for filter_id in filter_ids:
            filters.append(Filter.get(filter_id))
        return FilterSet(*filters)

    @property
    def matrix(self) -> npt.NDArray[np.floating]:
        """
        Transforms filter profiles' transmission spectral distribution
        into a matrix that simplifies the accounting of the grid step:
        A = T^T * Δλ  ->  y = A x, where x is a spectrum and y is a photospectrum
        """
        return self.spectral_dist.T * self.nm_step

    def __getitem__(self, index: int) -> Filter:
        """ Returns the filter profile with extra zeros trimmed off """
        # TODO: add support for a `slice` input and `FilterSet` output
        if not isinstance(index, int):
            raise TypeError(f'Index must be int, not {type(index).__name__}')
        # Trimming off the zeros and creating a new Filter object
        profile = self.spectral_dist[:,index]
        non_zero_indices = np.nonzero(profile)[0]
        start = non_zero_indices[0] - 1
        end = non_zero_indices[-1] + 2
        # Looking for the Filter name
        if isinstance(self.name, tuple) and len(self.name) == len(self):
            name = self.name[index]
        else:
            name = None
        return Filter(self.wavelength_nm[start:end], profile[start:end], name=name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
