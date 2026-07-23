from copy import deepcopy
import numpy as np
import numpy.typing as npt
from typing import Any, Sequence, Self
from functools import lru_cache

from .auxiliary import uniform_grid
from .core import nm_step, wavelength_nm_dtype
from .spectral_objects import SpectralObject, Spectrum, SpectralSet
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


class FilterObject(SpectralObject):
    """ Internal class for inheriting filter methods. """

    # Class-level sentinel
    #covariance_matrix: Final[None] = None

    def _init_from_spectral_data(self, data: Spectrum | SpectralSet) -> None:
        """ Initialize the target class from an arbitrary `Spectrum` or `SpectralSet`. """
        data = data.edges_to_zero().normalize()
        self.wavelength_nm = data.wavelength_nm
        self.spectral_dist = data.spectral_dist
        self.name = data.name

    def convert_for_photon_counter(self) -> Self:
        """
        Modifies the filter profile to account for photon-counting observation.
        See "Standard Photometric Systems" by Bessell, Michael S. (2005).
        """
        return (self * self.wavelength_nm).normalize()

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray) -> Self:
        """
        Directly uses the provided wavelength grid to create a new object. Non-strict!
        See `determine_at_wavelengths()` for the general case.
        """
        obj = deepcopy(self)
        min_nm = min(self.wavelength_nm[0], requested_wavelengths[0])
        max_nm = max(self.wavelength_nm[-1], requested_wavelengths[-1])
        obj.wavelength_nm = self._uniform_grid(min_nm, max_nm)
        obj.spectral_dist = np.zeros((obj.wavelength_nm.size, *self.spatial_shape))
        mask = np.searchsorted(obj.wavelength_nm, self.wavelength_nm)
        obj.spectral_dist[mask] = self.spectral_dist.copy()
        return obj

    def _apply_union(self, other: 'FilterObject') -> 'FilterSet':
        """ Internal logic of the `FilterObject` union. """
        filters = []
        # Populating combined filter list using the base other
        if isinstance(self, Filter):
            filters.append(self)
        elif isinstance(self, FilterSet):
            for i in range(len(self)):
                filters.append(self[i])
        else:
            raise TypeError(f'Cannot combine {type(self).__name__} with a `FilterObject`')
        # Populating combined filter list using the other object
        if isinstance(other, Filter):
            filters.append(other)
        elif isinstance(other, FilterSet):
            for i in range(len(other)):
                filters.append(other[i])
        else:
            raise TypeError(f'Cannot combine a `FilterObject` with {type(other).__name__}')
        return FilterSet.from_filters(filters)

    def __or__(self, other: Any) -> 'FilterSet':
        """ Combine this `FilterObject` with another `FilterObject` into a new `FilterSet`. """
        if isinstance(other, FilterObject):
            return self._apply_union(other)
        else:
            return NotImplemented

    def __ror__(self, other: Any) -> 'FilterSet':
        if isinstance(other, FilterObject):
            return other.__or__(self)
        else:
            return NotImplemented


class Filter(FilterObject, Spectrum):
    """
    Stores a filter profile as a normalized, edge-zeroed `Spectrum`.

    A `Filter` is created through the classmethod `get()` which supports an optional cache.
    Direct construction via `__init__()` always creates a new object (no caching).

    Attributes:
    - wavelength_nm     – spectral axis in nanometers
    - spectral_dist     – normalized transmission profile
    - covariance_matrix – always `None`
    - name              – human-readable identifier

    Example:
    ```
    >>> b = Filter.get('Generic_Bessell.B')
    >>> v = Filter.get('Generic_Bessell.V')
    >>> bv = b | v  # -> FilterSet containing B and V filters
    ```
    """

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        uncertainty: None = None,
        name: Any = None,
    ) -> None:
        """
        Creates a `Filter` from arrays of wavelength and transmission profile.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.
        An uncertainty cannot be assigned.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): normalized transmission profile
        - `name` (Any): human-readable identifier
        """
        spectrum = Spectrum(wavelength_nm, spectral_dist, name=name)
        self._init_from_spectral_data(spectrum)

    @staticmethod
    def get(filter_id: str) -> 'Filter':
        """ Use Spanish Virtual Observatory filter ID to create a `Filter` object. """
        return deepcopy(_cached_get(filter_id))


class FilterSet(FilterObject, SpectralSet):
    """
    Class to work with a set of filter profiles.

    Attributes:
    - wavelength_nm     – combined spectral axis covering all filter ranges
    - spectral_dist     – packed transmissions [len(nm), n_filters]
    - covariance_matrix – always `None` for FilterSet
    - name              – tuple of human-readable filter identifiers

    Example:
    ```
    >>> bvr = FilterSet.get('Generic_Bessell.B', 'Generic_Bessell.V', 'Generic_Bessell.R')
    >>> bv = Filter.get('Generic_Bessell.B') | Filter.get('Generic_Bessell.V')  # -> FilterSet
    >>> bvr = bv | Filter.get('Generic_Bessell.R')  # -> FilterSet
    ```
    """

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        uncertainty: None = None,
        name: Any = None,
    ) -> None:
        """
        Creates a `FilterSet` from arrays of wavelength and transmission profile.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.
        An uncertainty cannot be assigned.

        Args:
        - `wavelength_nm` (ArrayLike): combined spectral axis in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): transmission profile
        - `name` (Any): list of filter names or a human-readable identifier
        """
        spectral_set = SpectralSet(wavelength_nm, spectral_dist, name=name)
        self._init_from_spectral_data(spectral_set)

    @staticmethod
    def get(*filter_ids: str) -> 'FilterSet':
        """
        Use Spanish Virtual Observatory filter IDs to create a `FilterSet` object.
        """
        filters = []
        for filter_id in filter_ids:
            filters.append(Filter.get(filter_id))
        return FilterSet.from_filters(filters)

    @staticmethod
    def from_filters(filters: Sequence[Filter]) -> 'FilterSet':
        """
        Helper to build a `FilterSet` from a list of filters.
        Combines spectral axes into a single interval and packs transmissions.
        """
        # Getting the wavelength info and filter names
        names = []
        nm_min = np.inf
        nm_max = 0.
        for profile in filters:
            names.append(profile.name)
            nm_min = min(nm_min, float(profile.wavelength_nm[0]))
            nm_max = max(nm_max, float(profile.wavelength_nm[-1]))
        # Naming
        name: tuple[Any, ...] = tuple(names)
        # Matrix packing
        wavelength_nm = uniform_grid(nm_min, nm_max, nm_step, dtype=wavelength_nm_dtype)
        spectral_dist = np.zeros((len(wavelength_nm), len(filters)), dtype=np.float64)
        for i, profile in enumerate(filters):
            mask = (wavelength_nm >= int(profile.wavelength_nm[0])) & (wavelength_nm <= int(profile.wavelength_nm[-1]))
            spectral_dist[mask, i] = profile.spectral_dist
        return FilterSet(wavelength_nm, spectral_dist, name=name)

    @property
    def matrix(self) -> npt.NDArray[np.floating]:
        """
        Transforms filter profiles' transmission spectral distribution
        into a matrix that simplifies the accounting of the grid step:
        A = T^T * Δλ  ->  y = A x, where x is a spectrum and y is a photospectrum
        """
        return self.spectral_dist.T * nm_step

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
