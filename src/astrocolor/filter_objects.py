"""
Filter profiles and filter sets.

`Filter` loads, normalizes and caches SVO FPS profiles. `FilterSet` packs
several profiles on a common wavelength grid and exposes their matrix form for
synthetic photometry.
"""

from collections.abc import Iterator, Sequence
from copy import copy, deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Self, cast, overload, override

import numpy as np
import numpy.typing as npt

from .auxiliary import uniform_grid
from .config import Config
from .core import BaseObject, nm_step, wavelength_nm_dtype
from .errors import FilterNetworkError, FilterNotFoundError
from .filter_loader import (
    fetch_from_fps_raw,
    get_filter_folders,
    get_parameter,
    get_profile,
)
from .spectral_objects import SpectralObject, SpectralSet, Spectrum
from .uncertainty import Uncertainty

type _FilterCacheKey = tuple[Path, Path, Path | None, bool, str]


def _filter_cache_key() -> _FilterCacheKey:
    """
    Snapshot of `Config` settings that affect filter lookup.

    The snapshot is part of the `Filter.get()` cache key, so changing a lookup
    folder, network access or SVO endpoint does not return a profile loaded
    under the old configuration.
    """
    return (
        Config.get_bundled_filters_folder(),
        Config.get_cached_filters_folder(),
        Config.get_custom_filters_folder(),
        Config.allow_internet_access,
        Config.svo_fps_url,
    )


class FilterObject(SpectralObject):
    """
    Internal class for inheriting filter methods.

    `Filter` and `FilterSet` store normalized transmission profiles. This is a
    construction invariant: constructors, `get()`, `normalized()`, slicing and
    unions return normalized profiles, while arithmetic operators (`+`, `-`,
    `*`, `/`) return plain spectral objects and preserve the mathematical
    result. A scaled or combined profile is therefore never silently
    renormalized.
    """

    @override
    @classmethod
    def _from_computed_arrays(
        cls,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.NDArray[np.floating],
        uncertainty: Uncertainty | npt.ArrayLike | None,
        name: object = None
    ) -> Self:
        """ Construction helper for filter profiles; the result is normalized. """
        _ = uncertainty
        return cls(wavelength_nm, spectral_dist, name=name)

    def _as_spectral(self) -> Spectrum | SpectralSet:
        """
        Returns a trusted spectral view of the profile without re-normalization.

        Filter constructors validate the grid and normalize new profiles;
        arithmetic must not renormalize an already normalized (or deliberately
        clipped) profile, so a constructor round trip here would alter results.
        The returned object shares `spectral_dist`; it is an operator operand,
        not a mutable copy.
        """
        if self.ndim == 1:
            return Spectrum._from_trusted_arrays(
                self.wavelength_nm, self.spectral_dist, None, self.name
            )
        return SpectralSet._from_trusted_arrays(
            self.wavelength_nm, self.spectral_dist, None, self.name
        )

    @override
    def _as_operand(self) -> 'BaseObject':
        """ Filters participate in arithmetic as plain spectral objects. """
        return self._as_spectral()

    @override
    def __add__(self, other: 'BaseObject | npt.ArrayLike') -> SpectralObject:
        """ Returns the mathematical sum as a spectral object, without re-normalization. """
        return self._as_spectral() + other

    @override
    def __sub__(self, other: 'BaseObject | npt.ArrayLike') -> SpectralObject:
        """ Returns the mathematical difference as a spectral object, without re-normalization. """
        return self._as_spectral() - other

    @override
    def __mul__(self, other: 'BaseObject | npt.ArrayLike') -> SpectralObject:
        """ Returns the mathematical product as a spectral object, without re-normalization. """
        return self._as_spectral() * other

    @override
    def __truediv__(self, other: 'BaseObject | npt.ArrayLike') -> SpectralObject:
        """ Returns the mathematical quotient as a spectral object, without re-normalization. """
        return self._as_spectral() / other

    @override
    def normalized(self) -> Self:
        """
        Returns a normalized profile of the same filter family.

        Profiles are normalized by construction, so this is idempotent for
        canonical objects. Renormalization is still performed because a profile
        may have been clipped by `get_spectrometry(..., strictly=True)`.
        """
        # Calling the constructor here would re-enter `normalize()` and
        # `normalized()` recursively, so the result is built from the base
        # implementation and wrapped as a filter object instead.
        output = copy(self)
        result = SpectralObject.normalized(self)
        output.spectral_dist = result.spectral_dist
        output.uncertainty = None
        return output

    def convert_for_photon_counter(self) -> Self:
        """
        Returns a new filter profile adjusted for a photon-counting observation.
        See "Standard Photometric Systems" by Bessell, Michael S. (2005).

        A photon-counting detector measures a count rate proportional to
        the integral of the energy flux weighted by `wavelength * transmission`:
        a photon at a longer wavelength carries less energy, so the effective
        response is `lambda * T(lambda)`. The profile is normalized afterwards,
        which only fixes the arbitrary scale of a transmission curve. This order
        (wavelength weighting first, normalization second) is what changes the
        effective shape of the profile between photon and energy counting.
        """
        weighted = self._as_spectral() * self.wavelength_nm
        return type(self)(weighted.wavelength_nm, weighted.spectral_dist, name=self.name)

    @override
    def convert_from_photon_spectral_density(self) -> Self:
        """
        Returns a filter profile converted from photon to energy spectral density.
        The profile keeps the filter family and stays normalized.
        """
        converted = self._as_spectral().convert_from_photon_spectral_density()
        return type(self)(converted.wavelength_nm, converted.spectral_dist, name=self.name)

    @override
    def convert_from_energy_spectral_density_per_frequency(self) -> Self:
        """
        Returns a filter profile converted from frequency to wavelength density.
        The profile keeps the filter family and stays normalized.
        """
        converted = self._as_spectral().convert_from_energy_spectral_density_per_frequency()
        return type(self)(converted.wavelength_nm, converted.spectral_dist, name=self.name)

    @override
    def determine_at_trusted_wavelengths(self,
        requested_wavelengths: npt.NDArray[np.integer]
    ) -> Self:
        """
        Extends the filter profile to cover the requested wavelength grid,
        filling the new range with zeros. Non-strict.
        See `get_spectrometry()` for the general case.
        """
        obj = deepcopy(self)
        self_nm_0 = cast(int, self.wavelength_nm[0])
        self_nm_1 = cast(int, self.wavelength_nm[-1])
        other_nm_0 = cast(int, requested_wavelengths[0])
        other_nm_1 = cast(int, requested_wavelengths[-1])
        min_nm = min(self_nm_0, other_nm_0)
        max_nm = max(self_nm_1, other_nm_1)
        obj.wavelength_nm = self.uniform_grid(min_nm, max_nm)
        obj.spectral_dist = np.zeros((obj.wavelength_nm.size, *self.spatial_shape))
        mask = np.searchsorted(obj.wavelength_nm, self.wavelength_nm)
        obj.spectral_dist[mask] = self.spectral_dist.copy()
        return obj

    def _apply_union(self, other: 'FilterObject') -> 'FilterSet':
        """ Internal logic of the `FilterObject` union. """
        filters: list[Filter] = []
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

    def __or__(self, other: object) -> 'FilterSet':
        """ Combine this `FilterObject` with another `FilterObject` into a new `FilterSet`. """
        if isinstance(other, FilterObject):
            return self._apply_union(other)
        else:
            return NotImplemented

    def __ror__(self, other: object) -> 'FilterSet':
        if isinstance(other, FilterObject):
            return other.__or__(self)
        else:
            return NotImplemented


class Filter(FilterObject, Spectrum):
    """
    Stores a filter profile as a normalized, edge-zeroed `Spectrum`.

    A `Filter` is created through the classmethod `get()`, which uses an internal cache.
    Direct construction via `__init__()` always creates a new object (no caching).
    Arithmetic operators return plain `Spectrum` objects (see `FilterObject`);
    wrap their result in `Filter` to obtain a normalized profile again.

    Attributes:
    - wavelength_nm     – spectral axis in nanometers
    - spectral_dist     – normalized transmission profile
    - covariance_matrix – always `None`
    - name              – human-readable identifier

    Example:
    ```
    >>> b = Filter.get('Generic/Bessell.B')
    >>> v = Filter.get('Generic/Bessell.V')
    >>> bv = b | v  # -> FilterSet containing B and V filters
    ```
    """

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        name: object = None,
    ) -> None:
        """
        Creates a `Filter` from arrays of wavelength and transmission profile.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.
        An uncertainty cannot be assigned.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): normalized transmission profile
        - `name` (object): human-readable identifier
        """
        super().__init__(wavelength_nm, spectral_dist, name=name)
        self.edges_to_zero()
        self.normalize()

    @classmethod
    def _from_normalized_profile(
        cls,
        wavelength_nm: npt.NDArray[np.integer],
        spectral_dist: npt.NDArray[np.floating],
        name: object = None
    ) -> Self:
        """
        Builds a `Filter` from an already normalized, edge-zeroed profile.

        Used for profiles extracted from a `FilterSet`; re-normalizing them
        would change the last bits of `float64` and break the round trip
        `FilterSet[i] == Filter.get(...)`. The caller must pass owned arrays.
        """
        return cls._from_trusted_arrays(wavelength_nm, spectral_dist, None, name)

    @staticmethod
    def get(filter_id: str) -> 'Filter':
        """
        Use Spanish Virtual Observatory filter ID to create a `Filter` object.
        Searches: bundled filters → custom folder → cached downloads → SVO FPS
        (network fetch if Config.allow_internet_access is True).
        The cache respects the current filter lookup configuration.
        """
        return deepcopy(_cached_get(filter_id, _filter_cache_key()))


class FilterSet(FilterObject, SpectralSet):
    """
    Class to work with a set of filter profiles.

    Indexing returns a single `Filter` for an integer and a sliced `FilterSet`
    for a slice. Arithmetic operators return plain `SpectralSet` objects (see
    `FilterObject`); wrap their result in `FilterSet` to normalize it again.

    Attributes:
    - wavelength_nm     – combined spectral axis covering all filter ranges
    - spectral_dist     – packed transmissions [len(nm), n_filters]
    - covariance_matrix – always `None` for FilterSet
    - name              – tuple of human-readable filter identifiers

    Example:
    ```
    >>> bvr = FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
    >>> bv = Filter.get('Generic/Bessell.B') | Filter.get('Generic/Bessell.V')  # -> FilterSet
    >>> bvr = bv | Filter.get('Generic/Bessell.R')  # -> FilterSet
    >>> vb = bvr[::-1][:2]  # -> FilterSet with V and B
    ```
    """

    def __init__(
        self,
        wavelength_nm: npt.ArrayLike,
        spectral_dist: npt.ArrayLike,
        name: object = None,
    ) -> None:
        """
        Creates a `FilterSet` from arrays of wavelength and transmission profile.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.
        An uncertainty cannot be assigned.

        Args:
        - `wavelength_nm` (ArrayLike): combined spectral axis in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): transmission profile
        - `name` (object): list of filter names or a human-readable identifier
        """
        super().__init__(wavelength_nm, spectral_dist, name=name)
        self.edges_to_zero()
        self.normalize()

    @classmethod
    def _from_normalized_profiles(
        cls,
        wavelength_nm: npt.NDArray[np.integer],
        spectral_dist: npt.NDArray[np.floating],
        name: object = None
    ) -> Self:
        """
        Builds a `FilterSet` from already normalized, edge-zeroed profiles.

        Used by `from_filters()` and by slicing; re-normalization would change
        the last bits of `float64` and break exact round trips. The caller must
        pass owned arrays.
        """
        return cls._from_trusted_arrays(wavelength_nm, spectral_dist, None, name)

    @staticmethod
    def get(*filter_ids: str) -> 'FilterSet':
        """
        Use Spanish Virtual Observatory filter IDs to create a `FilterSet` object.
        """
        filters: list[Filter] = []
        for filter_id in filter_ids:
            filters.append(Filter.get(filter_id))
        return FilterSet.from_filters(filters)

    @staticmethod
    def from_filters(filters: Sequence[Filter]) -> 'FilterSet':
        """
        Helper to build a `FilterSet` from a list of filters.
        Combines spectral axes into a single interval and packs transmissions.
        Profiles are normalized already, so they are packed without a second
        normalization; this keeps `FilterSet[i] == Filter.get(...)` exact.
        """
        # Getting the wavelength info and filter names
        names: list[object] = []
        nm_min = np.inf
        nm_max = 0.
        for profile in filters:
            names.append(profile.name)
            nm_0 = cast(int, profile.wavelength_nm[0])
            nm_1 = cast(int, profile.wavelength_nm[-1])
            nm_min = min(nm_min, nm_0)
            nm_max = max(nm_max, nm_1)
        # Naming
        name: tuple[object, ...] = tuple(names)
        # Matrix packing
        wavelength_nm = uniform_grid(nm_min, nm_max, nm_step, dtype=wavelength_nm_dtype)
        spectral_dist = cast(npt.NDArray[np.floating], np.zeros((len(wavelength_nm), len(filters)), dtype=np.float64))
        for i, profile in enumerate(filters):
            nm_0 = cast(int, profile.wavelength_nm[0])
            nm_1 = cast(int, profile.wavelength_nm[-1])
            mask = (wavelength_nm >= nm_0) & (wavelength_nm <= nm_1)
            spectral_dist[mask, i] = profile.spectral_dist
        return FilterSet._from_normalized_profiles(wavelength_nm, spectral_dist, name=name)

    @property
    def matrix(self) -> npt.NDArray[np.floating]:
        """
        Transforms filter profiles' transmission spectral distribution
        into a matrix that simplifies the accounting of the grid step:
        A = T^T * Δλ  ->  y = A x, where x is a spectrum and y is a photospectrum
        """
        return self.spectral_dist.T * nm_step

    def _name_at(self, index: int) -> object:
        """ Returns the name of a single filter, if names are stored per filter. """
        if isinstance(self.name, tuple):
            names = cast(tuple[object, ...], self.name)
            if len(names) == len(self):
                return names[index]
        return None

    def _name_slice(self, index: slice) -> object:
        """ Returns the names of a subset, or the set name if it is not per-filter. """
        if not isinstance(self.name, tuple):
            return self.name
        names = cast(tuple[object, ...], self.name)
        if len(names) == len(self):
            return names[index]
        return names

    def _get_filter(self, index: int) -> Filter:
        """ Returns one normalized profile with extra zeros trimmed off. """
        profile = self.spectral_dist[:, index]
        non_zero = cast(npt.NDArray[np.intp], np.flatnonzero(profile))
        if non_zero.size == 0:
            # A degenerate all-zero profile cannot be trimmed; keep a stub.
            return Filter.stub(self._name_at(index))
        start = max(cast(int, non_zero[0]) - 1, 0)
        end = min(cast(int, non_zero[-1]) + 2, profile.size)
        return Filter._from_normalized_profile(  # pyright: ignore[reportPrivateUsage]
            self.wavelength_nm[start:end].copy(),
            profile[start:end].copy(),
            self._name_at(index)
        )

    def _get_filter_set(self, index: slice) -> 'FilterSet':
        """ Returns a subset of columns as a normalized `FilterSet`. """
        subset = self.spectral_dist[:, index]
        names = self._name_slice(index)
        if subset.shape[1] == 0:
            # Keep sequence semantics for empty slices instead of raising.
            empty = np.zeros((0, 0), dtype=np.float64)
            return FilterSet._from_normalized_profiles(
                self.wavelength_nm[:0].copy(), empty, names
            )
        non_zero_mask = cast(
            npt.NDArray[np.bool_],
            np.abs(subset).sum(axis=1) != 0.,
        )
        non_zero_rows = cast(npt.NDArray[np.intp], np.flatnonzero(non_zero_mask))
        if non_zero_rows.size == 0:
            # Degenerate all-zero selection: keep the columns and the grid as is.
            return FilterSet._from_normalized_profiles(
                self.wavelength_nm.copy(), subset.copy(), names
            )
        # Trim leading and trailing all-zero rows to one, keeping inner gaps.
        start = max(cast(int, non_zero_rows[0]) - 1, 0)
        end = min(cast(int, non_zero_rows[-1]) + 2, cast(int, subset.shape[0]))
        return FilterSet._from_normalized_profiles(
            self.wavelength_nm[start:end].copy(),
            subset[start:end].copy(),
            names
        )

    @overload
    def __getitem__(self, index: int) -> Filter: ...

    @overload
    def __getitem__(self, index: slice) -> 'FilterSet': ...

    @override
    def __getitem__(self, index: int | slice) -> 'Filter | FilterSet':  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Returns a `Filter` for an integer index or a `FilterSet` for a slice.

        Leading and trailing zero rows are trimmed to one, so the result equals
        the corresponding `Filter.get()` / `FilterSet.from_filters()` exactly.
        Profiles are not re-normalized.
        """
        if isinstance(index, slice):
            return self._get_filter_set(index)
        return self._get_filter(index)

    @override
    def __iter__(self) -> Iterator[Filter]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """ Iterates over the normalized filter profiles. """
        for index in range(len(self)):
            yield self._get_filter(index)


@lru_cache(maxsize=128)
def _cached_get(filter_id: str, config_key: _FilterCacheKey) -> 'Filter':
    """
    Returns a cached filter object for a filter lookup configuration.
    Search order:
      1. Cached in lru_cache (handled by decorator)
      2. Bundled filters shipped with the library
      3. Custom user-provided folder (if Config.get_custom_filters_folder is set)
      4. Local cache (downloaded from SVO FPS, saved locally if allowed)
      5. SVO FPS network fetch (if Config.allow_internet_access is True)
    `config_key` makes the cache sensitive to changes in these settings.
    This has been separated out to create copies of the results and avoid mutations.
    """
    allow_internet_access = config_key[3]
    filter_folders = get_filter_folders()

    try:
        facility, instrument_band = filter_id.split('/')
    except ValueError as e:
        raise FilterNotFoundError(filter_id) from e

    local_path: Path | None = None
    for root in filter_folders:
        candidate = next((root / facility).glob(f'{instrument_band}.*'), None)
        if candidate is not None:
            local_path = candidate
            break

    if local_path is not None:
        profile = np.loadtxt(local_path).T
        wl = cast(npt.NDArray[np.floating], profile[0])
        sd = cast(npt.NDArray[np.floating], profile[1])

        # - Extension parsing
        extension = local_path.suffix[1:]

        # Wavelength units:
        # 'N' (nm) — nanometer (default)
        # 'A' (Å) — angstrom
        # 'U' (μ) — micrometer
        wavelength_unit = 'Nanometer'
        if 'A' in extension:
            wavelength_unit = 'Angstrom'
        elif 'U' in extension:
            wavelength_unit = 'Micrometer'

        # Detector types:
        # 'E' — energy counter per wavelength (default)
        # 'P' — photon counter
        # 'J' — energy counter per frequency
        detector_type = 0
        if 'P' in extension:
            detector_type = 1
        if 'J' in extension:
            detector_type = 2

    else:
        # Local file not found: fetch from SVO FPS and save locally if allowed
        if not allow_internet_access:
            raise FilterNotFoundError(filter_id)

        try:
            xml_content = fetch_from_fps_raw(filter_id)
            wl, sd = get_profile(xml_content, filter_id)
            wavelength_unit = get_parameter(xml_content, 'WavelengthUnit', default='Angstrom')
            detector_type = int(get_parameter(xml_content, 'DetectorType', default='0'))

            # Extension generation
            extension = 'txt'
            if wavelength_unit == 'Angstrom':
                extension += 'A'
            if detector_type == 1:
                extension += 'P'

            # Save to cache folder for future use
            cached_facility = filter_folders[-1] / facility
            cached_facility.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                f'{cached_facility}/{instrument_band}.{extension}',
                np.column_stack([wl, sd]),
                fmt='%.3f\t%.6g',
            )
        except FilterNetworkError:
            raise FilterNotFoundError(filter_id)

    # Convert to nanometers
    match wavelength_unit:
        case 'Angstrom':
            wl *= 0.1
        case 'Micrometer':
            wl *= 1000.
        case _:
            pass

    # Create filter object
    filter_obj = Filter(wl, sd, name=filter_id)

    # Apply photon-counting conversion if the filter uses a Photon counter
    match detector_type:
        case 1:
            filter_obj = filter_obj.convert_for_photon_counter()
        # TODO: add inverse conversion from energy per frequency to energy per wavelength
        # case 2:
        #    filter_obj = filter_obj.convert_for_photon_counter()
        case _:
            pass

    return filter_obj
