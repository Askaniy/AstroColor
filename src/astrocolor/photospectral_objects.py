from collections.abc import Callable
from copy import deepcopy
from typing import Self, cast, override

import numpy as np
import numpy.typing as npt

from .config import Config
from .core import BaseObject, Cube, Item, Set, spectral_dist_dtype
from .errors import (
    InconsistentAxesError,
    InconsistentDimensionError,
    InconsistentUncertaintyShapeError,
    InconsistentUncertaintySizeError,
    nan_values_warning,
)

# No dependency on .spectral_objects to avoid a cycle!
from .filter_objects import FilterSet


class PhotospectralObject(BaseObject):
    """
    Internal parent class for Photospectrum (1D), PhotospectralSet (2D) and PhotospectralCube (3D).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    """

    def __init__(
        self,
        filter_set: FilterSet,
        spectral_dist: npt.ArrayLike,
        uncertainty: npt.ArrayLike | None = None,
        name: object = None
    ) -> None:
        """
        Args:
        - `filter_set` (FilterSet): instance of the class storing filter profiles
        - `spectral_dist` (ArrayLike): array of "brightness" in energy density units (not a photon counter)
        - `uncertainty` (ArrayLike): optional array of standard deviations or a covariance matrix
        - `name` (object): human-readable identifier
        """
        self.name: object = name
        # Spatial axis check
        self.spectral_dist: npt.NDArray[np.floating] = np.array(spectral_dist, dtype=spectral_dist_dtype)
        if self.ndim != self.spectral_dist.ndim:
            raise InconsistentDimensionError(self.ndim, self.spectral_dist.ndim, self.name)
        if np.any(np.isnan(self.spectral_dist)):
            self.spectral_dist = np.nan_to_num(self.spectral_dist)
            nan_values_warning('spectral_dist', self.name)
        # Spectral axis check
        self.filter_set: FilterSet = filter_set
        if (len_filters := len(self.filter_set)) != (len_values := len(self.spectral_dist)):
            raise InconsistentAxesError(len_filters, len_values, self.name)
        # Uncertainty check
        self.covariance_matrix: npt.NDArray[np.floating] | None = None
        if Config.ignore_uncertainty_for_cubes and self.ndim == 3:
            uncertainty = None
        if uncertainty is not None:
            uncertainty = np.array(uncertainty, dtype=spectral_dist_dtype)
            if (len_error := len(uncertainty)) != len_values:
                raise InconsistentUncertaintySizeError(len_error, len_values, name)
            if uncertainty.ndim == self.spectral_dist.ndim:
                self.covariance_matrix = np.diag(uncertainty**2)
            elif uncertainty.ndim == self.spectral_dist.ndim + 1:
                self.covariance_matrix = uncertainty
            else:
                raise InconsistentUncertaintyShapeError(uncertainty.ndim, self.spectral_dist.ndim, name)

    @override
    @classmethod
    def stub(cls, name: object = None) -> Self:
        """ Initializes an object in case of the data problems """
        stub_filter_set = FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V')
        return cls(stub_filter_set, np.zeros((2, 1, 1)[:cls.ndim]), name=name)

    @override
    def convert_from_photon_spectral_density(self) -> Self:
        """
        Returns a new PhotospectralObject converted from photon spectral density
        to energy spectral density, using the fact that E = h c / λ.
        """
        if len(self.filter_set) > 1:
            profiles = self.filter_set.normalized()
            scale_factors = (profiles / profiles.wavelength_nm).integrate()
            scale_factors = cast(npt.NDArray[np.floating], scale_factors) # not a float: len(filter_set) > 1
            scale_factors = cast(npt.NDArray[np.floating], scale_factors / scale_factors.mean())
            return self * scale_factors
        else:
            return deepcopy(self)

    @override
    def convert_from_energy_spectral_density_per_frequency(self) -> Self:
        """
        Returns a new PhotospectralObject converted from frequency spectral density
        to energy spectral density, using the fact that f_λ = f_ν c / λ².
        """
        if len(self.filter_set) > 1:
            profiles = self.filter_set.normalized()
            scale_factors = (profiles / profiles.wavelength_nm**2).integrate()
            scale_factors = cast(npt.NDArray[np.floating], scale_factors) # not a float: len(filter_set) > 1
            scale_factors = cast(npt.NDArray[np.floating], scale_factors / scale_factors.mean())
            return self * scale_factors
        else:
            return deepcopy(self)

    def determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray[np.integer]):
        """
        Directly uses the provided wavelength grid to create a new object. Non-strict!
        See `get_spectrometry()` for the general case.
        """
        from .reconstructed_objects import spectral_reconstruction
        obj = spectral_reconstruction(self, requested_wavelengths)
        return obj

    @override
    def _apply_element_wise_operation(
        self,
        other: 'BaseObject',
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating]],
        error_handling: Callable[[npt.ArrayLike, npt.ArrayLike | None, npt.ArrayLike, npt.ArrayLike | None], npt.NDArray[np.floating] | None]
    ) -> 'PhotospectralObject':
        """
        Returns a new PhotospectralObject formed from element-wise operation with
        a SpectralObject or another PhotospectralObject. Operations between objects
        of the same dimensionality and all (photo)spectrum operations are supported.

        The filter system of the second object, if it does not match, is converted
        to the filter system of the first object!
        """
        filter_set = self.filter_set
        from .spectral_objects import SpectralObject
        if isinstance(other, SpectralObject) or (isinstance(other, PhotospectralObject) and other.filter_set != filter_set):
            # Converting to a PhotospectralObject of the same filter system
            from .measurements import get_photometry
            other = get_photometry(other, filter_set)
        else:
            return NotImplemented
        value = value_handling(self.spectral_dist, other.spectral_dist)
        error = error_handling(self.spectral_dist, self.covariance_matrix, other.spectral_dist, other.covariance_matrix)
        higher_dim = (self, other)[self.ndim < other.ndim]
        return higher_dim.__class__(filter_set, value, error, name=higher_dim.name)

    @override
    def _generate_repr_config(self) -> dict[str, str]:
        """
        Generates configuration for string representation in __repr__().
        Replaces `wavelength_nm` with information about `FilterSet`.
        """
        old_repr_config = super()._generate_repr_config()
        new_repr_config: dict[str, str] = {}
        for key in old_repr_config:
            if key == 'wavelength_nm':
                new_repr_config['filter_set'] = self.filter_set.__repr__()
            else:
                new_repr_config[key] = old_repr_config[key]
        return new_repr_config



class Photospectrum(PhotospectralObject, Item['PhotospectralSet']):
    """
    Class to work with set of filters measurements (1D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    """


class PhotospectralSet(PhotospectralObject, Set['Photospectrum', 'PhotospectralCube']):
    """
    Class to work with set of filters measurements (2D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `size` (int): spatial axis length
    """


class PhotospectralCube(PhotospectralObject, Cube['PhotospectralSet']):
    """
    Class to work with set of filters measurements (3D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix` (NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (object): human-readable identifier
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `size` (int): number of pixels
    """

    @override
    def flatten(self) -> PhotospectralSet:
        """ Returns a PhotospectralSet with linearized spatial axis """
        value = self.spectral_dist.reshape(self.spectral_size, self.spatial_size)
        error = None
        if self.covariance_matrix is not None:
            error = self.covariance_matrix.reshape(self.spectral_size, self.spectral_size, self.spatial_size)
        return PhotospectralSet(self.filter_set, value, error, self.name)
