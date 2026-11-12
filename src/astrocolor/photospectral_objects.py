"""
Photometric measurements through filter sets.

`PhotospectralObject` stores one value per filter with optional spatial axes;
`Photospectrum`, `PhotospectralSet` and `PhotospectralCube` cover 1D, 2D and 3D
data. Multiplication and division operate on reconstructed spectra.
"""

from copy import deepcopy
from typing import Self, cast, override

import numpy as np
import numpy.typing as npt

from .algebra import make_same_ndim
from .config import Config
from .core import (
    BaseObject,
    Cube,
    Item,
    Set,
    UncertaintyHandler,
    ValueHandler,
    spectral_dist_dtype,
)
from .errors import (
    InconsistentAxesError,
    InconsistentDimensionError,
    non_finite_values_warning,
)

# No dependency on .spectral_objects to avoid a cycle!
from .filter_objects import FilterSet
from .uncertainty import Uncertainty


class PhotospectralObject(BaseObject):
    """
    Internal parent class for Photospectrum (1D), PhotospectralSet (2D) and PhotospectralCube (3D).

    Multiplication and division with a SpectralObject or another PhotospectralObject
    are performed on reconstructed spectra: both operands are reconstructed on the
    filter grid of this object, the spectra are multiplied/divided point-wise,
    and the result is observed back through `filter_set`.
    Addition and subtraction remain element-wise, because photometric integrals
    are linear.

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `uncertainty` (Uncertainty): optional compact uncertainty (variance or covariance)
    - `covariance_matrix` (NDArray): optional full matrix, materialized on access
    - `name` (object): human-readable identifier
    """

    def __init__(
        self,
        filter_set: FilterSet,
        spectral_dist: npt.ArrayLike,
        uncertainty: npt.ArrayLike | Uncertainty | None = None,
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
        self.uncertainty: Uncertainty | None = None
        # Spatial axis check
        self.spectral_dist: npt.NDArray[np.floating] = np.array(spectral_dist, dtype=spectral_dist_dtype)
        if self.ndim != self.spectral_dist.ndim:
            raise InconsistentDimensionError(self.ndim, self.spectral_dist.ndim, self.name)
        non_finite = ~np.isfinite(self.spectral_dist)
        if np.any(non_finite):
            self.spectral_dist[non_finite] = 0.
            non_finite_values_warning('spectral_dist', self.name)
        # Spectral axis check
        self.filter_set: FilterSet = filter_set
        if (len_filters := len(self.filter_set)) != (len_values := len(self.spectral_dist)):
            raise InconsistentAxesError(len_filters, len_values, self.name)
        # Uncertainty check
        if Config.ignore_uncertainty_for_cubes and self.ndim == 3 or uncertainty is None:
            self.uncertainty = None
        else:
            self.uncertainty = Uncertainty.from_input(
                uncertainty, self.spectral_dist.shape, name
            )

    @override
    def _comparison_wavelengths(self) -> npt.ArrayLike:
        """ Photospectral objects use the filter-set grid as their spectral axis. """
        return self.filter_set.wavelength_nm

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

    def determine_at_trusted_wavelengths(self,
        requested_wavelengths: npt.NDArray[np.integer],
        attach_photospectral_obj: bool = True
    ):
        """
        Directly uses the provided wavelength grid to create a new object. Non-strict!
        See `get_spectrometry()` for the general case.
        """
        from .reconstructed_objects import spectral_reconstruction
        obj = spectral_reconstruction(self, requested_wavelengths, attach_photospectral_obj=attach_photospectral_obj)
        return obj

    @override
    def _apply_element_wise_operation(
        self,
        other: 'BaseObject',
        value_handling: ValueHandler,
        uncertainty_handling: UncertaintyHandler
    ) -> 'PhotospectralObject':
        """
        Returns a new PhotospectralObject formed from a linear element-wise
        operation (addition or subtraction) with a SpectralObject or another
        PhotospectralObject.

        The filter set of the second object, if it does not match, is converted
        to the filter set of the first object!
        """
        filter_set = self.filter_set
        from .spectral_objects import SpectralObject
        if isinstance(other, PhotospectralObject):
            if other.filter_set != filter_set:
                # Converting to a PhotospectralObject of the same filter set
                from .measurements import get_photometry
                other = get_photometry(other, filter_set)
        elif isinstance(other, SpectralObject):
            # Converting to a PhotospectralObject of the same filter set
            from .measurements import get_photometry
            other = get_photometry(other, filter_set)
        else:
            return NotImplemented
        value = value_handling(self.spectral_dist, other.spectral_dist)
        uncertainty = uncertainty_handling(
            self.spectral_dist, self.uncertainty,
            other.spectral_dist, other.uncertainty,
        )
        higher_dim = (self, other)[self.ndim < other.ndim]
        return higher_dim.__class__(filter_set, value, uncertainty, name=higher_dim.name)

    @override
    def _apply_spectral_operation(
        self,
        other: 'BaseObject',
        value_handling: ValueHandler,
        uncertainty_handling: UncertaintyHandler
    ) -> 'PhotospectralObject':
        """
        Returns a new PhotospectralObject formed from an operation that is
        defined on the spectral distribution (multiplication or division).

        Both operands are reconstructed into continuous spectra on the filter
        grid of this object. Reconstructing both of them on one common
        wavelength interval is preferred over extrapolating or clipping an
        already reconstructed spectrum: an attached reconstructed object is
        therefore reconstructed again from its stored source photometry. The
        operation is applied point-wise, and the result is wrapped as a plain
        `Spectrum`/`SpectralSet`/`SpectralCube` before being observed through
        `self.filter_set`. The final `get_photometry()` therefore always measures
        the product spectrum itself.
        """
        from .measurements import get_photometry, get_spectrometry
        from .spectral_objects import (
            SpectralCube,
            SpectralObject,
            SpectralSet,
            Spectrum,
        )

        if not isinstance(other, (SpectralObject, PhotospectralObject)):
            return NotImplemented

        filter_set = self.filter_set
        target_nm = filter_set.wavelength_nm

        left = get_spectrometry(self, target_nm, strictly=True, attach_photospectral_obj=False)
        right = get_spectrometry(other, target_nm, strictly=True, attach_photospectral_obj=False)
        left_value, right_value = make_same_ndim(left.spectral_dist, right.spectral_dist)
        value = value_handling(left_value, right_value)
        # Confidence bands for spectral sets and cubes are not computed by the reconstruction.
        uncertainty = None
        if left.ndim == 1 and right.ndim == 1:
            uncertainty = uncertainty_handling(
                left.spectral_dist, left.uncertainty,
                right.spectral_dist, right.uncertainty,
            )
        higher = left if left.ndim >= right.ndim else right
        product_class = (Spectrum, SpectralSet, SpectralCube)[higher.ndim - 1]
        product = product_class(target_nm, value, uncertainty, name=higher.name)
        return get_photometry(product, filter_set)

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
    - `uncertainty` (Uncertainty): optional compact uncertainty (variance or covariance)
    - `covariance_matrix` (NDArray): optional full matrix, materialized on access
    - `name` (object): human-readable identifier
    """


class PhotospectralSet(PhotospectralObject, Set['Photospectrum', 'PhotospectralCube']):
    """
    Class to work with set of filters measurements (2D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `uncertainty` (Uncertainty): optional compact uncertainty (variance or covariance)
    - `covariance_matrix` (NDArray): optional full matrix, materialized on access
    - `name` (object): human-readable identifier
    - `spatial_size` (int): spatial axis length
    """


class PhotospectralCube(PhotospectralObject, Cube['PhotospectralSet']):
    """
    Class to work with set of filters measurements (3D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (NDArray): array of "brightness" in energy density units (not a photon counter)
    - `uncertainty` (Uncertainty): optional compact uncertainty (variance or covariance)
    - `covariance_matrix` (NDArray): optional full matrix, materialized on access
    - `name` (object): human-readable identifier
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `spatial_size` (int): number of pixels
    """

    @override
    def flatten(self) -> PhotospectralSet:
        """ Returns a PhotospectralSet with linearized spatial axis """
        value = self.spectral_dist.reshape(self.spectral_size, self.spatial_size).copy()
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = deepcopy(self.uncertainty.reshape_spatial((self.spatial_size,)))
        output = PhotospectralSet._from_trusted_arrays(self.filter_set.wavelength_nm, value, uncertainty, self.name)
        output.filter_set = self.filter_set
        return output
