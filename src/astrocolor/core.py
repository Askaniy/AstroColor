from collections.abc import Callable, Iterator
from copy import copy, deepcopy
from math import prod
from typing import ClassVar, Final, Self, TypeVar, cast, override

import numpy as np
import numpy.typing as npt

from .algebra import (
    add_value,
    div_value,
    mul_value,
    sub_value,
)
from .auxiliary import (
    get_extremal_grid_endpoints,
    grid_endpoints_preprocessing,
    repr_generator,
    spatial_downscaling,
    uniform_grid,
)
from .uncertainty import (
    Uncertainty,
    add_uncertainty,
    div_uncertainty,
    mul_uncertainty,
    sub_uncertainty,
)

ValueHandler = Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating]]
UncertaintyHandler = Callable[
    [npt.ArrayLike, Uncertainty | None, npt.ArrayLike, Uncertainty | None],
    Uncertainty | None,
]

# For the sake of simplifying work with the spectrum,
# its discretization step is fixed and frozen.
nm_step: Final[int] = 5  # nm

# Wavelength and brightness axis storage data type
wavelength_nm_dtype: Final[npt.DTypeLike] = np.int64
spectral_dist_dtype: Final[npt.DTypeLike] = np.float64
uncertainty_dtype: Final[npt.DTypeLike] = np.float64

# Maximum wavelength, the clipping level
# It is important that there be no overflow when raising a number to the second power
# See convert_from_energy_spectral_density_per_frequency()
nm_red_limit: Final[int] = int(cast(float, np.sqrt(np.iinfo(wavelength_nm_dtype).max))) # 3037000499 nm ~ 3 meters


class UniformSpectralGrid:
    """
    Internal class for inheriting spectral grid processing methods.
    """

    nm_step: Final[int] = nm_step
    nm_red_limit: Final[int] = nm_red_limit

    def get_extremal_grid_endpoints(
        self,
        requested_wavelengths: npt.ArrayLike
    ) -> tuple[int | float, int | float]:
        """
        Wavelength grid generation pipeline.
        Getting the minimum and maximum values of an untrusted array.

        Args:
        - requested_wavelengths: Array-like object containing wavelength values.

        Returns:
        - Tuple of (nm_min, nm_max) clamped to [0, nm_red_limit].
        """
        return get_extremal_grid_endpoints(requested_wavelengths, upper_limit=nm_red_limit)

    def _grid_endpoints_preprocessing(
        self,
        start: float,
        end: float
    ) -> tuple[int, int]:
        """
        Wavelength grid generation pipeline.
        Maps the endpoints to a standard grid (wavelengths are multiples of the grid step).

        Args:
        - start: Start wavelength value.
        - end: End wavelength value.

        Returns:
        - Tuple of (start, end) as integers after preprocessing.
        """
        return grid_endpoints_preprocessing(start, end, nm_step)

    def uniform_grid(
        self,
        start: float,
        end: float
    ) -> npt.NDArray[np.integer]:
        """
        Wavelength grid generation pipeline.
        Returns a uniform grid array with the points being multiples of the grid step (endpoints included).

        Args:
        - start: Start wavelength value.
        - end: End wavelength value.

        Returns:
        - Array of wavelengths as int values on a uniform grid.
        """
        return uniform_grid(start, end, nm_step, dtype=wavelength_nm_dtype)


class BaseObject(UniformSpectralGrid):
    """
    Internal class for inheriting spectral data properties.
    Provides common attributes and methods for all (photo)spectral objects.
    """
    wavelength_nm: npt.NDArray[np.integer] = NotImplemented  # the own spectral axis or the wavelength range of the filter set
    spectral_dist: npt.NDArray[np.floating] = NotImplemented
    uncertainty: Uncertainty | None = None
    name: object = None

    ndim: ClassVar[int] = NotImplemented

    wavelength_nm_dtype: Final[npt.DTypeLike] = wavelength_nm_dtype
    spectral_dist_dtype: Final[npt.DTypeLike] = spectral_dist_dtype
    uncertainty_dtype: Final[npt.DTypeLike] = uncertainty_dtype

    @property
    def spectral_size(self) -> int:
        """ Returns the spectral axis length. """
        # Alternative `self.wavelength_nm.size` is not used because `wavelength_nm` is not always implemented
        return cast(int, self.spectral_dist.shape[0])

    @property
    def spatial_size(self) -> int:
        """ Returns the total number of (photo)spectra stored in the object. """
        return prod(self.spatial_shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """ Returns the spatial axes shape: length of the set or (width, height). """
        return self.spectral_dist.shape[1:]

    @property
    def standard_deviation(self) -> npt.NDArray[np.floating] | None:
        """ Read-only standard deviations, or None if there is no uncertainty. """
        if self.uncertainty is None:
            return None
        return self.uncertainty.standard_deviation

    @property
    def covariance_matrix(self) -> npt.NDArray[np.floating] | None:
        """
        Read-only full covariance matrix, materialized on access.

        Mutating the returned array does not change the stored uncertainty;
        replace it through `uncertainty` instead.
        """
        if self.uncertainty is None:
            return None
        return self.uncertainty.covariance_matrix

    @classmethod
    def _from_trusted_arrays(
        cls,
        wavelength_nm: npt.NDArray[np.integer],
        spectral_dist: npt.NDArray[np.floating],
        uncertainty: Uncertainty | None,
        name: object = None
    ) -> Self:
        """ Internal fast constructor for already validated, owned arrays. """
        obj = object.__new__(cls)
        obj.wavelength_nm = wavelength_nm.copy()
        obj.spectral_dist = spectral_dist
        obj.uncertainty = uncertainty
        obj.name = name
        return obj

    @classmethod
    def stub(cls, name: object = None) -> Self:  # pyright: ignore[reportUnusedParameter]
        """
        Initializes a stub object in case of data problems.
        Implemented in the inherited classes.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def convert_from_photon_spectral_density(self) -> Self:
        """
        Returns a new BaseObject converted from photon spectral density
        to energy spectral density, using the fact that E = h c / λ.
        Implemented in the inherited classes.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def convert_from_energy_spectral_density_per_frequency(self) -> Self:
        """
        Returns a new BaseObject converted from frequency spectral density
        to energy spectral density, using the fact that f_λ = f_ν c / λ².
        Implemented in the inherited classes.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _apply_element_wise_operation(
        self,
        other: 'BaseObject',  # pyright: ignore[reportUnusedParameter]
        value_handling: ValueHandler,  # pyright: ignore[reportUnusedParameter]
        uncertainty_handling: UncertaintyHandler  # pyright: ignore[reportUnusedParameter]
    ) -> Self:
        """
        Returns a new object formed from element-wise operation.
        Implemented in the inherited classes.

        Args:
        - other: Another BaseObject for element-wise operations.
        - value_handling: Function to handle the value transformation.
        - uncertainty_handling: Function to handle the uncertainty propagation.

        Returns:
        - A new SpectralObject with the element-wise operation applied.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _apply_spectral_operation(
        self,
        other: 'BaseObject',
        value_handling: ValueHandler,
        uncertainty_handling: UncertaintyHandler
    ) -> Self:
        """
        Returns a new object formed from an operation defined on the spectral
        distribution rather than on its photometric integrals.
        By default, it is equivalent to the element-wise operation.

        Args:
        - other: Another BaseObject for the operation.
        - value_handling: Function to handle the value transformation.
        - uncertainty_handling: Function to handle the uncertainty propagation.

        Returns:
        - A new object with the spectral operation applied.
        """
        return self._apply_element_wise_operation(other, value_handling, uncertainty_handling)

    def _apply_scalar_operation(
        self,
        operand: npt.ArrayLike,
        value_handling: ValueHandler,
        uncertainty_handling: UncertaintyHandler
    ) -> Self:
        """
        Returns a new object of the same class transformed according to the operator.

        Args:
        - operand: A scalar or array-like value for the operation.
        - value_handling: Function to handle the value transformation.
        - uncertainty_handling: Function to handle the uncertainty propagation.

        Returns:
        - A new SpectralObject with the scalar operation applied.
        """
        # Perf: shallow copy avoids copying the full spectral array; the result
        # gets a new array below. cube * scalar (61 x 100 x 100): 3.68 -> 0.44 ms.
        output = copy(self)
        output.spectral_dist = value_handling(self.spectral_dist, operand)
        output.uncertainty = uncertainty_handling(self.spectral_dist, self.uncertainty, operand, None)
        return output

    def __add__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the addition operator.

        Returns:
        - A new SpectralObject with element-wise or scalar addition applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, add_value, add_uncertainty)
        else:
            return self._apply_scalar_operation(other, add_value, add_uncertainty)

    def __sub__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the subtraction operator.

        Returns:
        - A new SpectralObject with element-wise or scalar subtraction applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, sub_value, sub_uncertainty)
        else:
            return self._apply_scalar_operation(other, sub_value, sub_uncertainty)

    def __mul__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the multiplication operator.

        Returns:
        - A new object with spectral (element-wise) or scalar multiplication applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_spectral_operation(other, mul_value, mul_uncertainty)
        else:
            return self._apply_scalar_operation(other, mul_value, mul_uncertainty)

    def __truediv__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the division operator.

        Returns:
        - A new object with spectral (element-wise) or scalar division applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_spectral_operation(other, div_value, div_uncertainty)
        else:
            return self._apply_scalar_operation(other, div_value, div_uncertainty)

    @override
    def __eq__(self, other: object) -> bool:
        """
        Checks equality with another BaseObject instance.

        Returns:
        - True if both wavelength and spectral distribution arrays are equal.
        """
        if isinstance(other, BaseObject):
            return np.array_equal(self.wavelength_nm, other.wavelength_nm) and np.array_equal(self.spectral_dist, other.spectral_dist)
        return False

    def _generate_repr_config(self) -> dict[str, str]:
        """ Generates default configuration for string representation in __repr__() """
        # Name preparation
        if self.name is None:
            repr_config = {}
        else:
            if isinstance(self.name, str):
                name_str = f"'{self.name}'"
            else:
                name_str = str(self.name)
            repr_config = {
                'name': name_str,
            }
        size_str = f'{self.spectral_size} spectral'
        # Size preparation
        if len(self.spatial_shape) != 0:
            if len(self.spatial_shape) == 1:
                spatial_info = self.spatial_shape[0]
            else:
                spatial_info = str(self.spatial_shape).replace(', ', ' × ')
            size_str += f' × {spatial_info} spatial'
        size_str += ' data points'
        # Create configuration
        repr_config |= {
            'size': size_str,
        }
        if self.wavelength_nm is NotImplemented:
            nm_str = 'NotImplemented'
        else:
            nm_str = repr_generator(self.wavelength_nm)
        repr_config |= {
            'wavelength_nm': nm_str,
            'spectral_dist': repr_generator(self.spectral_dist)
        }
        if self.uncertainty is not None:
            if self.spectral_dist.ndim > 1:
                # Perf: a summary avoids materializing a full std/covariance
                # array just to print a few rows or a generic placeholder.
                # repr on 21 x 500 x 1000 cube with std: 98.4 -> 0.10 ms.
                repr_config |= {'uncertainty': repr(self.uncertainty)}
            elif self.uncertainty.is_covariance:
                repr_config |= {
                    'covariance_matrix': repr_generator(self.uncertainty.covariance_matrix)
                }
            else:
                repr_config |= {
                    'standard_deviation': repr_generator(self.uncertainty.standard_deviation)
                }
        return repr_config

    @override
    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        The string is formatted based on the `_generate_repr_config`, unique for different classes.
        """
        repr_config = self._generate_repr_config()
        output = f'{self.__class__.__name__}('
        for key, value in repr_config.items():
            output += f'\n\t{key} = {value.replace("\n", "\n\t")},'
        output = output[:-1] # removing the last comma
        output += '\n)'
        return output


# For type checkers, this structure needed to set a dimensional hierarchy between classes
LowerDimensionalType = TypeVar('LowerDimensionalType', bound='BaseObject')
HigherDimensionalType = TypeVar('HigherDimensionalType', bound='BaseObject')


class Item[HigherDimensionalType: BaseObject](BaseObject):
    """
    Internal class for inheriting spatial data properties (1D).
    Represents a single spectrum.
    """

    ndim: ClassVar[int] = 1


class Set[HigherDimensionalType: BaseObject, LowerDimensionalType: BaseObject](BaseObject):
    """
    Internal class for inheriting spatial data properties (2D).
    Represents a set of spectra.
    """

    ndim: ClassVar[int] = 2

    def __len__(self) -> int:
        """ Returns the spatial axis length (alias for .spatial_size). """
        return self.spatial_size

    def __iter__(self) -> Iterator[Self]:
        """ Creates an iterator over the elements in the set. """
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item: int | slice) -> Self:
        """ Returns the spatial axis element or slice. """
        # Perf: copy metadata instead of deep-copying the whole object.
        # Slicing 10 of 100 spectra from a 61 x 100 set: 0.35 -> 0.003 ms.
        output = copy(self)
        if isinstance(output.wavelength_nm, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
            output.wavelength_nm = output.wavelength_nm.copy()
        output.spectral_dist = output.spectral_dist[:,item]
        if output.uncertainty is not None:
            output.uncertainty = output.uncertainty.index_spatial(item)
        return output


class Cube[LowerDimensionalType: BaseObject](BaseObject):
    """
    Internal class for inheriting spatial data properties (3D).
    Represents a cube of spectra.
    """

    ndim: ClassVar[int] = 3

    def downscale(
        self,
        pixels_limit: int
    ) -> Self:
        """
        Brings the spatial resolution of the cube to approximately match the number of pixels.

        Args:
        - pixels_limit: Target maximum number of pixels in the output.
        """
        # Perf: copy only the decimated data, not the whole cube.
        # cube.downscale(1000) on 61 x 100 x 100: 0.39 -> 0.07 ms.
        output = copy(self)
        if isinstance(output.wavelength_nm, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
            output.wavelength_nm = output.wavelength_nm.copy()
        output.spectral_dist = spatial_downscaling(output.spectral_dist, None, pixels_limit)[0].copy()
        if output.uncertainty is not None:
            output.uncertainty = deepcopy(output.uncertainty.downscale_spatial(pixels_limit))
        return output

    def flatten(self) -> LowerDimensionalType:
        """
        Returns a (photo)spectral set with linearized spatial axis.
        Implemented in the inherited classes.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    @property
    def width(self) -> int:
        """ Returns horizontal spatial axis length. """
        return self.spatial_shape[0]

    @property
    def height(self) -> int:
        """ Returns vertical spatial axis length. """
        return self.spatial_shape[1]
