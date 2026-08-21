from collections.abc import Callable, Iterator
from copy import deepcopy
from math import prod
from typing import ClassVar, Final, Self, cast, override

import numpy as np
import numpy.typing as npt

from .algebra import (
    add_error,
    add_value,
    div_error,
    div_value,
    mul_error,
    mul_value,
    sub_error,
    sub_value,
)
from .auxiliary import (
    get_extremal_grid_endpoints,
    grid_endpoints_preprocessing,
    repr_generator,
    spatial_downscaling,
    uniform_grid,
)

# For the sake of simplifying work with the spectrum,
# its discretization step is fixed and frozen.
nm_step: Final[int] = 5  # nm

# Wavelength and brightness axis storage data type
wavelength_nm_dtype: Final[npt.DTypeLike] = np.int32
spectral_dist_dtype: Final[npt.DTypeLike] = np.float64

# Maximum wavelength, the clipping level
# It is important that there be no overflow when raising a number to the second power
# See convert_from_energy_spectral_density_per_frequency()
nm_red_limit: Final[int] = int(cast(float, np.sqrt(np.iinfo(wavelength_nm_dtype).max))) # 46340 nm


class BaseObject:
    """
    Internal class for inheriting spectral data properties.
    Provides common attributes and methods for all (photo)spectral objects.
    """
    wavelength_nm: npt.NDArray[np.integer] = NotImplemented  # the own spectral axis or the wavelength range of the filter set
    spectral_dist: npt.NDArray[np.floating] = NotImplemented
    covariance_matrix: npt.NDArray[np.floating] | None = None
    name: object = None

    ndim: ClassVar[int] = NotImplemented

    # Storing important class properties within the class
    nm_step: Final[int] = nm_step
    wavelength_nm_dtype: Final[npt.DTypeLike] = wavelength_nm_dtype
    spectral_dist_dtype: Final[npt.DTypeLike] = spectral_dist_dtype
    nm_red_limit: Final[int] = nm_red_limit

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
        """
        Calculates an array of standard deviations from the covariance matrix.

        Returns:
            Array of standard deviations, or None if no covariance matrix exists.
        """
        if self.covariance_matrix is None:
            return None
        else:
            # TODO: support for sets and cubes
            return np.sqrt(np.diag(self.covariance_matrix))

    @classmethod
    def stub(cls, name: object = None) -> Self:  # pyright: ignore[reportUnusedParameter]
        """
        Initializes a stub object in case of data problems.
        Implemented in the inherited classes.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

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
        if (shift := start % self.nm_step) != 0:
            start += self.nm_step - shift
        if end % self.nm_step == 0:
            end += self.nm_step # to include the last point
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
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating]],  # pyright: ignore[reportUnusedParameter]
        error_handling: Callable[[npt.ArrayLike, npt.ArrayLike | None, npt.ArrayLike, npt.ArrayLike | None], npt.NDArray[np.floating] | None]  # pyright: ignore[reportUnusedParameter]
    ) -> Self:
        """
        Returns a new object formed from element-wise operation.
        Implemented in the inherited classes.

        Args:
        - other: Another BaseObject for element-wise operations.
        - value_handling: Function to handle the value transformation.
        - error_handling: Function to handle the uncertainty propagation.

        Returns:
        - A new SpectralObject with the element-wise operation applied.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _apply_scalar_operation(
        self,
        operand: npt.ArrayLike,
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray[np.floating]],
        error_handling: Callable[[npt.ArrayLike, npt.ArrayLike | None, npt.ArrayLike, npt.ArrayLike | None], npt.NDArray[np.floating] | None]
    ) -> Self:
        """
        Returns a new object of the same class transformed according to the operator.

        Args:
        - operand: A scalar or array-like value for the operation.
        - value_handling: Function to handle the value transformation.
        - error_handling: Function to handle the uncertainty propagation.

        Returns:
            A new SpectralObject with the scalar operation applied.
        """
        output = deepcopy(self)
        output.spectral_dist = value_handling(self.spectral_dist, operand)
        output.covariance_matrix = error_handling(self.spectral_dist, self.covariance_matrix, operand, None)
        return output

    def __add__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the addition operator.

        Returns:
        - A new SpectralObject with element-wise or scalar addition applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, add_value, add_error)
        else:
            return self._apply_scalar_operation(other, add_value, add_error)

    def __sub__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the subtraction operator.

        Returns:
        - A new SpectralObject with element-wise or scalar subtraction applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, sub_value, sub_error)
        else:
            return self._apply_scalar_operation(other, sub_value, sub_error)

    def __mul__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the multiplication operator.

        Returns:
        - A new SpectralObject with element-wise or scalar multiplication applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, mul_value, mul_error)
        else:
            return self._apply_scalar_operation(other, mul_value, mul_error)

    def __truediv__(self, other: 'BaseObject | npt.ArrayLike') -> Self:
        """
        Implements the division operator.

        Returns:
        - A new SpectralObject with element-wise or scalar division applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, div_value, div_error)
        else:
            return self._apply_scalar_operation(other, div_value, div_error)

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
            'wavelength_nm': repr_generator(self.wavelength_nm),
            'spectral_dist': repr_generator(self.spectral_dist),
        }
        if self.covariance_matrix is not None:
            repr_config |= {
                'covariance_matrix': repr_generator(self.covariance_matrix)
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


class Item(BaseObject):
    """
    Internal class for inheriting spatial data properties (1D).
    Represents a single spectrum.
    """

    ndim: ClassVar[int] = 1


class Set(BaseObject):
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
        output = deepcopy(self)
        output.spectral_dist = output.spectral_dist[:,item]
        if output.covariance_matrix is not None:
            output.covariance_matrix = output.covariance_matrix[:,:,item]
        return output


class Cube(BaseObject):
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
        output = deepcopy(self)
        output.spectral_dist, output.covariance_matrix = \
            spatial_downscaling(output.spectral_dist, output.covariance_matrix, pixels_limit)
        return output

    def flatten(self) -> 'Set':
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
