import numpy as np
import numpy.typing as npt
from math import prod
from collections.abc import Callable
from typing import cast, Self, Any, Final, ClassVar, TypeAlias, Iterator
from copy import deepcopy

from .auxiliary import (
    get_extremal_grid_endpoints,
    grid_endpoints_preprocessing,
    uniform_grid,
    repr_generator,
    spatial_downscaling
)
from .algebra import (
    add_value,
    add_error,
    sub_value,
    sub_error,
    mul_value,
    mul_error,
    div_value,
    div_error
)
from .errors import InconsistentAxesError, InconsistentUncertaintySizeError


# For the sake of simplifying work with the spectrum,
# its discretization step is fixed and frozen.
nm_step: Final[int] = 5  # nm

# Wavelength and brightness axis storage data type
wavelength_nm_dtype: Final[npt.DTypeLike] = np.int32
spectral_dist_dtype: Final[npt.DTypeLike] = np.float64

# Maximum wavelength, the clipping level
# It is important that there be no overflow when raising a number to the second power
# See convert_from_energy_spectral_density_per_frequency()
nm_red_limit: Final[int] = int(np.sqrt(np.iinfo(wavelength_nm_dtype).max)) # 46340 nm


class BaseObject:
    """
    Internal class for inheriting spectral data properties.
    Provides common attributes and methods for all (photo)spectral objects.
    """
    wavelength_nm: npt.NDArray = NotImplemented  # the own spectral axis or the wavelength range of the filter set
    spectral_dist: npt.NDArray = NotImplemented
    covariance_matrix: npt.NDArray | None = None
    name: Any = None

    ndim: ClassVar[int] = NotImplemented

    # Storing important class properties within the class
    nm_step: Final[int] = nm_step
    wavelength_nm_dtype: Final[npt.DTypeLike] = wavelength_nm_dtype
    spectral_dist_dtype: Final[npt.DTypeLike] = spectral_dist_dtype
    nm_red_limit: Final[int] = nm_red_limit

    # When processing images through spectral cubes, performance is prioritized,
    # and uncertainty is not saved (yet). Therefore it is disabled by default.
    ignore_uncertainty_forCubes: bool = True

    @property
    def spectral_size(self) -> int:
        """ Returns the spectral axis length. """
        return self.spectral_dist.shape[0]

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
    def stub(cls, name: Any = None) -> Self:
        """
        Initializes a stub object in case of data problems.
        Implemented in the inherited classes.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _get_extremal_grid_endpoints(
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
        return get_extremal_grid_endpoints(requested_wavelengths, nm_red_limit)

    def _grid_endpoints_preprocessing(
        self,
        start: int | float,
        end: int | float
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

    def _uniform_grid(
        self,
        start: int | float,
        end: int | float
    ) -> npt.NDArray:
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

    def determine_at_wavelengths(
        self,
        requested_wavelengths: npt.ArrayLike,
        strictly: bool = False
    ):
        """
        Returns a new SpectralObject, guaranteeing that the specified wavelength range
        has been determined or reconstructed for it.
        If `strictly=True`, then the new object is defined exclusively
        on the specified wavelength range.
        Only the minimum and maximum wavelengths are extracted from the specified range,
        based on which a uniform grid is constructed.

        Args:
        - requested_wavelengths: Wavelength values to determine at.
        - strictly: If True, clip the result to the exact requested range.

        Returns:
        - A new SpectralObject with data determined at the specified wavelengths.

        Example:
        ```
        >>> spectrum = photospectrum.determine_at_wavelengths([400, 700])
        ```
        """
        nm_min, nm_max = self._get_extremal_grid_endpoints(requested_wavelengths)
        requested_wavelengths = self._uniform_grid(nm_min, nm_max)
        from.spectral_objects import SpectralObject
        spectral_obj = cast(SpectralObject, self._determine_at_trusted_wavelengths(requested_wavelengths))
        # Spectral range clipping
        if strictly and not np.array_equal(spectral_obj.wavelength_nm, requested_wavelengths):
            spectral_obj.spectral_dist = spectral_obj.get_spectral_dist_at_wavelengths(nm_min, nm_max)
            spectral_obj.covariance_matrix = spectral_obj.get_covariance_matrix_at_wavelengths(nm_min, nm_max)
            spectral_obj.wavelength_nm = requested_wavelengths
        # Sanity checks
        if (len_nm := spectral_obj.wavelength_nm.size) != (len_values := len(spectral_obj.spectral_dist)):
            raise InconsistentAxesError(len_nm, len_values, spectral_obj.name)
        if spectral_obj.covariance_matrix is not None and (len_error := len(spectral_obj.covariance_matrix)) != len_nm:
            raise InconsistentUncertaintySizeError(len_error, len_values, spectral_obj.name)
        return spectral_obj

    def _determine_at_trusted_wavelengths(
        self,
        requested_wavelengths: npt.NDArray
    ):
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        Implemented in the inherited classes.

        Args:
        - requested_wavelengths: The trusted wavelength array to use.

        Returns:
        - A new SpectralObject with data determined at the trusted wavelengths.
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
        other: 'BaseObject',
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
        error_handling: Callable[[npt.ArrayLike, npt.NDArray | None, npt.ArrayLike, npt.NDArray | None], npt.NDArray | None]
    ) -> Self:
        """
        Returns a new object formed from element-wise operation.
        Implemented in the inherited classes.

        Args:
        - operand: Another BaseObject for element-wise operations.
        - value_handling: Function to handle the value transformation.
        - error_handling: Function to handle the uncertainty propagation.

        Returns:
        - A new SpectralObject with the element-wise operation applied.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _apply_scalar_operation(
        self,
        operand: npt.ArrayLike,
        value_handling: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
        error_handling: Callable[[npt.ArrayLike, npt.NDArray | None, npt.ArrayLike, None], npt.NDArray | None]
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

    def __add__(self, other) -> Self:
        """
        Implements the addition operator.

        Returns:
        - A new SpectralObject with element-wise or scalar addition applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, add_value, add_error)
        else:
            return self._apply_scalar_operation(other, add_value, add_error)

    def __sub__(self, other) -> Self:
        """
        Implements the subtraction operator.

        Returns:
        - A new SpectralObject with element-wise or scalar subtraction applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, sub_value, sub_error)
        else:
            return self._apply_scalar_operation(other, sub_value, sub_error)

    def __mul__(self, other) -> Self:
        """
        Implements the multiplication operator.

        Returns:
        - A new SpectralObject with element-wise or scalar multiplication applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, mul_value, mul_error)
        else:
            return self._apply_scalar_operation(other, mul_value, mul_error)

    def __truediv__(self, other) -> Self:
        """
        Implements the division operator.

        Returns:
        - A new SpectralObject with element-wise or scalar division applied.
        """
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, div_value, div_error)
        else:
            return self._apply_scalar_operation(other, div_value, div_error)

    def __hash__(self) -> int:
        """
        Returns the hash value based on the object's name.

        Raises:
        - TypeError: If the object has no name (name is None).
        """
        if self.name is None:
            raise TypeError('Unhashable type: "NoneType"')
        return hash(self.name)

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
        if not isinstance(item, int | slice):
            raise TypeError(f'Index must be int or slice, not {type(item).__name__}')
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


RealObject: TypeAlias = Item | Set | Cube
"""
Type alias for any real (photo)spectral object.
Can be an Item (1D), Set (2D), or Cube (3D).
"""
