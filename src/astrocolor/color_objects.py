"""
Color containers and the display pipeline.

`ColorObject` stores linear color values together with their `ColorSystem` and
optional `Uncertainty`. The first axis always contains the three color
channels; the remaining spatial axes distinguish `ColorPoint`, `ColorLine` and
`ColorImage`. Objects are created from linear data, from a spectrum via
`from_spectral_data()`, or with `get_color()`.

`DisplaySettings` describes presentation rather than stored data: target color
system, transfer function, brightness normalization and scaling. `render()` and
`to_html()` never modify the object, so the same linear color can be displayed
in several ways. `to_color_system()` changes the stored color system.

`spectral_dist` is always linear. Pass `gamma_corrected=True` to the constructor
when the input values are transfer-encoded; they are decoded immediately.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Self, TypeVar, cast, overload, override

import numpy as np
import numpy.typing as npt

from .auxiliary import spatial_downscaling
from .color_standards import (
    ColorSystem,
    default_color_system,
    xyz_cmf,
    xyz_color_system,
)
from .config import Config
from .errors import InvalidColorChannelsError
from .measurements import get_photometry
from .photospectral_objects import (
    PhotospectralCube,
    PhotospectralObject,
    PhotospectralSet,
    Photospectrum,
)
from .spectral_objects import SpectralCube, SpectralObject, SpectralSet, Spectrum
from .uncertainty import Uncertainty


@dataclass(frozen=True)
class DisplaySettings:
    """
    Settings for converting linear color data into display-ready values.

    Attributes:
    - `color_system`: target color system. The default is the recommended
      sRGB color space with a chromatic adaptation to illuminant E white point.
    - `gamma_correction`: apply the transfer function of the target color
      space. Each color space uses its own curve (sRGB, gamma 2.2, ProPhoto,
      Rec. 709, etc.); linear spaces use the identity.
    - `maximize_brightness`: normalize the output to its brightest value.
    - `scale_factor`: multiply the output by a non-negative constant.
    """

    color_system: ColorSystem = default_color_system
    gamma_correction: bool = True
    maximize_brightness: bool = True
    scale_factor: float = 1.

    def __post_init__(self) -> None:
        factor = float(self.scale_factor)
        if not np.isfinite(factor) or factor < 0:
            raise ValueError('DisplaySettings.scale_factor must be a finite non-negative number.')
        object.__setattr__(self, 'scale_factor', factor)


# Recommended display configuration:
# - sRGB color space
# - chromatic adaptation to illuminant E white point
# - sRGB transfer function
# - full brightness
DEFAULT_SETTINGS = DisplaySettings()
LINEAR_SETTINGS = DisplaySettings(gamma_correction=False, maximize_brightness=False)


# For type checkers, this type specifies that each color class
# can only store hyperspectral objects of its own dimension.
DimensionallyCorrectType = TypeVar('DimensionallyCorrectType', bound='SpectralObject | PhotospectralObject')

class ColorObject[DimensionallyCorrectType: SpectralObject | PhotospectralObject]:
    """
    This class stores a linear color brightness array (`self.spectral_dist`),
    its `uncertainty`, the corresponding color system, and provides conversion
    and display methods.

    The brightness array is required to be of length 3 along the first axis.
    The indices correspond to the three channels of `color_system` (RGB, XYZ,
    or another supported system).

    `spectral_dist` is always linear, including when the constructor receives
    gamma-corrected values. Use `render()` or `to_html()` for display output.
    `color_system` is a frozen attribute, change it with `to_color_system()`.
    """

    def __init__(
        self,
        spectral_dist: npt.ArrayLike,
        uncertainty: npt.ArrayLike | Uncertainty | None = None,
        color_system: ColorSystem = xyz_color_system,
        *,
        gamma_corrected: bool = False,
    ) -> None:
        """
        ColorObject requires a brightness array and corresponding color system.
        Default color space is CIE 1931 XYZ. Uncertainty is optional.

        `spectral_dist` is always stored linear. Set `gamma_corrected=True`
        when the input is display-encoded in `color_system`; it is decoded
        immediately. Covariance propagation through that nonlinear step is a
        separate, not yet implemented operation.

        Raises:
        - `InvalidColorChannelsError`: if the first axis does not contain
          exactly three color channels.
        """
        values = np.asarray(spectral_dist, dtype=np.float64)
        if values.ndim == 0 or values.shape[0] != 3:
            raise InvalidColorChannelsError(values.shape)
        uncertainty_obj = (
            None if uncertainty is None
            else Uncertainty.from_input(uncertainty, values.shape)
        )
        # `spectral_dist` is stored linear, so encoded input is decoded here.
        if gamma_corrected:
            if uncertainty_obj is None:
                values = color_system.decode(values)
            else:
                covariance = uncertainty_obj.covariance_matrix
                decoded_covariance = color_system.decode_covariance(values, covariance)
                values = color_system.decode(values)
                uncertainty_obj = Uncertainty.from_covariance(
                    decoded_covariance,
                    values.shape,
                )
        self.spectral_dist: npt.NDArray[np.floating] = values
        self.uncertainty: Uncertainty | None = uncertainty_obj
        self._color_system: ColorSystem = color_system

    @property
    def covariance_matrix(self) -> npt.NDArray[np.floating] | None:
        """ Returns the full covariance matrix, materialized on access. """
        if self.uncertainty is None:
            return None
        return self.uncertainty.covariance_matrix

    @property
    def standard_deviation(self) -> npt.NDArray[np.floating] | None:
        """ Returns standard deviations, or None if there is no uncertainty. """
        if self.uncertainty is None:
            return None
        return self.uncertainty.standard_deviation

    @classmethod
    def from_spectral_data(cls, data: DimensionallyCorrectType) -> Self:
        """ Convolves a (photo)spectral object with the CIE 1931 XYZ color matching functions """
        photometry = get_photometry(data, xyz_cmf)
        return cls(photometry.spectral_dist, photometry.uncertainty, xyz_color_system)

    def to_color_system(self, new_color_system: ColorSystem) -> Self:
        """
        Return a new ColorObject with changed color system.

        `spectral_dist` is linear, so only the RGB <-> XYZ matrices are applied;
        transfer functions are not involved. If the target system defines the
        same transform as the current one, no matrix conversion is performed.

        Attention! For saturated colors, color system conversion is not always reversible!
        """
        source_system = self._color_system
        output = deepcopy(self)
        output._color_system = new_color_system
        if new_color_system == source_system:
            return output

        xyz_values, xyz_covariance = source_system.rgb_to_xyz(
            self.spectral_dist, self.covariance_matrix
        )
        output.spectral_dist, rgb_covariance = new_color_system.xyz_to_rgb(
            xyz_values, xyz_covariance
        )
        output.uncertainty = (
            None if rgb_covariance is None
            else Uncertainty.from_covariance(rgb_covariance, output.spectral_dist.shape)
        )
        minimum = cast(npt.NDArray[np.floating], np.min(output.spectral_dist, axis=0))
        if np.any(minimum < 0):
            # We're not in the color system gamut: approximate by desaturating
            # 1D implementation: rgb -= np.min(rgb)
            output.spectral_dist -= np.minimum(minimum, 0.)
        return output

    def render(
        self,
        settings: DisplaySettings = DEFAULT_SETTINGS,
    ) -> npt.NDArray[np.floating]:
        """
        Returns a display-ready array: converts the color to `settings.color_system`,
        optionally normalizes brightness, applies `scale_factor` and the target
        color space transfer function.
        The object itself is not modified.
        """
        color = self
        if self.color_system != settings.color_system:
            color = self.to_color_system(settings.color_system)
        arr = np.nan_to_num(color.spectral_dist, copy=True)
        maximum = cast(float, arr.max())
        if settings.maximize_brightness and maximum != 0:
            arr /= maximum
        if settings.scale_factor != 1:
            arr *= settings.scale_factor
        if settings.gamma_correction:
            arr = settings.color_system.encode(arr)
        return arr

    def grayscale(self) -> npt.NDArray[np.floating] | float:
        """ Converts color to linear grayscale using CIE 1931 luminance (Y in XYZ color space) """
        sd = self.to_color_system(xyz_color_system).spectral_dist
        return cast(npt.NDArray[np.floating] | float, sd[1])

    @property
    def color_system(self) -> ColorSystem:
        """
        Color system cannot be set directly for safety reasons.
        Use to_color_system() to perform data conversion together with the color system.
        """
        return self._color_system


class ColorPoint(ColorObject[Spectrum | Photospectrum]):
    """
    Class to work with a point of three color channels.
    Stores linear brightness values in the `spectral_dist` attribute, numpy array of shape (3).
    Use `render()` or `to_html()` for display-ready values.
    """

    def to_bit(
        self,
        bit: int,
        clip: bool = False,
        settings: DisplaySettings = DEFAULT_SETTINGS,
    ) -> npt.NDArray[np.floating]:
        """ Returns display-ready color array, scaled to the appropriate power of two (not rounded) """
        factor = cast(int, 2**bit) - 1
        arr = self.render(settings)
        if clip:
            arr = np.clip(arr, 0, 1)
        return arr * factor

    def to_html(self, settings: DisplaySettings = DEFAULT_SETTINGS) -> str:
        """ Converts linear color values to a display-ready HTML hexadecimal string """
        return '#{:02x}{:02x}{:02x}'.format(*self.to_bit(8, clip=True, settings=settings).round().astype('int'))  # pyright: ignore[reportAny]


class ColorLine(ColorObject[SpectralSet | PhotospectralSet]):
    """
    Class to work with a line of three color channels.
    Stores linear brightness values in the `spectral_dist` attribute, numpy array of shape (3, X).
    Use `render()` for display-ready values.
    """

    def __init__(self, *args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownArgumentType]
        self.spectral_dist: npt.NDArray[np.floating] = np.atleast_2d(self.spectral_dist) # interprets color points as lines
        if self.uncertainty is not None:
            self.uncertainty: Uncertainty | None = self.uncertainty.reshape_spatial(self.spectral_dist.shape[1:])

    @property
    def size(self) -> int:
        """ Returns spatial axis length """
        return cast(int, self.spectral_dist.shape[1])


class ColorImage(ColorObject[SpectralCube | PhotospectralCube]):
    """
    Class to work with an image of three color channels.
    Stores linear brightness values in the `spectral_dist` attribute, numpy array of shape (3, X, Y).
    Use `render()` for display-ready values.
    """

    def __init__(self, *args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownArgumentType]
        self.spectral_dist: npt.NDArray[np.floating] = np.atleast_3d(self.spectral_dist) # interprets color points and lines as images
        if self.uncertainty is not None:
            self.uncertainty: Uncertainty | None = self.uncertainty.reshape_spatial(self.spectral_dist.shape[1:])

    @override
    @classmethod
    def from_spectral_data(cls, data: SpectralCube | PhotospectralCube) -> Self:
        """ Convolves a (photo)spectral cube with the CIE 1931 XYZ color matching functions """
        px_num = data.spatial_size
        px_limit = Config.pixel_upper_limit
        if px_num <= px_limit:
            # The image size does not exceed the size of the batch
            return super().from_spectral_data(data)
        # Processing by dividing into batches
        message = f'Processing {px_num} pixels'
        if data.name is not None:
            message += f' of "{data.name}"'
        print(message + '...')
        start_time = monotonic()
        flatten_cube = data.flatten()
        flatten_img = np.empty((3, px_num))
        flatten_cov: npt.NDArray[np.floating] | None = None
        batches = range(0, px_num, px_limit)
        batch_num = len(batches)
        for batch_index, x0 in enumerate(batches, start=1):
            x1 = min(x0 + px_limit, px_num)
            batch_color = ColorLine.from_spectral_data(flatten_cube[x0:x1])
            flatten_img[:, x0:x1] = batch_color.spectral_dist
            covariance = batch_color.covariance_matrix
            if covariance is not None:
                if flatten_cov is None:
                    flatten_cov = np.zeros((3, 3, px_num))
                flatten_cov[:, :, x0:x1] = covariance
            print(f'Progress: {batch_index} / {batch_num}')
        # End of processing, summarizing
        time = monotonic() - start_time
        speed = px_num / time
        print(f'Processing took {time:.1f} seconds ({speed:.1f} px/sec)')
        covariance = None if flatten_cov is None else flatten_cov.reshape(3, 3, data.width, data.height)
        return cls(
            flatten_img.reshape(3, data.width, data.height),
            covariance,
            color_system=xyz_color_system,
        )

    def upscale(
        self,
        times: int
    ) -> 'ColorImage':
        """ Creates a new ColorImage with increased size by an integer number of times """
        output = deepcopy(self)
        output.spectral_dist = np.repeat(
            np.repeat(output.spectral_dist, times, axis=1), times, axis=2
        )
        if output.uncertainty is not None:
            output.uncertainty = output.uncertainty.repeat_spatial(times)
        return output

    def downscale(
        self,
        pixels_limit: int
    ) -> 'ColorImage':
        """ Brings the resolution of the image to approximately match the number of pixels """
        output = deepcopy(self)
        output.spectral_dist = spatial_downscaling(output.spectral_dist, None, pixels_limit)[0]
        if output.uncertainty is not None:
            output.uncertainty = output.uncertainty.downscale_spatial(pixels_limit)
        return output

    @property
    def width(self) -> int:
        """ Returns horizontal spatial axis length """
        return cast(int, self.spectral_dist.shape[1])

    @property
    def height(self) -> int:
        """ Returns vertical spatial axis length """
        return cast(int, self.spectral_dist.shape[2])

    @property
    def size(self) -> int:
        """ Returns the number of pixels """
        return self.width * self.height


@overload
def get_color(
    data: Spectrum | Photospectrum,
    color_system: ColorSystem = default_color_system,
) -> ColorPoint:
    ...

@overload
def get_color(
    data: SpectralSet | PhotospectralSet,
    color_system: ColorSystem = default_color_system,
) -> ColorLine:
    ...

@overload
def get_color(
    data: SpectralCube | PhotospectralCube,
    color_system: ColorSystem = default_color_system,
) -> ColorImage:
    ...

@overload
def get_color(
    data: SpectralObject | PhotospectralObject,
    color_system: ColorSystem = default_color_system,
) -> ColorPoint | ColorLine | ColorImage:
    ...

def get_color(
    data: SpectralObject | PhotospectralObject,
    color_system: ColorSystem = default_color_system,
) -> ColorPoint | ColorLine | ColorImage:
    """
    Converts (photo)spectral data into a linear color object in the requested
    color system. The default is the recommended sRGB with Illuminant E.

    Use `render()` or `to_html()` on the result for display-ready values.

    Example:
    ```
    >>> color = get_color(sun_tsis1_hsrs)
    >>> color.to_html()
    '#ecfeff'
    ```
    """
    color: ColorPoint | ColorLine | ColorImage
    match data.ndim:
        case 1:
            color = ColorPoint.from_spectral_data(cast(Spectrum | Photospectrum, data))
        case 2:
            color = ColorLine.from_spectral_data(cast(SpectralSet | PhotospectralSet, data))
        case 3:
            color = ColorImage.from_spectral_data(cast(SpectralCube | PhotospectralCube, data))
        case _:
            raise ValueError(f'Unsupported color data dimensionality: {data.ndim}')
    if color.color_system is color_system:
        return color
    return color.to_color_system(color_system)
