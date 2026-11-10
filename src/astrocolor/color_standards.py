from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, cast, override

import numpy as np
import numpy.typing as npt

from .filter_objects import FilterSet
from .measurements import get_photometry
from .reference_spectra import sun_tsis1_hsrs, vega_calspec
from .spectral_objects import Spectrum
from .uncertainty import propagate_covariance

# CIE 1931 XYZ color matching functions, 2-deg
# https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
# http://www.cvrl.org/cie.htm
xyz_cmf = FilterSet.get('CIE/1931_2deg.x', 'CIE/1931_2deg.y', 'CIE/1931_2deg.z')
visible_range = xyz_cmf.wavelength_nm # original CMF definition range is 360-830 nm

# There are CMFs transformed from the CIE (2006) LMS functions, 2-deg
# (https://cie.co.at/datatable/cie-2006-lms-cone-fundamentals-2-field-size-terms-energy)
# here: http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm, http://www.cvrl.org/ciexyzpr.htm
# However, the CIE 1931 XYZ standard is still widely used.

# Bradford chromatic adaptation matrices
# http://www.brucelindbloom.com/Eqn_ChromAdapt.html
matrix_B = np.array((
    ( 0.8951000,  0.2664000, -0.1614000),
    (-0.7502000,  1.7135000,  0.0367000),
    ( 0.0389000, -0.0685000,  1.0296000),
))
matrix_B_inv = np.array((
    ( 0.9869929, -0.1470543,  0.1599627),
    ( 0.4323053,  0.5183603,  0.0492912),
    (-0.0085287,  0.0400428,  0.9684867),
))


class TransferFunction(ABC):
    """
    Maps linear color values to nonlinear display values and back.

    Transfer functions are color-space specific: sRGB uses a piecewise curve,
    Adobe RGB uses gamma 2.2, ProPhoto RGB uses gamma 1.8 with a linear toe,
    and so on. Linear color spaces use the identity transfer function.
    """

    @abstractmethod
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """ Converts linear values to display-encoded values. """
        ...

    @abstractmethod
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """ Converts display-encoded values back to linear values. """
        ...

    @override
    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


class LinearTransferFunction(TransferFunction):
    """ Identity transfer function for linear color spaces. """

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return np.asarray(values, dtype=np.float64)

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return np.asarray(values, dtype=np.float64)


class SRGBTransferFunction(TransferFunction):
    """ The piecewise sRGB transfer function, also used by Display P3. """

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = arr * 12.92
        nonlinear = 1.055 * np.power(np.maximum(arr, 0.), 1./2.4) - 0.055
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 0.0031308, linear, nonlinear),
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = arr / 12.92
        nonlinear = np.power(np.maximum((arr + 0.055) / 1.055, 0.), 2.4)
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 0.04045, linear, nonlinear),
        )


class GammaTransferFunction(TransferFunction):
    """ A simple power-law transfer function with the specified gamma. """

    def __init__(self, gamma: float) -> None:
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('Gamma must be a finite positive number.')
        self.gamma: float = float(gamma)

    @override
    def __repr__(self) -> str:
        return f'GammaTransferFunction(gamma={self.gamma})'

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        return cast(
            npt.NDArray[np.floating],
            np.sign(arr) * np.power(np.abs(arr), 1./self.gamma),
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        return cast(
            npt.NDArray[np.floating],
            np.sign(arr) * np.power(np.abs(arr), self.gamma),
        )


class ProPhotoTransferFunction(TransferFunction):
    """ The ProPhoto RGB transfer function: gamma 1.8 with a linear toe. """

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        nonlinear = np.power(np.maximum(arr, 0.), 1./1.8)
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 1./512., arr * 16., nonlinear),
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        nonlinear = np.power(np.maximum(arr, 0.), 1.8)
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 1./32., arr / 16., nonlinear),
        )


class Rec709TransferFunction(TransferFunction):
    """ The Rec. 709 (HDTV) transfer function. """

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = 4.5 * arr
        nonlinear = 1.099 * np.power(np.maximum(arr, 0.), 0.45) - 0.099
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 0.018, linear, nonlinear),
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = arr / 4.5
        nonlinear = np.power(np.maximum((arr + 0.099) / 1.099, 0.), 1./0.45)
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 0.081, linear, nonlinear),
        )


class Rec2020TransferFunction(TransferFunction):
    """ The Rec. 2020 (UHDTV) transfer function. """

    alpha: ClassVar[float] = 1.09929682680944
    beta: ClassVar[float] = 0.018053968510807

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = 4.5 * arr
        nonlinear = self.alpha * np.power(np.maximum(arr, 0.), 0.45) - (self.alpha - 1.)
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < self.beta, linear, nonlinear),
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = arr / 4.5
        nonlinear = np.power(
            np.maximum((arr + (self.alpha - 1.)) / self.alpha, 0.),
            1./0.45,
        )
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < 4.5 * self.beta, linear, nonlinear),
        )


linear_transfer_function = LinearTransferFunction()
srgb_transfer_function = SRGBTransferFunction()
gamma_22_transfer_function = GammaTransferFunction(2.2)
prophoto_transfer_function = ProPhotoTransferFunction()
rec709_transfer_function = Rec709TransferFunction()
rec2020_transfer_function = Rec2020TransferFunction()


@dataclass(frozen=True)
class Chromaticity:
    """ CIE xy chromaticity coordinates. """

    x: float
    y: float

    @property
    def z(self) -> float:
        """ The implicit z = 1 - x - y coordinate. """
        return 1. - self.x - self.y

    def normalized_xyz(self) -> npt.NDArray[np.floating]:
        """ Returns (x, y, z) scaled so that Y = 1. """
        return np.array((self.x, self.y, self.z)) / self.y


@dataclass(frozen=True)
class WhitePoint:
    """ A named white point with its CIE xy chromaticity. """

    name: str
    chromaticity: Chromaticity

    @property
    def x(self) -> float:
        return self.chromaticity.x

    @property
    def y(self) -> float:
        return self.chromaticity.y

    def normalized_xyz(self) -> npt.NDArray[np.floating]:
        """ Returns the white point as XYZ scaled so that Y = 1. """
        return self.chromaticity.normalized_xyz()

    def __post_init__(self) -> None:
        if self.chromaticity.y <= 0:
            raise ValueError(f'White point "{self.name}" must have a positive y coordinate.')


@dataclass(frozen=True)
class ColorSpace:
    """
    Static description of an RGB color space.

    `primaries` are ordered red, green, blue. `white_point` is the native
    white point of the space, and `transfer_function` maps its linear values
    to display-encoded values and back.
    """

    name: str
    primaries: tuple[Chromaticity, Chromaticity, Chromaticity]
    white_point: WhitePoint
    transfer_function: TransferFunction

    def __post_init__(self) -> None:
        if len(self.primaries) != 3:
            raise ValueError(f'Color space "{self.name}" must have exactly three primaries.')


def chromaticity_of_spectrum(spectrum: Spectrum) -> Chromaticity:
    """ Returns the CIE xy chromaticity of a spectrum. """
    xyz = np.asarray(
        get_photometry(spectrum, xyz_cmf).spectral_dist,
        dtype=np.float64,
    )
    total = cast(float, xyz.sum())
    return Chromaticity(
        cast(float, xyz[0]) / total,
        cast(float, xyz[1]) / total,
    )


# White points are defined once and reused by color spaces and chromatic adaptation.
# https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
white_point_A = WhitePoint('Illuminant A', Chromaticity(0.44758, 0.40745))
white_point_B = WhitePoint('Illuminant B', Chromaticity(0.34842, 0.35161))
white_point_C = WhitePoint('Illuminant C', Chromaticity(0.31006, 0.31616))
white_point_D50 = WhitePoint('Illuminant D50', Chromaticity(0.34567, 0.35850))
white_point_D55 = WhitePoint('Illuminant D55', Chromaticity(0.33242, 0.34743))
white_point_D65 = WhitePoint('Illuminant D65', Chromaticity(0.31272, 0.32903))
white_point_D75 = WhitePoint('Illuminant D75', Chromaticity(0.29902, 0.31485))
white_point_D93 = WhitePoint('Illuminant D93', Chromaticity(0.28315, 0.29711))
white_point_E = WhitePoint('Illuminant E', Chromaticity(1./3., 1./3.))
white_point_Vega = WhitePoint('Vega', chromaticity_of_spectrum(vega_calspec))
white_point_Sun = WhitePoint('Sun', chromaticity_of_spectrum(sun_tsis1_hsrs))

WHITE_POINTS = (
    white_point_A,
    white_point_B,
    white_point_C,
    white_point_D50,
    white_point_D55,
    white_point_D65,
    white_point_D75,
    white_point_D93,
    white_point_E,
    white_point_Vega,
    white_point_Sun,
)
WHITE_POINTS_BY_NAME = {white_point.name: white_point for white_point in WHITE_POINTS}


# RGB primaries and white points.
# https://en.wikipedia.org/wiki/RGB_color_spaces
# http://brucelindbloom.com/index.html?WorkingSpaceInfo.html
COLOR_SPACES = (
    ColorSpace(
        name='CIE 1931 XYZ',
        primaries=(
            Chromaticity(1., 0.),
            Chromaticity(0., 1.),
            Chromaticity(0., 0.),
        ),
        white_point=white_point_E,
        transfer_function=linear_transfer_function,
    ),
    ColorSpace(
        name='CIE 1931 RGB',
        primaries=(
            Chromaticity(0.73474284, 0.26525716),
            Chromaticity(0.27377903, 0.7174777),
            Chromaticity(0.16655563, 0.00891073),
        ),
        white_point=white_point_E,
        transfer_function=linear_transfer_function,
    ),
    ColorSpace(
        name='sRGB',
        primaries=(
            Chromaticity(0.64, 0.33),
            Chromaticity(0.30, 0.60),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=srgb_transfer_function,
    ),
    ColorSpace(
        name='Display P3',
        primaries=(
            Chromaticity(0.68, 0.32),
            Chromaticity(0.265, 0.69),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=srgb_transfer_function,
    ),
    ColorSpace(
        name='Adobe RGB',
        primaries=(
            Chromaticity(0.64, 0.33),
            Chromaticity(0.21, 0.71),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=gamma_22_transfer_function,
    ),
    ColorSpace(
        name='Wide Gamut RGB',
        primaries=(
            Chromaticity(0.7347, 0.2653),
            Chromaticity(0.1152, 0.8264),
            Chromaticity(0.1566, 0.0177),
        ),
        white_point=white_point_D50,
        transfer_function=gamma_22_transfer_function,
    ),
    ColorSpace(
        name='ProPhoto RGB',
        primaries=(
            Chromaticity(0.734699, 0.265301),
            Chromaticity(0.159597, 0.840403),
            Chromaticity(0.036598, 0.000105),
        ),
        white_point=white_point_D50,
        transfer_function=prophoto_transfer_function,
    ),
    ColorSpace(
        name='HDTV',
        primaries=(
            Chromaticity(0.67, 0.33),
            Chromaticity(0.21, 0.71),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=rec709_transfer_function,
    ),
    ColorSpace(
        name='UHDTV',
        primaries=(
            Chromaticity(0.708, 0.292),
            Chromaticity(0.170, 0.797),
            Chromaticity(0.13, 0.046),
        ),
        white_point=white_point_D65,
        transfer_function=rec2020_transfer_function,
    ),
)
COLOR_SPACES_BY_NAME = {color_space.name: color_space for color_space in COLOR_SPACES}


class ColorSystem:
    """
    Builds and stores the RGB <-> XYZ transformation matrices for a color
    space, optionally adapted to another white point.

    The static color space metadata lives in `ColorSpace`, `WhitePoint` and
    `Chromaticity` instances; this class only computes and applies transforms.
    The implementation is based on
    http://brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html,
    http://www.brucelindbloom.com/Eqn_ChromAdapt.html,
    and https://scipython.com/blog/converting-a-spectrum-to-a-colour/.
    The formula derivation is simplified and is given in the (unpublished) paper.
    """

    supported_color_spaces: ClassVar[dict[str, ColorSpace]] = COLOR_SPACES_BY_NAME
    supported_white_points: ClassVar[dict[str, WhitePoint]] = WHITE_POINTS_BY_NAME

    def __init__(self, color_space: str, adaptation_white_point: str | None = None):
        """
        Initialize the ColorSystem object.
        The color space and optional white point of the chromatic adaptation
        must be among the supported options.
        """
        self.color_space: ColorSpace = self.supported_color_spaces[color_space]
        self.adaptation_white_point: WhitePoint | None = (
            self.supported_white_points[adaptation_white_point]
            if adaptation_white_point
            else None
        )
        # Reading chromaticity coordinates of the primary colors
        matrix_M = np.array([
            (primary.x, primary.y)
            for primary in self.color_space.primaries
        ]).T
        # Converting reduced chromaticity coordinates (x, y) to (x, y, z=1-x-y)
        matrix_M = np.vstack((matrix_M, 1 - matrix_M.sum(axis=0)))
        # Scaling the white point by the brightness of the Y component
        vector_WPi = self.color_space.white_point.normalized_xyz()
        # Calculating the inverse chromaticity matrix
        matrix_M_inv = np.linalg.inv(matrix_M)
        # Calculating the scaling vector
        vector_S = cast(npt.NDArray[np.floating], matrix_M_inv.dot(vector_WPi))
        # RGB -> XYZ transformation matrix
        self.matrix: npt.NDArray[np.floating] = matrix_M * vector_S[np.newaxis, :]
        # XYZ -> RGB transformation matrix
        self.inv_matrix: npt.NDArray[np.floating] = matrix_M_inv / vector_S[:, np.newaxis]
        # Optional chromatic adaptation
        if (
            self.adaptation_white_point is not None
            and self.adaptation_white_point != self.color_space.white_point
        ):
            vector_WPa = self.adaptation_white_point.normalized_xyz()
            # White scale in cone response domain
            vector_A = cast(npt.NDArray[np.floating], matrix_B.dot(vector_WPi) / matrix_B.dot(vector_WPa))
            # Applying the chromatic adaptation.
            # The adaptation matrix is B^-1 @ diag(1/a) @ B,
            # and its inverse is B^-1 @ diag(a) @ B.
            self.matrix = matrix_B_inv @ (matrix_B / vector_A[:, np.newaxis]) @ self.matrix
            self.inv_matrix = self.inv_matrix @ matrix_B_inv @ (matrix_B * vector_A[:, np.newaxis])
        # The matrices are used transposed
        #self.matrix = self.matrix.T
        #self.inv_matrix = self.inv_matrix.T

    @property
    def color_space_name(self) -> str:
        """ Name of the internal color space. """
        return self.color_space.name

    @property
    def white_point_name(self) -> str:
        """ Name of the adaptation white point, or an empty string if not adapted. """
        return '' if self.adaptation_white_point is None else self.adaptation_white_point.name

    @property
    def internal_white_point(self) -> WhitePoint:
        """ The white point that defines the color space itself. """
        return self.color_space.white_point

    @property
    def transfer_function(self) -> TransferFunction:
        """ The transfer function of the internal color space. """
        return self.color_space.transfer_function

    @override
    def __repr__(self) -> str:
        return (
            f'ColorSystem(color_space={self.color_space_name!r}, '
            f'adaptation_white_point={self.white_point_name!r})'
        )

    def xyz_to_rgb(
        self,
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """ Converts XYZ color array into a RGB color space array """
        # 1D implementation: rgb = self.inv_matrix @ xyz
        value1 = cast(npt.NDArray[np.floating], np.einsum('ij, j... -> i...', self.inv_matrix, value0))
        if error0 is None:
            error1 = None
        else:
            # 1D implementation: cov_rgb = self.inv_matrix @ cov_xyz @ self.inv_matrix.T
            error1 = propagate_covariance(error0, self.inv_matrix)
        return value1, error1

    def rgb_to_xyz(
        self,
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """ Converts RGB color array into the XYZ color space array """
        # 1D implementation: rgb = self.matrix @ xyz
        value1 = cast(npt.NDArray[np.floating], np.einsum('ij, j... -> i...', self.matrix, value0))
        if error0 is None:
            error1 = None
        else:
            # 1D implementation: cov_xyz = self.matrix @ cov_rgb @ self.matrix.T
            error1 = propagate_covariance(error0, self.matrix)
        return value1, error1

    def encode(
        self,
        values: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """ Applies the color space transfer function: linear -> display. """
        return self.transfer_function.encode(values)

    def decode(
        self,
        values: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """ Applies the inverse color space transfer function: display -> linear. """
        return self.transfer_function.decode(values)

    @staticmethod
    def spectrum_to_white_point(spectrum: Spectrum) -> tuple[float, float]:
        """ Returns (x, y) coordinates of the spectrum on the chromaticity diagram """
        chromaticity = chromaticity_of_spectrum(spectrum)
        return chromaticity.x, chromaticity.y


xyz_color_system = ColorSystem('CIE 1931 XYZ', 'Illuminant E')
default_color_system = ColorSystem('sRGB', 'Illuminant E')
