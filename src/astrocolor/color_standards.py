"""
Color standards and runtime color-system transforms.

The module is organized in two layers:

- Immutable descriptors define the standards: `Chromaticity` stores CIE xy
  coordinates, `WhitePoint` attaches a name to a chromaticity, and `ColorSpace`
  combines RGB primaries, a white point and a transfer function. The curves
  themselves live in the lightweight `transfer_functions` module.
- `ColorSystem` is the runtime object. It resolves a `ColorSpace` (by name or
  directly), optionally applies Bradford chromatic adaptation to another white
  point, and computes and stores the resulting RGB <-> XYZ matrices.

Standard white points and color spaces are exposed through the `WHITE_POINTS`,
`COLOR_SPACES` and `ColorSystem.supported_*` registries. These registries are
only convenience lookups: custom `WhitePoint` and `ColorSpace` instances can be
passed directly to `ColorSystem`, so users do not need to register anything.

The public color pipeline uses linear values internally. Transfer functions are
applied only at the display boundary (`ColorSystem.encode`/`decode`) or when a
caller explicitly marks values as gamma-corrected.
"""

from dataclasses import dataclass
from typing import ClassVar, Self, TypeVar, cast, override

import numpy as np
import numpy.typing as npt

from .filter_objects import FilterSet
from .measurements import get_photometry
from .reference_spectra import sun_tsis1_hsrs, vega_calspec
from .spectral_objects import Spectrum
from .transfer_functions import (
    TransferFunction,
    gamma_22,
    linear,
    prophoto,
    rec709,
    rec2020,
    srgb,
)
from .uncertainty import propagate_covariance

_StandardType = TypeVar('_StandardType')

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

    @classmethod
    def from_xy(cls, name: str, x: float, y: float) -> Self:
        """ Creates a white point from CIE xy coordinates. """
        return cls(name, Chromaticity(x, y))

    @classmethod
    def from_spectrum(cls, name: str, spectrum: Spectrum) -> Self:
        """ Creates a white point from the CIE xy chromaticity of a spectrum. """
        return cls(name, chromaticity_of_spectrum(spectrum))

    def same_chromaticity(self, other: 'WhitePoint') -> bool:
        """ Checks whether two white points describe the same chromaticity, ignoring names. """
        return bool(np.allclose(
            self.normalized_xyz(),
            other.normalized_xyz(),
            rtol=0.,
            atol=1e-12,
        ))

    def __post_init__(self) -> None:
        if not np.isfinite((self.x, self.y)).all():
            raise ValueError(f'White point "{self.name}" must have finite coordinates.')
        if self.chromaticity.y <= 0:
            raise ValueError(f'White point "{self.name}" must have a positive y coordinate.')


@dataclass(frozen=True)
class ColorSpace:
    """
    Static description of an RGB color space.

    `primaries` are ordered red, green, blue. `white_point` is the native
    white point of the space, and `transfer_function` maps its linear values
    to display-encoded values and back. Custom instances can be passed directly
    to `ColorSystem`; they do not need to be registered.
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
white_point_A = WhitePoint.from_xy('Illuminant A', 0.44758, 0.40745)
white_point_B = WhitePoint.from_xy('Illuminant B', 0.34842, 0.35161)
white_point_C = WhitePoint.from_xy('Illuminant C', 0.31006, 0.31616)
white_point_D50 = WhitePoint.from_xy('Illuminant D50', 0.34567, 0.35850)
white_point_D55 = WhitePoint.from_xy('Illuminant D55', 0.33242, 0.34743)
white_point_D65 = WhitePoint.from_xy('Illuminant D65', 0.31272, 0.32903)
white_point_D75 = WhitePoint.from_xy('Illuminant D75', 0.29902, 0.31485)
white_point_D93 = WhitePoint.from_xy('Illuminant D93', 0.28315, 0.29711)
white_point_E = WhitePoint.from_xy('Illuminant E', 1./3., 1./3.)
white_point_Vega = WhitePoint.from_spectrum('Vega', vega_calspec)
white_point_Sun = WhitePoint.from_spectrum('Sun', sun_tsis1_hsrs)

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
        transfer_function=linear,
    ),
    ColorSpace(
        name='CIE 1931 RGB',
        primaries=(
            Chromaticity(0.73474284, 0.26525716),
            Chromaticity(0.27377903, 0.7174777),
            Chromaticity(0.16655563, 0.00891073),
        ),
        white_point=white_point_E,
        transfer_function=linear,
    ),
    ColorSpace(
        name='sRGB',
        primaries=(
            Chromaticity(0.64, 0.33),
            Chromaticity(0.30, 0.60),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=srgb,
    ),
    ColorSpace(
        name='Display P3',
        primaries=(
            Chromaticity(0.68, 0.32),
            Chromaticity(0.265, 0.69),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=srgb,
    ),
    ColorSpace(
        name='Adobe RGB',
        primaries=(
            Chromaticity(0.64, 0.33),
            Chromaticity(0.21, 0.71),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=gamma_22,
    ),
    ColorSpace(
        name='Wide Gamut RGB',
        primaries=(
            Chromaticity(0.7347, 0.2653),
            Chromaticity(0.1152, 0.8264),
            Chromaticity(0.1566, 0.0177),
        ),
        white_point=white_point_D50,
        transfer_function=gamma_22,
    ),
    ColorSpace(
        name='ProPhoto RGB',
        primaries=(
            Chromaticity(0.734699, 0.265301),
            Chromaticity(0.159597, 0.840403),
            Chromaticity(0.036598, 0.000105),
        ),
        white_point=white_point_D50,
        transfer_function=prophoto,
    ),
    ColorSpace(
        name='HDTV',
        primaries=(
            Chromaticity(0.67, 0.33),
            Chromaticity(0.21, 0.71),
            Chromaticity(0.15, 0.06),
        ),
        white_point=white_point_D65,
        transfer_function=rec709,
    ),
    ColorSpace(
        name='UHDTV',
        primaries=(
            Chromaticity(0.708, 0.292),
            Chromaticity(0.170, 0.797),
            Chromaticity(0.13, 0.046),
        ),
        white_point=white_point_D65,
        transfer_function=rec2020,
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

    def __init__(
        self,
        color_space: ColorSpace | str,
        adaptation_white_point: WhitePoint | str | None = None,
    ):
        """
        Initialize the ColorSystem object.

        `color_space` can be a name from `supported_color_spaces` or a custom
        `ColorSpace` instance. `adaptation_white_point` can likewise be a name
        from `supported_white_points`, a custom `WhitePoint`, or `None` for no
        chromatic adaptation. Passing objects directly requires no registration.
        """
        self.color_space: ColorSpace = self._resolve_color_space(color_space)
        self.adaptation_white_point: WhitePoint | None = (
            self._resolve_white_point(adaptation_white_point)
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
        try:
            matrix_M_inv = np.linalg.inv(matrix_M)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f'Color space "{self.color_space.name}" has degenerate primaries.'
            ) from error
        # Calculating the scaling vector
        vector_S = cast(npt.NDArray[np.floating], matrix_M_inv.dot(vector_WPi))
        # RGB -> XYZ transformation matrix
        self.matrix: npt.NDArray[np.floating] = matrix_M * vector_S[np.newaxis, :]
        # XYZ -> RGB transformation matrix
        self.inv_matrix: npt.NDArray[np.floating] = matrix_M_inv / vector_S[:, np.newaxis]
        # Optional chromatic adaptation
        if (
            self.adaptation_white_point is not None
            and not self.adaptation_white_point.same_chromaticity(self.color_space.white_point)
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
    def effective_white_point(self) -> WhitePoint:
        """ The white point after the optional chromatic adaptation. """
        return self.adaptation_white_point or self.color_space.white_point

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

    @override
    def __eq__(self, other: object) -> bool:
        """
        Color systems are equal when they define the same RGB <-> XYZ transform.

        Matrices, rather than metadata, are compared, so explicitly passing the
        native white point of a space is equivalent to passing no adaptation.
        """
        if not isinstance(other, ColorSystem):
            return NotImplemented
        return (
            np.array_equal(self.matrix, other.matrix)
            and np.array_equal(self.inv_matrix, other.inv_matrix)
        )

    @override
    def __hash__(self) -> int:
        """ Hashes the transformation matrices consistently with `__eq__`. """
        return hash((
            self.matrix.astype(np.float64, copy=False).tobytes(),
            self.inv_matrix.astype(np.float64, copy=False).tobytes(),
        ))

    @staticmethod
    def _resolve_registered(
        value: str,
        registry: dict[str, _StandardType],
        kind: str,
    ) -> _StandardType:
        """ Looks up a named standard and reports all known names on failure. """
        try:
            return registry[value]
        except KeyError as error:
            available = ', '.join(registry)
            raise ValueError(
                f'Unknown {kind} {value!r}. Available: {available}.'
            ) from error

    @classmethod
    def _resolve_color_space(cls, value: ColorSpace | str) -> ColorSpace:
        """ Resolves a color-space name or returns a custom color space as is. """
        if isinstance(value, ColorSpace):
            return value
        return cls._resolve_registered(value, cls.supported_color_spaces, 'color space')

    @classmethod
    def _resolve_white_point(
        cls,
        value: WhitePoint | str | None,
    ) -> WhitePoint | None:
        """ Resolves a white-point name, keeps a custom white point, or returns None. """
        if value is None:
            return None
        if isinstance(value, str):
            if not value:
                return None
            return cls._resolve_registered(value, cls.supported_white_points, 'white point')
        return value

    @staticmethod
    def _apply_matrix(
        matrix: npt.NDArray[np.floating],
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """ Applies a color matrix to values and, when present, to their covariance. """
        # 1D implementation: value1 = matrix @ value0
        value1 = cast(npt.NDArray[np.floating], np.einsum('ij, j... -> i...', matrix, value0))
        if error0 is None:
            error1 = None
        else:
            # 1D implementation: error1 = matrix @ error0 @ matrix.T
            error1 = propagate_covariance(error0, matrix)
        return value1, error1

    def xyz_to_rgb(
        self,
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """ Converts XYZ color array into a RGB color space array """
        return self._apply_matrix(self.inv_matrix, value0, error0)

    def rgb_to_xyz(
        self,
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """ Converts RGB color array into the XYZ color space array """
        return self._apply_matrix(self.matrix, value0, error0)

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

    def encode_covariance(
        self,
        values: npt.NDArray[np.floating],
        covariance: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """ Propagates covariance through the transfer function: linear -> display. """
        return self.transfer_function.encode_covariance(values, covariance)

    def decode_covariance(
        self,
        values: npt.NDArray[np.floating],
        covariance: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """ Propagates covariance through the inverse transfer function: display -> linear. """
        return self.transfer_function.decode_covariance(values, covariance)


xyz_color_system = ColorSystem('CIE 1931 XYZ', 'Illuminant E')
default_color_system = ColorSystem('sRGB', 'Illuminant E')
