"""
Transfer functions between linear and display-encoded color values.

The module is intentionally independent of color-space metadata, spectral data
and filters: it only contains the nonlinear curves and their inverse. Standard
singletons are exposed at module level (`linear`, `srgb`, `gamma_22`,
`prophoto`, `rec709`, `rec2020`) and can be reused when constructing custom
`ColorSpace` objects.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, cast, override

import numpy as np
import numpy.typing as npt


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

    def encode_covariance(
        self,
        values: npt.NDArray[np.floating],
        covariance: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """
        Propagates covariance through `encode`.

        Stub for the next implementation step: nonlinear transfer functions
        need a local Jacobian, and spatial axes must be preserved.
        """
        _ = values, covariance  # Part of the future propagation contract.
        raise NotImplementedError(
            'Covariance propagation through transfer functions is not implemented yet.'
        )

    def decode_covariance(
        self,
        values: npt.NDArray[np.floating],
        covariance: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """
        Propagates covariance through `decode`.

        Stub for the next implementation step: nonlinear transfer functions
        need a local Jacobian, and spatial axes must be preserved.
        """
        _ = values, covariance  # Part of the future propagation contract.
        raise NotImplementedError(
            'Covariance propagation through transfer functions is not implemented yet.'
        )

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


class _PiecewiseTransferFunction(TransferFunction):
    """
    Shared implementation for power-law curves with a linear segment near zero.

    sRGB, ProPhoto RGB and Rec. 709/2020 differ only in their constants, not in
    the shape of the curve. Subclasses define those constants, while the common
    encode/decode logic lives here.
    """

    encode_threshold: ClassVar[float]
    decode_threshold: ClassVar[float]
    encode_slope: ClassVar[float]
    decode_slope: ClassVar[float]
    gamma: ClassVar[float]
    alpha: ClassVar[float] = 1.
    offset: ClassVar[float] = 0.

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = self.encode_slope * arr
        nonlinear = self.alpha * np.power(np.maximum(arr, 0.), 1./self.gamma) - self.offset
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < self.encode_threshold, linear, nonlinear),
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        arr = np.asarray(values, dtype=np.float64)
        linear = self.decode_slope * arr
        nonlinear = np.power(
            np.maximum((arr + self.offset) / self.alpha, 0.),
            self.gamma,
        )
        return cast(
            npt.NDArray[np.floating],
            np.where(arr < self.decode_threshold, linear, nonlinear),
        )


class SRGBTransferFunction(_PiecewiseTransferFunction):
    """ The piecewise sRGB transfer function, also used by Display P3. """

    encode_threshold: ClassVar[float] = 0.0031308
    decode_threshold: ClassVar[float] = 0.04045
    encode_slope: ClassVar[float] = 12.92
    decode_slope: ClassVar[float] = 1./12.92
    gamma: ClassVar[float] = 2.4
    alpha: ClassVar[float] = 1.055
    offset: ClassVar[float] = 0.055


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


class ProPhotoTransferFunction(_PiecewiseTransferFunction):
    """ The ProPhoto RGB transfer function: gamma 1.8 with a linear toe. """

    encode_threshold: ClassVar[float] = 1./512.
    decode_threshold: ClassVar[float] = 1./32.
    encode_slope: ClassVar[float] = 16.
    decode_slope: ClassVar[float] = 1./16.
    gamma: ClassVar[float] = 1.8


class Rec709TransferFunction(_PiecewiseTransferFunction):
    """ The Rec. 709 (HDTV) transfer function. """

    encode_threshold: ClassVar[float] = 0.018
    decode_threshold: ClassVar[float] = 0.081
    encode_slope: ClassVar[float] = 4.5
    decode_slope: ClassVar[float] = 1./4.5
    gamma: ClassVar[float] = 1./0.45
    alpha: ClassVar[float] = 1.099
    offset: ClassVar[float] = 0.099


class Rec2020TransferFunction(_PiecewiseTransferFunction):
    """ The Rec. 2020 (UHDTV) transfer function. """

    alpha: ClassVar[float] = 1.09929682680944
    beta: ClassVar[float] = 0.018053968510807
    encode_threshold: ClassVar[float] = beta
    decode_threshold: ClassVar[float] = 4.5 * beta
    encode_slope: ClassVar[float] = 4.5
    decode_slope: ClassVar[float] = 1./4.5
    gamma: ClassVar[float] = 1./0.45
    offset: ClassVar[float] = alpha - 1.


linear = LinearTransferFunction()
srgb = SRGBTransferFunction()
gamma_22 = GammaTransferFunction(2.2)
prophoto = ProPhotoTransferFunction()
rec709 = Rec709TransferFunction()
rec2020 = Rec2020TransferFunction()

__all__ = (
    'GammaTransferFunction',
    'LinearTransferFunction',
    'ProPhotoTransferFunction',
    'Rec2020TransferFunction',
    'Rec709TransferFunction',
    'SRGBTransferFunction',
    'TransferFunction',
    'gamma_22',
    'linear',
    'prophoto',
    'rec2020',
    'rec709',
    'srgb',
)
