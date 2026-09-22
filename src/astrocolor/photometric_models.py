""" Experimental support of some photometric models used in TrueColorTools """

from typing import Final, cast, override

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CloughTocher2DInterpolator, PchipInterpolator

from .algebra import (
    mul_error,
    mul_value,
)
from .auxiliary import (
    integrate,
    parse_value_std,
)

# ------------ Phase Photometry Section ------------

class PhotometricModel:
    """
    Class to store a photometric model parameters,
    calculate their phase functions and albedo.
    """
    geometric_albedo: tuple[float, float | None] | None = None
    phase_integral: tuple[float, float | None] | None = None

    def __init__(self,
        params: dict[str, float | tuple[float]] = {},
        filter_or_nm: str | float = ''
    ) -> None:
        self.params: dict[str, float | tuple[float]] = params
        self.filter_or_nm: str | float = filter_or_nm
        self._integrate()

    def _integrate(self) -> None:
        """ Analytically or numerically computes usable values from a model parameters """
        raise NotImplementedError('Must be implemented in the inherited classes.')

    def phase_function(self, alpha: float | npt.ArrayLike): # input in radians!
        raise NotImplementedError('Must be implemented in the inherited classes.')

    @property
    def spherical_albedo(self):
        if self.geometric_albedo is not None and self.phase_integral is not None:
            a = mul_value(self.geometric_albedo[0], self.phase_integral[0])
            a_std = mul_error(self.geometric_albedo[0], self.geometric_albedo[1], self.phase_integral[0], self.phase_integral[1])
            return a, a_std
        else:
            return None

    def estimate_geometric_albedo(self, spherical_in_V: tuple[float, float | None]):
        """
        Returns exact or estimated value of geometric albedo with the flag showing the case.

        For phase integral estimation, model by Shevchenko et al. (2019) is used.
        https://ui.adsabs.harvard.edu/abs/2019A%26A...626A..87S/abstract
        q = 0.359 (± 0.005) + 0.47 (± 0.03) p, where `p` is geometric albedo.
        """
        if self.phase_integral is not None:
            geometric_in_V = spherical_in_V[0] / self.phase_integral[0]
            return geometric_in_V, False
        else:
            geometric_in_V = (cast(float, np.sqrt(0.359**2 + 4 * 0.47 * spherical_in_V[0]) - 0.359)) / (2 * 0.47)
            return geometric_in_V, True

    def estimate_spherical_albedo(self, geometric_in_V: tuple[float, float | None]):
        """
        Returns exact or estimated value of spherical albedo with the flag showing the case.

        For phase integral estimation, model by Shevchenko et al. (2019) is used.
        https://ui.adsabs.harvard.edu/abs/2019A%26A...626A..87S/abstract
        q = 0.359 (± 0.005) + 0.47 (± 0.03) p, where `p` is geometric albedo.
        """
        if self.phase_integral is not None:
            spherical_in_V = geometric_in_V[0] * self.phase_integral[0]
            return spherical_in_V, False
        else:
            spherical_in_V = geometric_in_V[0] * (0.359 + 0.47 * geometric_in_V[0])
            return spherical_in_V, True

class DefaultModel(PhotometricModel):
    """ Class for objects with unknown phase function """

    @override
    def _integrate(self) -> None:
        pass

class PhaseCoefficient(PhotometricModel):
    """
    One-parameter model with phase coefficient β (in mag/deg).

    The integration formula used was derived in M. Noland, J. Veverka, 1976:
    https://www.sciencedirect.com/science/article/abs/pii/0019103576901548

    The error propagation formula was used to handle the uncertainty.
    """

    _k: Final = 180 / np.pi * 0.4 * cast(float, np.log(10)) # ≈ 52.77

    @override
    def _integrate(self) -> None:
        beta, beta_std = parse_value_std(self.params['beta'])
        beta *= self._k
        _exp = np.exp(-np.pi * beta)
        denominator = 1 + beta * beta
        phase_integral = 2 * (1 + _exp) / denominator
        if beta_std is not None:
            phase_integral_std = beta_std * 2 * self._k * (np.pi * _exp + beta * phase_integral) / denominator
        else:
            phase_integral_std = None
        self.phase_integral = (phase_integral, phase_integral_std)

    @override
    def phase_function(self, alpha: float | npt.ArrayLike):
        beta, _ = parse_value_std(self.params['beta'])
        return np.exp(-self._k * beta * np.array(alpha))
        # equivalent to 10**(-0.4 * beta * alpha / np.pi * 180)


class Exponentials(PhotometricModel):
    """
    Describes phase function with a sum of exponentials.
    Usually one (like phase coefficient), two (Akimov, 1988) or three (Velikodsky, 2011) are used.
    """

    _A: npt.NDArray[np.floating] | None = None
    _mu: npt.NDArray[np.floating] | None = None

    @override
    def _integrate(self) -> None:
        n_exponentials = len(self.params) // 2
        self._A = np.empty(n_exponentials)
        self._mu = np.empty(n_exponentials)
        for i in range(n_exponentials):
            self._A[i] = parse_value_std(self.params[f'A_{i+1}'])[0]
            self._mu[i] = parse_value_std(self.params[f'mu_{i+1}'])[0]
        if (zero_phase_angle := self._A.sum()) != 1:
            # if function was not normalized, it shows geometric albedo at 0 phase angle
            self.geometric_albedo = zero_phase_angle, None
        self.phase_integral = 2 * np.sum(self._A * (1 + np.exp(-self._mu * np.pi)) / (1 + self._mu**2)) / zero_phase_angle, None

    @override
    def phase_function(self, alpha: float | npt.ArrayLike):
        phi = np.sum(self._A[:, np.newaxis] * np.exp(-self._mu[:, np.newaxis] * alpha), axis=0)
        if phi.size == 1:
            phi = phi[0]
        if self.geometric_albedo is not None:
           phi /= self.geometric_albedo[0]
        return phi


class HG(PhotometricModel):
    """
    Two-parameter magnitude system model:
    - H: “reduced magnitude” at zero phase angle
    - G: “slope parameter” that describes the shape of the phase curve

    See Bowell et al. (1989). Application of photometric models to asteroids.
    https://ui.adsabs.harvard.edu/abs/1989aste.conf..524B/abstract
    """

    @override
    def _integrate(self) -> None:
        g, g_std = parse_value_std(self.params['G'])
        q = 0.290 + 0.684 * g
        q_std = None if g_std is None else 0.684 * g_std
        self.phase_integral = (q, q_std)

    @override
    def phase_function(self, alpha: float | npt.ArrayLike):
        g, _ = parse_value_std(self.params['G'])
        alpha = np.array(alpha)
        alpha2 = 0.5 * alpha
        sin_alpha = np.sin(alpha)
        tan_alpha2 = np.tan(alpha2)
        sin_fraction = sin_alpha / (0.119 + 1.341 * sin_alpha - 0.754 * sin_alpha**2)
        phi1s = 1 - 0.986 * sin_fraction
        phi1l = np.exp(-3.332 * tan_alpha2**0.631)
        phi2s = 1 - 0.238 * sin_fraction
        phi2l = np.exp(-1.862 * tan_alpha2**1.218)
        w = np.exp(-90.56 * tan_alpha2**2)
        v = 1 - w
        phi1 = w * phi1s + v * phi1l
        phi2 = w * phi2s + v * phi2l
        return (1 - g) * phi1 + g * phi2
        # Approximation:
        # (1 - g) * np.exp(-3.33 * tan_alpha2**0.63) + g * np.exp(-1.87 * tan_alpha2**1.22)


class HG1G2(PhotometricModel):
    """
    Three-parameter magnitude system model:
    - H: “reduced magnitude” at zero phase angle
    - G1: the first “slope parameter”
    - G2: the second “slope parameter”

    See Muinonen et al. (2010). A three-parameter magnitude phase function for asteroids.
    https://www.sciencedirect.com/science/article/abs/pii/S001910351000151X
    """

    # HG1G2 base functions
    # https://github.com/milicolazo/Pyedra/blob/master/pyedra/datasets/penttila2016.csv

    _alpha: Final = np.array([
        0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 36, 39, 42, 45,
        48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114,
        117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150
    ]) / 180 * np.pi

    _phi1: Final = np.array([
        1, 0.99916667, 0.99833333, 0.9975, 0.99666667, 0.995, 0.99333333, 0.99166667, 0.99, 0.98833333,
        0.98666667, 0.985, 0.98333333, 0.98166667, 0.98, 0.97833333, 0.97666667, 0.975, 0.97333333,
        0.97166667, 0.97, 0.96833333, 0.96666667, 0.95833333, 0.95, 0.94166667, 0.93333333, 0.91666667,
        0.9, 0.88333333, 0.86666667, 0.85, 0.83333333, 0.81666667, 0.8, 0.78333333, 0.76666667, 0.75,
        0.7335651, 0.70205874, 0.67230993, 0.64424623, 0.6177952, 0.59288439, 0.56944137, 0.5473937,
        0.52666893, 0.50719463, 0.48889834, 0.47170764, 0.45555008, 0.44035322, 0.42604461, 0.41255183,
        0.39980241, 0.38772394, 0.37624396, 0.36529003, 0.35478972, 0.34467057, 0.33486016, 0.3068662,
        0.28089877, 0.25685779, 0.2346432, 0.21415492, 0.19529288, 0.177957, 0.16204721, 0.14746343,
        0.1341056, 0.1218788, 0.11070881, 0.10052653, 0.09126291, 0.08284886, 0.07521532, 0.06829321,
        0.06201346, 0.056307, 0.05110476, 0.0463475, 0.04201539, 0.03809846, 0.0345867, 0.03147014,
        0.02873879, 0.02638265, 0.02439175, 0.02275609, 0.02146569, 0.02048884, 0.019707, 0.01897989,
        0.01816723, 0.01712876, 0.01572421, 0.01381329, 0.01125575, 0.00791131, 0.0036397
    ])

    _phi2: Final = np.array([
        1, 0.99975, 0.9995, 0.99925, 0.999, 0.9985, 0.998, 0.9975, 0.997, 0.9965, 0.996, 0.9955, 0.995,
        0.9945, 0.994, 0.9935, 0.993, 0.9925, 0.992, 0.9915, 0.991, 0.9905, 0.99, 0.9875, 0.985, 0.9825,
        0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.945, 0.94, 0.935, 0.93, 0.925, 0.91993295,
        0.90940957, 0.89839618, 0.88692759, 0.87503862, 0.86276408, 0.85013879, 0.83719757, 0.82397523,
        0.81050658, 0.79682645, 0.78296964, 0.76897098, 0.75486528, 0.74068735, 0.72647201, 0.71225408,
        0.69806837, 0.6839497, 0.66993288, 0.65605272, 0.64234406, 0.62884169, 0.58974571, 0.55271081,
        0.51762803, 0.4843884, 0.45288296, 0.42300275, 0.39463881, 0.36768217, 0.34202387, 0.31755495,
        0.2941793, 0.27185223, 0.25054189, 0.23021646, 0.21084408, 0.19239292, 0.17483114, 0.15812689,
        0.14224835, 0.12716367, 0.11285041, 0.09932372, 0.08660818, 0.07472832, 0.06370871, 0.05357391,
        0.04434847, 0.03605695, 0.02872391, 0.0223739, 0.01701331, 0.0125758, 0.00897685, 0.00613196,
        0.00395661, 0.00236629, 0.00127648, 0.00060269, 0.00026038, 0.00016506
    ])

    _phi3: Final = np.array([
        1, 0.9980637, 0.99261656, 0.98406203, 0.9728036, 0.94378889, 0.90880018, 0.87106525, 0.83381185,
        0.7996872, 0.76901633, 0.74154373, 0.71701389, 0.69517129, 0.67576042, 0.65852577, 0.64321182,
        0.62956307, 0.61732399, 0.60623909, 0.59605283, 0.58650972, 0.57735424, 0.53374972, 0.49331752,
        0.45592704, 0.42144772, 0.36065328, 0.30965499, 0.26712671, 0.2317423, 0.20228652, 0.17798763,
        0.15818479, 0.14221716, 0.12942389, 0.11914414, 0.11071705, 0.10348178, 0.09076411, 0.07980824,
        0.07025206, 0.06173347, 0.05394627, 0.04680785, 0.04029153, 0.03437061, 0.02901839, 0.02420818,
        0.01991328, 0.01610701, 0.01276361, 0.00986117, 0.00737872, 0.0052953, 0.00358992, 0.00224164,
        0.00122947, 0.00053245, 0.00012962, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    hg1g2_phi1: Final = PchipInterpolator(_alpha, _phi1, extrapolate=True)
    hg1g2_phi2: Final = PchipInterpolator(_alpha, _phi2, extrapolate=True)
    hg1g2_phi3: Final = PchipInterpolator(_alpha, _phi3, extrapolate=True)

    @override
    def _integrate(self) -> None:
        g1, g1_std = parse_value_std(self.params['G_1'])
        g2, g2_std = parse_value_std(self.params['G_2'])
        q = 0.009082 + 0.4061 * g1 + 0.8092 * g2
        if g1_std is None and g2_std is None:
            q_std = None
        else:
            if g1_std is None:
                g1_std = 0
            if g2_std is None:
                g2_std = 0
            q_std = 0.4061 * g1_std + 0.8092 * g2_std
        self.phase_integral = (q, q_std)

    @override
    def phase_function(self, alpha: float | npt.ArrayLike):
        g1, _ = parse_value_std(self.params['G_1'])
        g2, _ = parse_value_std(self.params['G_2'])
        return g1 * self.hg1g2_phi1(alpha) + g2 * self.hg1g2_phi2(alpha) + (1 - g1 - g2) * self.hg1g2_phi3(alpha)


class Hapke(PhotometricModel):
    """
    Hapke photometric model. A common, partially empirical model.

    See Hapke, B. (1984). Bidirectional reflectance spectroscopy.
    Icarus, 59(1), 41–59. doi:10.1016/0019-1035(84)90054-x
    https://www.sciencedirect.com/science/article/abs/pii/001910358490054X
    """

    # Macroscopic roughness angle of Hapke 1984 model
    # Hapke, B. (1984). Bidirectional reflectance spectroscopy. Icarus, 59(1), 41–59. doi:10.1016/0019-1035(84)90054-x
    # https://www.sciencedirect.com/science/article/abs/pii/001910358490054X

    # "When Eqs. (57) and (58) for a macroscopically rough surface were inserted into the above equations,
    # analytic expressions for the physical albedo and integral phase function could not be obtained.
    # Hence, the integration was done numerically for values of θ up to 60°."

    _alpha: Final = np.array([
        0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 110, 120, 130, 140, 150, 160, 170, 180
    ]) / 180 * np.pi

    _theta: Final = np.array([0, 10, 20, 30, 40, 50, 60]) / 180 * np.pi

    _k: Final = np.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [1.00, 0.997, 0.991, 0.984, 0.974, 0.961, 0.943],
        [1.00, 0.994, 0.981, 0.965, 0.944, 0.918, 0.881],
        [1.00, 0.991, 0.970, 0.943, 0.909, 0.866, 0.809],
        [1.00, 0.988, 0.957, 0.914, 0.861, 0.797, 0.715],
        [1.00, 0.986, 0.947, 0.892, 0.825, 0.744, 0.644],
        [1.00, 0.984, 0.938, 0.871, 0.789, 0.692, 0.577],
        [1.00, 0.982, 0.926, 0.846, 0.748, 0.635, 0.509],
        [1.00, 0.979, 0.911, 0.814, 0.698, 0.570, 0.438],
        [1.00, 0.974, 0.891, 0.772, 0.637, 0.499, 0.366],
        [1.00, 0.968, 0.864, 0.719, 0.566, 0.423, 0.296],
        [1.00, 0.959, 0.827, 0.654, 0.487, 0.346, 0.231],
        [1.00, 0.946, 0.777, 0.575, 0.403, 0.273, 0.175],
        [1.00, 0.926, 0.708, 0.484, 0.320, 0.208, 0.130],
        [1.00, 0.894, 0.617, 0.386, 0.243, 0.153, 0.0936],
        [1.00, 0.840, 0.503, 0.290, 0.175, 0.107, 0.064],
        [1.00, 0.747, 0.374, 0.201, 0.117, 0.070, 0.041],
        [1.00, 0.590, 0.244, 0.123, 0.069, 0.040, 0.023],
        [1.00, 0.366, 0.127, 0.060, 0.032, 0.018, 0.010],
        [1.00, 0.128, 0.037, 0.016, 0.0085, 0.0047, 0.0026],
        [1.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    _A, _B = np.meshgrid(_alpha, _theta, indexing='ij')
    hapke_k: Final = CloughTocher2DInterpolator(np.column_stack([_A.ravel(), _B.ravel()]), _k.ravel())

    @staticmethod
    def henyey_greenstein(alpha: float | npt.ArrayLike, b: float, c: float) -> float | npt.NDArray[np.floating]:
        """ Double Henyey-Greenstein (1941) single particle scattering function """
        return (1 - b**2) * ((1 - c) * (1 + 2*b*np.cos(alpha) + b**2)**(-1.5) + c * (1 - 2*b*np.cos(alpha) + b**2)**(-1.5))

    @override
    def _integrate(self):
        w, _ = parse_value_std(self.params['w']) # single particle scattering albedo
        bo, _ = parse_value_std(self.params['bo']) # amplitude of opposition surge
        h, _ = parse_value_std(self.params['h']) # width of opposition surge
        b, _ = parse_value_std(self.params['b']) # Henyey-Greenstein single particle scattering function parameter
        c, _ = parse_value_std(self.params['c']) # Henyey-Greenstein single particle scattering function parameter
        theta = cast(float, np.radians(parse_value_std(self.params['theta'])[0])) # macroscopic roughness angle
        gamma = cast(float, np.sqrt(1 - w))
        r0 = (1 - gamma) / (1 + gamma) # bihemispherical reflectance
        # geometric albedo:
        C = 1 - r0 * (0.048 * theta + 0.0041 * theta**2) - r0**2 * (0.33 * theta + 0.0049 * theta**2)
        self.geometric_albedo = (
            cast(float, w / 8 * ((1 + bo) * self.henyey_greenstein(0, b, c) - 1) + C * 0.5 * r0 * (1 + r0 / 3)),
            None
        )
        # phase function:
        def phase_function(alpha: float | npt.ArrayLike):
            # without this check the output would be nan
            alpha = np.array(alpha, dtype='float')
            mask = alpha == 0
            phi = np.empty_like(alpha)
            phi[mask] = 1.
            alpha = alpha[~mask]
            alpha2 = alpha * 0.5
            B = bo / (1 + np.tan(alpha2) / h)
            phi[~mask] = w / 8 * ((1 + B) * self.henyey_greenstein(alpha, b, c) - 1) + 0.5 * r0 * (1 - r0)
            phi[~mask] *= 1 + np.sin(alpha2) * np.tan(alpha2) * np.log(np.tan(0.5 * alpha2))
            phi[~mask] += 2/3 * r0**2 * (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
            phi[~mask] *= self.hapke_k(alpha, theta) * phi[~mask] / self.geometric_albedo[0]
            return phi
        self.phase_function = phase_function
        # Numerical calculation of spherical albedo
        step = 0.01
        a = np.arange(0, np.pi, step) # 0°-180° phase angle array (radians)
        self.phase_integral = (
            2 * cast(float, integrate(phase_function(a) * np.sin(a), step=step, precisely=True)),
            None
        )
