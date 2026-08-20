from typing import cast

import numpy as np
import numpy.typing as npt

from .config import Config
from .spectral_objects import Spectrum

sun_data = cast(np.lib.npyio.NpzFile, np.load(Config.library_folder/'data/Sun_CALSPEC.npz'))
sun_CALSPEC = Spectrum(sun_data['wavelength_nm'], sun_data['spectral_dist'], name='Sun [CALSPEC]')
del sun_data

vega_data = cast(np.lib.npyio.NpzFile, np.load(Config.library_folder/'data/Vega_CALSPEC.npz'))
vega_CALSPEC = Spectrum(vega_data['wavelength_nm'], vega_data['spectral_dist'], name='Vega [CALSPEC]')
del vega_data



# TODO: StellarModel, RayleighModel, SynchrotronModel, ...


h = 6.626e-34 # Planck constant
c = 299792458 # Speed of light
k = 1.381e-23 # Boltzmann constant
const1 = 2 * h * c * c # * np.pi to get exitance (W/m2) in the assumption of Lambertian surface
const2 = h * c / k


class BlackBodyModel:
    """ Creates a Spectrum object based on Planck's law and redshift formulas """

    def __init__(self, temperature: float, velocity: float = 0., vII: float = 0.) -> None:
        self.T: float = temperature
        self.v: float = velocity
        self.vII: float = vII

    def planck_radiance(
        self,
        nm: float | npt.NDArray[np.integer | np.floating]
    ) -> float | npt.NDArray[np.floating]:
        m = nm * 1e-9
        radiance = const1 / (m**5 * (np.exp(const2 / (m * self.T)) - 1))
        return radiance * 1e-9 # per m -> per nm

    def determine_at_trusted_wavelengths(
        self,
        requested_wavelengths: npt.NDArray[np.integer]
    ) -> Spectrum:
        """
        Directly uses the provided wavelength grid to create a new object. Non-strict!
        See `get_spectrometry()` for the general case.
        """
        doppler = 1
        grav = 1
        if self.T == 0:
            physics = False
        else:
            physics = True
            if self.v != 0:
                if abs(self.v) != 1:
                    doppler = cast(float, np.sqrt((1-self.v) / (1+self.v)))
                else:
                    physics = False
            if self.vII != 0:
                if self.vII != 1:
                    grav = cast(float, np.exp(-0.5 * self.vII**2))
                else:
                    physics = False
        if physics:
            br = self.planck_radiance(requested_wavelengths * doppler * grav)
        else:
            br = np.zeros(requested_wavelengths.size)
        return Spectrum(requested_wavelengths, br, name=f'BB with T={round(self.T)} K')
