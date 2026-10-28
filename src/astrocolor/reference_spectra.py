# Reference spectra of Sun and Vega

# CALSPEC: composite ultraviolet and optical absolute calibrated reference spectra of the HST standards
# Main link: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
# Database references: Bohlin, Gordon, & Tremblay (2014), Bohlin, Hubeny, & Rauch (2020), Bohlin et al. (2022), Bohlin et al. (2025).

# TSIS-1 HSRS: TSIS-1 Hybrid Solar Reference Spectrum (HSRS) - Full Spectrum Extension
# Main link: https://lasp.colorado.edu/lisird/data/tsis1_hsrs_binned_fs
# Related research: "Version 2 of the TSIS-1 Hybrid Solar Reference Spectrum and Extension to the Full Spectrum."
# Coddington, O. M., Richard, E. C., Harber, D., Pilewskie, P., Woods, T. N., Snow, M., et al. (2023).
# Earth and Space Science, 10, e2022EA002637. https://doi.org/10.1029/2022EA002637


from typing import cast

import numpy as np

from .config import Config
from .spectral_objects import Spectrum

# - Solar reference spectrum from CALSPEC
# Binned version of sun_reference_stis_002.fits
# Ultraviolet to near-infrared absolute flux distribution of the Sun to 2.7 microns. It combines absolute
# flux measurements from space and the ground with a model spectrum for the near-infrared (Colina, Bohlin, & Castelli 1996).
sun_calspec_data = cast(np.lib.npyio.NpzFile, np.load(Config.library_folder/'data/Sun_CALSPEC.npz'))
sun_calspec = Spectrum(sun_calspec_data['wavelength_nm'], sun_calspec_data['spectral_dist'], name='Sun [CALSPEC]')
del sun_calspec_data

# - Solar reference spectrum from TSIS-1 HSRS
# Time range: Dec 01, 2019 to Dec 06, 2019
# Spectral range: 115 nm to 200000 nm
# License: Creative Commons - Attribution 4.0 International
sun_tsis1_hsrs_data = cast(np.lib.npyio.NpzFile, np.load(Config.library_folder/'data/Sun_TSIS-1_HSRS.npz'))
sun_tsis1_hsrs = Spectrum(sun_tsis1_hsrs_data['wavelength_nm'], sun_tsis1_hsrs_data['spectral_dist'], name='Sun [TSIS-1 HSRS]')
del sun_tsis1_hsrs_data

# - Vegan reference spectrum from CALSPEC
# Binned version of alpha_lyr_stis_012.fits
# The composite flux standard consists of a special Kurucz 9550K model from 900-1152A (Kurucz 2003),
# IUE data from 1152-1675A, STIS CCD fluxes from 1675-10200A (Bohlin & Gilliland 2004a), and the 9550K model longward of 10200A
vega_data = cast(np.lib.npyio.NpzFile, np.load(Config.library_folder/'data/Vega_CALSPEC.npz'))
vega_calspec = Spectrum(vega_data['wavelength_nm'], vega_data['spectral_dist'], name='Vega [CALSPEC]')
del vega_data
