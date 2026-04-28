import src.astro_color as ac

# Synthetic photometry calculation
spectrum = ac.Spectrum(
   wavelength_nm=[400, 500, 600, 700],
   spectral_dist=[1, 2, 2, 1]
)
johnson_system = ac.FilterSet(
    'Generic_Bessell.B',
    'Generic_Bessell.V',
    'Generic_Bessell.R'
)
photospectrum_BVR = ac.observe(spectrum, johnson_system)
flux_value, flux_error = ac.observe(spectrum, ac.Filter('Generic_Bessell.V'))

# Reconstruct photometry measurements into a smooth spectrum
reconstructed = ac.spectral_reconstruction(photospectrum_BVR, requested_wavelengths=[400, 700])

# Convert measurements directly between photometric systems
sloan_system = ac.FilterSet('SLOAN_SDSS.g', 'SLOAN_SDSS.r')
photospectrum_gr = ac.observe(photospectrum_BVR, sloan_system)

# True color calculation
color_xyz = ac.ColorPoint.from_spectral_data(ac.sun_CALSPEC)
color_system = ac.ColorSystem('sRGB', 'Illuminant E') # recommended
color_html = color_xyz.to_color_system(color_system).to_html()

# Coming soon
# cov_matrix = ac.ColorPoint.from_spectral_data(reconstructed).to_color_system(color_system).covariance_matrix
# spectrum = ac.BlackBodyModel(3000).determine_at_wavelengths([400, 700])
