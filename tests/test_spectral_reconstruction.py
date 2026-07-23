import numpy as np
import astrocolor as ac
#import matplotlib.pyplot as plt


def test_flat_photospectrum_reconstruction(ubv_filterset):
    photospectrum = ac.Photospectrum(ubv_filterset, (1, 1, 1), name='test photospectrum')
    reconstructed = photospectrum.determine_at_wavelengths(ac.visible_range, strictly=True)
    # for band in ubv_filterset:
    #     plt.plot(band.wavelength_nm, band.spectral_dist / band.spectral_dist.max(), '-', color='gray')
    # plt.plot(photospectrum.filter_set.mean_nm(), photospectrum.spectral_dist, 'o')
    # plt.plot(reconstructed.wavelength_nm, reconstructed.spectral_dist, '-', color='black')
    # plt.plot(ac.visible_range, reconstructed.spectral_dist)
    # plt.show()
    np.testing.assert_allclose(reconstructed.spectral_dist, np.ones(ac.visible_range.size))
