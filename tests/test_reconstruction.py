import numpy as np
import pytest

import astrocolor as ac

#import matplotlib.pyplot as plt


def test_attach_photospectral_obj_is_propagated_to_reconstructed_objects(ubv_filterset: ac.FilterSet):
    nm = ubv_filterset.wavelength_nm
    photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], name='test photospectrum')

    attached = ac.get_spectrometry(
        photospectrum,
        nm,
        strictly=True,
        attach_photospectral_obj=True,
    )
    detached = ac.get_spectrometry(
        attached,
        nm,
        strictly=True,
        attach_photospectral_obj=False,
    )
    reattached = ac.get_spectrometry(
        attached,
        nm,
        strictly=True,
        attach_photospectral_obj=True,
    )

    assert isinstance(attached, ac.ReconstructedSpectrum)
    assert attached.photospectral_obj is not None
    assert not isinstance(detached, ac.ReconstructedSpectrum)
    assert getattr(detached, 'photospectral_obj', None) is None
    assert isinstance(reattached, ac.ReconstructedSpectrum)
    assert reattached.photospectral_obj is not None
    np.testing.assert_allclose(detached.spectral_dist, attached.spectral_dist)
    np.testing.assert_allclose(reattached.spectral_dist, attached.spectral_dist)


def test_flat_photospectrum_reconstruction(ubv_filterset: ac.FilterSet):
    photospectrum = ac.Photospectrum(ubv_filterset, (1, 1, 1), name='test photospectrum')
    reconstructed = ac.get_spectrometry(photospectrum, ac.visible_range, strictly=True)
    # for band in ubv_filterset:
    #     plt.plot(band.wavelength_nm, band.spectral_dist / band.spectral_dist.max(), '-', color='gray')
    # plt.plot(photospectrum.filter_set.mean_nm(), photospectrum.spectral_dist, 'o')
    # plt.plot(reconstructed.wavelength_nm, reconstructed.spectral_dist, '-', color='black')
    # plt.plot(ac.visible_range, reconstructed.spectral_dist)
    # plt.show()
    np.testing.assert_allclose(reconstructed.spectral_dist, np.ones(ac.visible_range.size))


PhotospectralClass = type[ac.Photospectrum | ac.PhotospectralSet | ac.PhotospectralCube]
ReconstructedClass = type[ac.ReconstructedSpectrum | ac.ReconstructedSpectralSet | ac.ReconstructedSpectralCube]
PlainClass = type[ac.Spectrum | ac.SpectralSet | ac.SpectralCube]

@pytest.mark.parametrize(
    'photospectral_type, reconstructed_type, plain_type, shape',
    [
        (ac.Photospectrum, ac.ReconstructedSpectrum, ac.Spectrum, (3,)),
        (ac.PhotospectralSet, ac.ReconstructedSpectralSet, ac.SpectralSet, (3, 2)),
        (ac.PhotospectralCube, ac.ReconstructedSpectralCube, ac.SpectralCube, (3, 2, 2)),
    ],
)
def test_attach_flag_selects_return_class(
    ubv_filterset: ac.FilterSet,
    photospectral_type: PhotospectralClass,
    reconstructed_type: ReconstructedClass,
    plain_type: PlainClass,
    shape: tuple[int, ...],
):
    obj = photospectral_type(ubv_filterset, np.ones(shape))
    nm = ubv_filterset.wavelength_nm

    attached = ac.get_spectrometry(
        obj,
        nm,
        strictly=True,
        attach_photospectral_obj=True,
    )
    detached = ac.get_spectrometry(
        obj,
        nm,
        strictly=True,
        attach_photospectral_obj=False,
    )

    assert isinstance(attached, reconstructed_type)
    assert isinstance(detached, plain_type)
    assert not isinstance(detached, reconstructed_type)
