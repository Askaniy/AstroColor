import numpy as np
import pytest

import astrocolor as ac


def test_flat_photospectrum_reconstruction(ubv_filterset: ac.FilterSet):
    photospectrum = ac.Photospectrum(ubv_filterset, (1, 1, 1), name='test photospectrum')
    reconstructed = ac.get_spectrometry(photospectrum, ac.visible_range, strictly=True)
    np.testing.assert_allclose(reconstructed.spectral_dist, np.ones(ac.visible_range.size))


PhotospectralClass = type[ac.Photospectrum | ac.PhotospectralSet | ac.PhotospectralCube]
PlainClass = type[ac.Spectrum | ac.SpectralSet | ac.SpectralCube]

@pytest.mark.parametrize(
    'photospectral_type, plain_type, shape',
    [
        (ac.Photospectrum, ac.Spectrum, (3,)),
        (ac.PhotospectralSet, ac.SpectralSet, (3, 2)),
        (ac.PhotospectralCube, ac.SpectralCube, (3, 2, 2)),
    ],
)
def test_reconstruction_returns_plain_spectral_object(
    ubv_filterset: ac.FilterSet,
    photospectral_type: PhotospectralClass,
    plain_type: PlainClass,
    shape: tuple[int, ...],
):
    photometry = photospectral_type(ubv_filterset, np.ones(shape))
    nm = ubv_filterset.wavelength_nm

    reconstructed = ac.get_spectrometry(photometry, nm, strictly=True)

    assert isinstance(reconstructed, plain_type)
    assert not hasattr(reconstructed, 'photospectral_obj')
    assert reconstructed.spectral_dist.shape == (nm.size, *shape[1:])


def test_reconstruct_from_photometry_returns_plain_spectrum(ubv_filterset: ac.FilterSet):
    photometry = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], name='test photospectrum')
    nm = ubv_filterset.wavelength_nm

    direct = ac.reconstruct_from_photometry(photometry, nm)
    measured = ac.get_spectrometry(photometry, nm)

    assert isinstance(direct, ac.Spectrum)
    assert isinstance(measured, ac.Spectrum)
    np.testing.assert_allclose(direct.spectral_dist, measured.spectral_dist)
    np.testing.assert_array_equal(direct.wavelength_nm, measured.wavelength_nm)


def test_1d_reconstruction_propagates_uncertainty(ubv_filterset: ac.FilterSet):
    photometry = ac.Photospectrum(
        ubv_filterset,
        [0.5, 1.0, 2.0],
        uncertainty=[0.05, 0.1, 0.2],
    )

    reconstructed = ac.get_spectrometry(photometry, ubv_filterset.wavelength_nm, strictly=True)

    assert reconstructed.uncertainty is not None
    assert reconstructed.standard_deviation is not None
    assert reconstructed.standard_deviation.shape == reconstructed.spectral_dist.shape


def test_flatten_reconstructed_cube(ubv_filterset: ac.FilterSet):
    nm = ubv_filterset.wavelength_nm
    cube = ac.PhotospectralCube(ubv_filterset, np.ones((3, 2, 2)), name='test cube')
    reconstructed = ac.get_spectrometry(cube, nm, strictly=True)
    assert isinstance(reconstructed, ac.SpectralCube)

    flattened = reconstructed.flatten()
    assert isinstance(flattened, ac.SpectralSet)
    assert flattened.spectral_dist.shape == (nm.size, 4)
    np.testing.assert_allclose(flattened.spectral_dist, reconstructed.spectral_dist.reshape(nm.size, 4))
