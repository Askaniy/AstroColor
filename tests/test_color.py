from typing import cast

import numpy as np
import numpy.typing as npt

import astrocolor as ac

# === Color System Tests ===

def test_color_of_flat_spectrum():
    nm = np.arange(200, 800, 5)
    spectrum = ac.Spectrum(nm, np.ones(nm.size), name='Flat spectrum test')
    color = ac.ColorPoint.from_spectral_data(spectrum)
    target_rgb = np.ones(3)
    np.testing.assert_allclose(color.to_array(), target_rgb)
    color_system = ac.ColorSystem('sRGB', 'Illuminant E')
    color = color.to_color_system(color_system)
    np.testing.assert_allclose(color.to_array(), target_rgb)

def test_color_system_reversibility():
    """ Converting a color point between systems should be reversible to machine precision. """
    srgb = ac.ColorSystem('sRGB')
    xyz = ac.ColorSystem('CIE 1931 XYZ')
    color0 = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)  # in XYZ
    color1 = color0.to_color_system(srgb)
    color2 = color1.to_color_system(xyz)
    np.testing.assert_allclose(color0.to_array(), color2.to_array(), rtol=1e-13)

def test_adaptation_white_point():
    """ CIE 1931 RGB should have Illuminant E white point """
    rgb = ac.ColorSystem('CIE 1931 RGB')
    rgb_ = ac.ColorSystem('CIE 1931 RGB', adaptation_white_point='Illuminant E')
    color0 = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)  # in XYZ
    color1 = color0.to_color_system(rgb)
    color2 = color1.to_color_system(rgb_)
    np.testing.assert_allclose(color1.to_array(), color2.to_array(), rtol=1e-13)


def transform_covariance(
    matrix: npt.NDArray[np.floating],
    covariance: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """ Independent oracle for `matrix @ covariance @ matrix.T` with optional spatial axes. """
    return cast(
        npt.NDArray[np.floating],
        np.einsum('ij,jk...,lk->il...', matrix, covariance, matrix, optimize=True),
    )


class TestColorObjectUncertainty:
    """ Covariance propagation through color system conversions. """

    @staticmethod
    def _small_covariance() -> npt.NDArray[np.floating]:
        # A valid covariance close to the one produced by a real photospectrum.
        # The previous `matrix @ covariance @ matrix` formula gave a negative
        # variance for this matrix in sRGB.
        return np.array([
            [1.30509531e-07, 4.86807268e-08, 2.36600566e-07],
            [4.86807268e-08, 2.40317984e-08, 2.61052037e-07],
            [2.36600566e-07, 2.61052037e-07, 5.51256834e-06],
        ], dtype=np.float64)

    def test_xyz_to_rgb_covariance_uses_transpose(self):
        system = ac.ColorSystem('sRGB')
        covariance = self._small_covariance()

        _, actual = system.xyz_to_rgb(np.array([0.2, 0.3, 0.4]), covariance)

        assert actual is not None
        expected = transform_covariance(system.inv_matrix, covariance)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
        assert np.all(np.diag(actual) >= 0.)

    def test_rgb_to_xyz_covariance_uses_transpose(self):
        system = ac.ColorSystem('sRGB')
        covariance: npt.NDArray[np.floating] = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03],
        ], dtype=np.float64)

        _, actual = system.rgb_to_xyz(np.array([0.2, 0.3, 0.4]), covariance)

        assert actual is not None
        expected = transform_covariance(system.matrix, covariance)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)

    def test_color_point_conversion_does_not_create_negative_variances(self):
        covariance = self._small_covariance()
        color = ac.ColorPoint(np.array([0.2, 0.3, 0.4]), covariance)
        target_system = ac.ColorSystem('sRGB')

        converted = color.to_color_system(target_system)

        actual = converted.covariance_matrix
        assert actual is not None
        expected = transform_covariance(
            target_system.inv_matrix,
            transform_covariance(color.color_system.matrix, covariance),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
        assert np.all(np.diag(actual) >= 0.)

    def test_color_point_conversion_roundtrip_preserves_covariance(self):
        covariance = self._small_covariance()
        color = ac.ColorPoint(np.array([0.2, 0.3, 0.4]), covariance)
        source_system = color.color_system
        target_system = ac.ColorSystem('sRGB')

        restored = color.to_color_system(target_system).to_color_system(source_system)

        actual = restored.covariance_matrix
        assert actual is not None
        np.testing.assert_allclose(actual, covariance, rtol=1e-10, atol=1e-18)

    def test_color_line_conversion_propagates_covariance_along_spatial_axis(self):
        covariance = self._small_covariance()[:, :, np.newaxis] * np.array([1., 2., 3.])
        color = ac.ColorLine(np.ones((3, 3)) * 0.5, covariance)
        target_system = ac.ColorSystem('sRGB')

        converted = color.to_color_system(target_system)

        actual = converted.covariance_matrix
        assert actual is not None
        assert actual.shape == (3, 3, 3)
        expected = transform_covariance(
            target_system.inv_matrix,
            transform_covariance(color.color_system.matrix, covariance),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
        np.testing.assert_allclose(actual, np.swapaxes(actual, 0, 1), rtol=1e-12, atol=1e-18)
        assert np.all(np.diagonal(actual, axis1=0, axis2=1) >= 0.)

    def test_color_image_conversion_propagates_covariance_along_spatial_axes(self):
        scales = np.array([[1., 2.], [3., 4.]])
        covariance = self._small_covariance()[:, :, np.newaxis, np.newaxis] * scales
        color = ac.ColorImage(np.ones((3, 2, 2)) * 0.5, covariance)
        target_system = ac.ColorSystem('sRGB')

        converted = color.to_color_system(target_system)

        actual = converted.covariance_matrix
        assert actual is not None
        assert actual.shape == (3, 3, 2, 2)
        expected = transform_covariance(
            target_system.inv_matrix,
            transform_covariance(color.color_system.matrix, covariance),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
        np.testing.assert_allclose(actual, np.swapaxes(actual, 0, 1), rtol=1e-12, atol=1e-18)
        assert np.all(np.diagonal(actual, axis1=0, axis2=1) >= 0.)
