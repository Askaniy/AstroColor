import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.errors import InvalidColorChannelsError


class TestColorBasics:
    """ Basic color construction and color-system conversion behavior. """

    def test_invalid_color_channels(self):
        for color_type in (ac.ColorPoint, ac.ColorLine, ac.ColorImage):
            with pytest.raises(InvalidColorChannelsError, match='three channels'):
                _ = color_type([1., 2.])

    def test_color_of_flat_spectrum(self):
        nm = np.arange(200, 800, 5)
        spectrum = ac.Spectrum(nm, np.ones(nm.size), name='Flat spectrum test')
        color = ac.ColorPoint.from_spectral_data(spectrum)
        target_rgb = np.ones(3)
        np.testing.assert_allclose(color.spectral_dist, target_rgb)
        color_system = ac.ColorSystem('sRGB', 'Illuminant E')
        color = color.to_color_system(color_system)
        np.testing.assert_allclose(color.spectral_dist, target_rgb)

    def test_color_system_reversibility(self):
        """ Converting a color point between systems should be reversible to machine precision. """
        srgb = ac.ColorSystem('sRGB')
        xyz = ac.ColorSystem('CIE 1931 XYZ')
        color0 = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)  # in XYZ
        color1 = color0.to_color_system(srgb)
        color2 = color1.to_color_system(xyz)
        np.testing.assert_allclose(color0.spectral_dist, color2.spectral_dist, rtol=1e-13)

    def test_adaptation_white_point(self):
        """ CIE 1931 RGB should have Illuminant E white point """
        rgb = ac.ColorSystem('CIE 1931 RGB')
        rgb_ = ac.ColorSystem('CIE 1931 RGB', adaptation_white_point='Illuminant E')
        color0 = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)  # in XYZ
        color1 = color0.to_color_system(rgb)
        color2 = color1.to_color_system(rgb_)
        np.testing.assert_allclose(color1.spectral_dist, color2.spectral_dist, rtol=1e-13)


class TestColorObjectUncertainty:
    """ Covariance propagation through color system conversions. """

    @staticmethod
    def transform_covariance(
        matrix: npt.NDArray[np.floating],
        covariance: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """ Independent oracle for `matrix @ covariance @ matrix.T` with optional spatial axes. """
        return cast(
            npt.NDArray[np.floating],
            np.einsum('ij,jk...,lk->il...', matrix, covariance, matrix, optimize=True),
        )

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
        expected = self.transform_covariance(system.inv_matrix, covariance)
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
        expected = self.transform_covariance(system.matrix, covariance)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)

    def test_color_point_conversion_does_not_create_negative_variances(self):
        covariance = self._small_covariance()
        color = ac.ColorPoint(np.array([0.2, 0.3, 0.4]), covariance)
        target_system = ac.ColorSystem('sRGB')

        converted = color.to_color_system(target_system)

        actual = converted.covariance_matrix
        assert actual is not None
        expected = self.transform_covariance(
            target_system.inv_matrix,
            self.transform_covariance(color.color_system.matrix, covariance),
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
        expected = self.transform_covariance(
            target_system.inv_matrix,
            self.transform_covariance(color.color_system.matrix, covariance),
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
        expected = self.transform_covariance(
            target_system.inv_matrix,
            self.transform_covariance(color.color_system.matrix, covariance),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
        np.testing.assert_allclose(actual, np.swapaxes(actual, 0, 1), rtol=1e-12, atol=1e-18)
        assert np.all(np.diagonal(actual, axis1=0, axis2=1) >= 0.)


class TestGammaCorrectedInput:
    """ Gamma-corrected input is decoded at construction; storage stays linear. """

    def test_srgb_d65_to_srgb_e(self):
        source = ac.ColorSystem('sRGB')
        target = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([0.5, 0.25, 0.1])
        color = ac.ColorPoint(encoded, color_system=source, gamma_corrected=True)

        np.testing.assert_allclose(color.spectral_dist, source.decode(encoded))
        converted = color.to_color_system(target)

        expected = target.inv_matrix @ source.matrix @ source.decode(encoded)
        np.testing.assert_allclose(converted.spectral_dist, expected)
        assert converted.color_system is target
        # The returned object stays linear; render()/encode() produce display values.
        np.testing.assert_allclose(
            converted.render(ac.DisplaySettings(color_system=target, maximize_brightness=False)),
            target.encode(expected),
        )
        # Treating the same encoded values as linear must give a different result.
        linear_color = ac.ColorPoint(encoded, color_system=source)
        assert not np.allclose(
            converted.spectral_dist,
            linear_color.to_color_system(target).spectral_dist,
        )

    def test_gamma_corrected_input_is_decoded_on_construction(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([0.5, 0.25, 0.1])
        color = ac.ColorPoint(encoded, color_system=system, gamma_corrected=True)

        np.testing.assert_allclose(color.spectral_dist, system.decode(encoded))
        converted = color.to_color_system(system)
        np.testing.assert_allclose(converted.spectral_dist, color.spectral_dist)
        assert converted.color_system is system

    def test_gamma_corrected_line_is_decoded_on_construction(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([
            [0.5, 0.25],
            [0.25, 0.5],
            [0.1, 0.2],
        ])
        line = ac.ColorLine(encoded, color_system=system, gamma_corrected=True)

        np.testing.assert_allclose(line.spectral_dist, system.decode(encoded))

    def test_gamma_corrected_input_rejects_uncertainty(self):
        with pytest.raises(
            NotImplementedError,
            match='Covariance propagation through transfer functions',
        ):
            _ = ac.ColorPoint(
                np.array([0.5, 0.25, 0.1]),
                np.eye(3) * 0.01,
                color_system=ac.ColorSystem('sRGB'),
                gamma_corrected=True,
            )


class TestGamutHandling:
    """ Out-of-gamut colors are desaturated without changing their shape. """

    def test_out_of_gamut_color_point_is_desaturated(self):
        color = ac.ColorPoint(
            np.array([0.25146044, -0.26420973, 1.2808453]),
            color_system=ac.xyz_color_system,
        )

        converted = color.to_color_system(ac.ColorSystem('sRGB'))

        assert converted.spectral_dist.shape == (3,)
        np.testing.assert_allclose(converted.spectral_dist, [1.2686396, 0., 2.10821783], rtol=1e-6)
        assert np.all(converted.spectral_dist >= 0)

    def test_out_of_gamut_line_desaturates_only_affected_points(self):
        xyz = np.array([
            [0.3, 0.25146044],
            [0.4, -0.26420973],
            [0.5, 1.2808453],
        ])
        color = ac.ColorLine(xyz, color_system=ac.xyz_color_system)
        target = ac.ColorSystem('sRGB')

        converted = color.to_color_system(target).spectral_dist
        raw = target.xyz_to_rgb(xyz)[0]

        np.testing.assert_allclose(converted[:, 0], raw[:, 0])
        minimum = float(raw[:, 1].min())
        np.testing.assert_allclose(converted[:, 1], raw[:, 1] - minimum)
        assert np.all(converted >= 0)


class TestDisplaySettings:
    """ Display settings and the render pipeline. """

    def test_default_settings_use_recommended_color_system(self):
        assert ac.DEFAULT_SETTINGS.color_system is ac.default_color_system
        assert ac.DEFAULT_SETTINGS.gamma_correction
        assert ac.DEFAULT_SETTINGS.maximize_brightness
        assert ac.DEFAULT_SETTINGS.scale_factor == 1.0
        assert not ac.LINEAR_SETTINGS.gamma_correction
        assert not ac.LINEAR_SETTINGS.maximize_brightness

    def test_render_default_path(self):
        color = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)
        raw = color.spectral_dist.copy()
        np.testing.assert_allclose(
            color.render(),
            [0.92652538, 0.99430624, 1.0],
            rtol=1e-7,
        )
        assert color.to_html() == '#ecfeff'
        np.testing.assert_allclose(color.spectral_dist, raw)

    def test_render_linear_settings(self):
        color = ac.get_color(ac.sun_tsis1_hsrs)
        np.testing.assert_allclose(color.render(ac.LINEAR_SETTINGS), color.spectral_dist)

    def test_render_custom_color_system(self):
        color = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)
        rendered = color.render(ac.DisplaySettings(color_system=ac.xyz_color_system))
        raw = color.spectral_dist
        raw_max = float(raw.max())
        np.testing.assert_allclose(rendered, raw / raw_max)

    def test_render_scale_factor_and_no_maximize(self):
        color = ac.get_color(ac.sun_tsis1_hsrs)
        settings = ac.DisplaySettings(maximize_brightness=False, scale_factor=0.5)
        np.testing.assert_allclose(
            color.render(settings),
            ac.default_color_system.encode(color.spectral_dist * 0.5),
        )

    def test_render_uses_target_color_space_transfer_function(self):
        adobe = ac.ColorSystem('Adobe RGB')
        color = ac.ColorPoint(np.full(3, 0.5), color_system=adobe)
        rendered = color.render(ac.DisplaySettings(color_system=adobe, maximize_brightness=False))
        np.testing.assert_allclose(rendered, math.pow(0.5, 1./2.2), rtol=1e-15)

    def test_render_does_not_modify_object(self):
        color = ac.get_color(ac.sun_tsis1_hsrs)
        before = color.spectral_dist.copy()
        _ = color.render()
        _ = color.to_html()
        np.testing.assert_allclose(color.spectral_dist, before)

    def test_render_image(self):
        image = ac.ColorImage(
            np.ones((3, 2, 2)) * 0.5,
            color_system=ac.default_color_system,
        )
        rendered = image.render(ac.DisplaySettings(maximize_brightness=False))
        expected = ac.default_color_system.encode(np.ones((3, 2, 2)) * 0.5)
        np.testing.assert_allclose(rendered, expected)
        assert rendered.shape == (3, 2, 2)

    def test_invalid_scale_factor(self):
        for value in (-1.0, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                _ = ac.DisplaySettings(scale_factor=value)


class TestGetColor:
    """ The one-call shortcut for linear color data. """

    def test_get_color_uses_recommended_system(self):
        color = ac.get_color(ac.sun_tsis1_hsrs)
        assert isinstance(color, ac.ColorPoint)
        assert color.color_system is ac.default_color_system
        assert color.to_html() == '#ecfeff'

    def test_get_color_preserves_dimensions(self):
        nm = np.arange(400, 501, 10)
        spectrum = ac.Spectrum(nm, np.ones(nm.size))
        spectral_set = ac.SpectralSet(nm, np.ones((nm.size, 2)))
        cube = ac.SpectralCube(nm, np.ones((nm.size, 2, 2)))

        assert isinstance(ac.get_color(spectrum), ac.ColorPoint)
        assert isinstance(ac.get_color(spectral_set), ac.ColorLine)
        assert isinstance(ac.get_color(cube), ac.ColorImage)

    def test_get_color_can_return_xyz(self):
        color = ac.get_color(ac.sun_tsis1_hsrs, ac.xyz_color_system)
        assert color.color_system is ac.xyz_color_system

    def test_get_color_preserves_an_equal_custom_system(self):
        target = ac.ColorSystem('CIE 1931 XYZ', 'Illuminant E')
        color = ac.get_color(ac.sun_tsis1_hsrs, target)
        assert color.color_system is target
        np.testing.assert_allclose(
            color.spectral_dist,
            ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs).spectral_dist,
        )

    def test_grayscale_is_linear_luminance(self):
        color = ac.get_color(ac.sun_tsis1_hsrs)
        expected = cast(
            float,
            color.to_color_system(ac.xyz_color_system).spectral_dist[1],
        )
        assert color.grayscale() == pytest.approx(expected)
