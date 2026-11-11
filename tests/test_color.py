import math
from typing import cast, override

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.errors import InvalidColorChannelsError

# === Color System Tests ===

def test_invalid_color_channels():
    for color_type in (ac.ColorPoint, ac.ColorLine, ac.ColorImage):
        with pytest.raises(InvalidColorChannelsError, match='three channels'):
            _ = color_type([1., 2.])

def test_color_of_flat_spectrum():
    nm = np.arange(200, 800, 5)
    spectrum = ac.Spectrum(nm, np.ones(nm.size), name='Flat spectrum test')
    color = ac.ColorPoint.from_spectral_data(spectrum)
    target_rgb = np.ones(3)
    np.testing.assert_allclose(color.spectral_dist, target_rgb)
    color_system = ac.ColorSystem('sRGB', 'Illuminant E')
    color = color.to_color_system(color_system)
    np.testing.assert_allclose(color.spectral_dist, target_rgb)

def test_color_system_reversibility():
    """ Converting a color point between systems should be reversible to machine precision. """
    srgb = ac.ColorSystem('sRGB')
    xyz = ac.ColorSystem('CIE 1931 XYZ')
    color0 = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)  # in XYZ
    color1 = color0.to_color_system(srgb)
    color2 = color1.to_color_system(xyz)
    np.testing.assert_allclose(color0.spectral_dist, color2.spectral_dist, rtol=1e-13)

def test_adaptation_white_point():
    """ CIE 1931 RGB should have Illuminant E white point """
    rgb = ac.ColorSystem('CIE 1931 RGB')
    rgb_ = ac.ColorSystem('CIE 1931 RGB', adaptation_white_point='Illuminant E')
    color0 = ac.ColorPoint.from_spectral_data(ac.sun_tsis1_hsrs)  # in XYZ
    color1 = color0.to_color_system(rgb)
    color2 = color1.to_color_system(rgb_)
    np.testing.assert_allclose(color1.spectral_dist, color2.spectral_dist, rtol=1e-13)


def transform_covariance(
    matrix: npt.NDArray[np.floating],
    covariance: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """ Independent oracle for `matrix @ covariance @ matrix.T` with optional spatial axes. """
    return cast(
        npt.NDArray[np.floating],
        np.einsum('ij,jk...,lk->il...', matrix, covariance, matrix, optimize=True),
    )


class _NoMatrixColorSystem(ac.ColorSystem):
    """ Fails the test if a matrix conversion is performed. """

    @override
    def rgb_to_xyz(
        self,
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None = None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        raise AssertionError('rgb_to_xyz() must not be called for equal color systems')

    @override
    def xyz_to_rgb(
        self,
        value0: npt.NDArray[np.floating],
        error0: npt.NDArray[np.floating] | None = None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        raise AssertionError('xyz_to_rgb() must not be called for equal color systems')


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


class TestColorSpaceMetadata:
    """ Human-readable color space definitions and registries. """

    def test_color_space_registry_is_self_consistent(self):
        for name, space in ac.ColorSystem.supported_color_spaces.items():
            assert name == space.name
            assert isinstance(space, ac.ColorSpace)
            assert len(space.primaries) == 3
            assert all(isinstance(primary, ac.Chromaticity) for primary in space.primaries)
            assert isinstance(space.white_point, ac.WhitePoint)
            assert isinstance(space.transfer_function, ac.TransferFunction)

    def test_white_point_registry_is_self_consistent(self):
        for name, white_point in ac.ColorSystem.supported_white_points.items():
            assert name == white_point.name
            assert isinstance(white_point, ac.WhitePoint)
            assert 0 < white_point.x < 1
            assert 0 < white_point.y < 1

    def test_white_point_normalized_xyz(self):
        white_point = ac.ColorSystem.supported_white_points['Illuminant E']
        np.testing.assert_allclose(white_point.normalized_xyz(), np.ones(3))

    def test_custom_white_point_can_be_used_directly(self):
        custom = ac.WhitePoint.from_xy('Custom D65', 0.31272, 0.32903)
        system = ac.ColorSystem('sRGB', custom)

        assert system.adaptation_white_point is custom
        assert system.white_point_name == 'Custom D65'
        # The custom coordinates match the native sRGB white point,
        # so no chromatic adaptation should be applied.
        assert system == ac.ColorSystem('sRGB')

    def test_custom_white_point_can_be_derived_from_a_spectrum(self):
        custom = ac.WhitePoint.from_spectrum('Sun-like', ac.sun_tsis1_hsrs)
        standard = ac.ColorSystem.supported_white_points['Sun']

        assert custom.same_chromaticity(standard)
        assert custom.x == pytest.approx(standard.x)
        assert custom.y == pytest.approx(standard.y)
        assert ac.ColorSystem('sRGB', custom).effective_white_point is custom

    def test_custom_color_space_can_be_used_directly(self):
        custom_white = ac.WhitePoint.from_xy('Custom E', 1./3., 1./3.)
        custom_space = ac.ColorSpace(
            name='Custom sRGB/E',
            primaries=ac.ColorSystem.supported_color_spaces['sRGB'].primaries,
            white_point=custom_white,
            transfer_function=ac.transfer_functions.srgb,
        )

        system = ac.ColorSystem(custom_space)

        assert system.color_space is custom_space
        assert system.internal_white_point is custom_white
        rgb_white, _ = system.rgb_to_xyz(np.ones(3))
        np.testing.assert_allclose(rgb_white, custom_white.normalized_xyz())

    def test_custom_white_point_can_drive_chromatic_adaptation(self):
        target = ac.WhitePoint.from_xy('Custom target', 0.4, 0.4)
        base = ac.ColorSystem('sRGB')
        adapted = ac.ColorSystem('sRGB', target)

        assert adapted.effective_white_point is target
        assert not np.allclose(adapted.matrix, base.matrix)
        np.testing.assert_allclose(adapted.matrix @ adapted.inv_matrix, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(adapted.inv_matrix @ adapted.matrix, np.eye(3), atol=1e-6)

    def test_unknown_color_standard_names_are_reported(self):
        with pytest.raises(ValueError, match='Unknown color space'):
            _ = ac.ColorSystem('No such space')
        with pytest.raises(ValueError, match='Unknown white point'):
            _ = ac.ColorSystem('sRGB', 'No such white point')

    def test_chromaticity_z(self):
        chromaticity = ac.Chromaticity(0.64, 0.33)
        assert chromaticity.z == pytest.approx(0.03)

    def test_sun_white_point_matches_its_spectrum(self):
        white_point = ac.ColorSystem.supported_white_points['Sun']
        from_spectrum = ac.WhitePoint.from_spectrum('Sun test', ac.sun_tsis1_hsrs)
        assert white_point.same_chromaticity(from_spectrum)
        assert white_point.x == pytest.approx(from_spectrum.x)
        assert white_point.y == pytest.approx(from_spectrum.y)


class TestColorSystemTransferFunctions:
    """ Color-space-specific transfer functions and chromatic adaptation. """

    def test_adaptation_matrices_are_inverse(self):
        for color_space in ac.ColorSystem.supported_color_spaces:
            system = ac.ColorSystem(color_space, 'Illuminant E')
            identity = np.eye(3)
            np.testing.assert_allclose(system.matrix @ system.inv_matrix, identity, atol=1e-6)
            np.testing.assert_allclose(system.inv_matrix @ system.matrix, identity, atol=1e-6)

    def test_adapted_color_system_roundtrip(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        rgb = np.array([0.2, 0.3, 0.4])
        xyz, _ = system.rgb_to_xyz(rgb)
        restored, _ = system.xyz_to_rgb(xyz)
        np.testing.assert_allclose(restored, rgb, atol=1e-6)

    def test_transfer_functions_live_in_their_own_namespace(self):
        for name in ('linear', 'srgb', 'gamma_22', 'prophoto', 'rec709', 'rec2020'):
            assert isinstance(getattr(ac.transfer_functions, name), ac.TransferFunction)
        assert ac.transfer_functions.srgb is ac.ColorSystem('sRGB').transfer_function
        assert not hasattr(ac, 'srgb_transfer_function')

    def test_each_color_space_uses_its_own_transfer_function(self):
        values = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, -0.2])
        for color_space in ac.ColorSystem.supported_color_spaces:
            system = ac.ColorSystem(color_space)
            encoded = system.encode(values)
            decoded = system.decode(encoded)
            np.testing.assert_allclose(decoded, values, rtol=1e-12, atol=1e-12)

    def test_srgb_transfer_function_known_value(self):
        assert ac.default_color_system.encode(0.5) == pytest.approx(0.7353569830524495)
        assert ac.default_color_system.decode(0.5) == pytest.approx(0.21404114048223255)

    def test_adobe_rgb_uses_gamma_2_2(self):
        adobe = ac.ColorSystem('Adobe RGB')
        np.testing.assert_allclose(adobe.encode(0.5), math.pow(0.5, 1./2.2), rtol=1e-15)

    def test_piecewise_curves_keep_their_standard_constants(self):
        value = 0.5
        assert ac.transfer_functions.prophoto.encode(value) == pytest.approx(value**(1./1.8))
        assert ac.transfer_functions.rec709.encode(value) == pytest.approx(
            1.099 * value**0.45 - 0.099
        )
        alpha = 1.09929682680944
        assert ac.transfer_functions.rec2020.encode(value) == pytest.approx(
            alpha * value**0.45 - (alpha - 1.)
        )

    def test_linear_spaces_use_identity_transfer_function(self):
        values = np.array([0.0, 0.25, 1.0])
        np.testing.assert_allclose(ac.xyz_color_system.encode(values), values)
        np.testing.assert_allclose(ac.xyz_color_system.decode(values), values)


class TestColorSystemEquality:
    """ Structural equality and no-op conversions. """

    def test_equal_definitions_compare_equal(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        same_system = ac.ColorSystem('sRGB', 'Illuminant E')
        assert system == same_system
        assert hash(system) == hash(same_system)
        assert ac.ColorSystem('sRGB') != ac.ColorSystem('sRGB', 'Illuminant E')

    def test_conversion_to_equal_system_skips_matrix_work(self):
        source = _NoMatrixColorSystem('sRGB', 'Illuminant E')
        target = _NoMatrixColorSystem('sRGB', 'Illuminant E')
        color = ac.ColorPoint(np.array([0.2, 0.3, 0.4]), color_system=source)

        converted = color.to_color_system(target)

        np.testing.assert_allclose(converted.spectral_dist, color.spectral_dist)
        assert converted.color_system is target


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
        minimum = cast(float, raw[:, 1].min())
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
        raw_max = cast(float, raw.max())
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
