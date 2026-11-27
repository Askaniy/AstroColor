import math
import warnings
from typing import cast, override

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.errors import (
    InvalidColorChannelsError,
    MissingDerivativeWarning,
    NegativeStandardDeviationError,
    NonFiniteScaleWarning,
    NonFiniteValuesWarning,
)

from .color_oracles import srgb_decode_derivative, srgb_encode_derivative


class _SquareTransferFunction(ac.transfer_functions.TransferFunction):
    """ Custom curve without an analytic derivative. """

    @override
    def encode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return cast(
            npt.NDArray[np.floating],
            np.asarray(values, dtype=np.float64) ** 2,
        )

    @override
    def decode(self, values: npt.ArrayLike) -> npt.NDArray[np.floating]:
        return cast(
            npt.NDArray[np.floating],
            np.sqrt(np.asarray(values, dtype=np.float64)),
        )


class TestColorBasics:
    """ Basic color construction and color-system conversion behavior. """

    def test_invalid_color_channels(self):
        for color_type in (ac.ColorPoint, ac.ColorLine, ac.ColorImage):
            with pytest.raises(InvalidColorChannelsError, match='three channels'):
                _ = color_type([1., 2.])

    def test_invalid_color_channels_message_uses_name(self):
        with pytest.raises(InvalidColorChannelsError) as exc_info:
            _ = ac.ColorPoint([1., 2.], name='broken color')

        expected = 'Color data must have exactly three channels along the first axis, not shape (2,) for "broken color".'
        assert str(exc_info.value) == expected

        with pytest.raises(InvalidColorChannelsError) as exc_info:
            _ = ac.ColorPoint([1., 2.])
        assert '"None"' not in str(exc_info.value)

    def test_color_object_stores_name(self):
        for color_type in (ac.ColorPoint, ac.ColorLine, ac.ColorImage):
            color = color_type([1., 2., 3.], name='test color')
            assert color.name == 'test color'

    def test_uncertainty_error_uses_color_object_name(self):
        std = np.array([0.1, -0.1, 0.2])
        with pytest.raises(NegativeStandardDeviationError) as exc_info:
            _ = ac.ColorPoint(np.ones(3), std, name='broken color')

        expected = 'Standard deviations contain negative values for "broken color".'
        assert str(exc_info.value) == expected

    def test_color_of_flat_spectrum(self):
        nm = np.arange(200, 800, 5)
        spectrum = ac.Spectrum(nm, np.ones(nm.size), name='Flat spectrum test')
        color = ac.ColorPoint.from_spectral_data(spectrum)
        assert color.name == 'Flat spectrum test'
        target_rgb = np.ones(3)
        np.testing.assert_allclose(color.spectral_dist, target_rgb)
        color_system = ac.ColorSystem('sRGB', 'Illuminant E')
        color = color.to_color_system(color_system)
        assert color.name == 'Flat spectrum test'
        np.testing.assert_allclose(color.spectral_dist, target_rgb)

    def test_color_system_reversibility(self, sun_tsis1_hsrs: ac.Spectrum):
        """ Converting a color point between systems should be reversible to machine precision. """
        srgb = ac.ColorSystem('sRGB')
        xyz = ac.ColorSystem('CIE 1931 XYZ')
        color0 = ac.ColorPoint.from_spectral_data(sun_tsis1_hsrs)  # in XYZ
        color1 = color0.to_color_system(srgb)
        color2 = color1.to_color_system(xyz)
        np.testing.assert_allclose(color0.spectral_dist, color2.spectral_dist, rtol=1e-13)

    def test_adaptation_white_point(self, sun_tsis1_hsrs: ac.Spectrum):
        """ CIE 1931 RGB should have Illuminant E white point """
        rgb = ac.ColorSystem('CIE 1931 RGB')
        rgb_ = ac.ColorSystem('CIE 1931 RGB', adaptation_white_point='Illuminant E')
        color0 = ac.ColorPoint.from_spectral_data(sun_tsis1_hsrs)  # in XYZ
        color1 = color0.to_color_system(rgb)
        color2 = color1.to_color_system(rgb_)
        np.testing.assert_allclose(color1.spectral_dist, color2.spectral_dist, rtol=1e-13)


class TestNonFiniteValues:
    """ Non-finite color values are replaced with zeros and reported. """

    @pytest.mark.parametrize(
        ('color_type', 'values'),
        [
            (ac.ColorPoint, np.array([1., np.nan, 3.])),
            (
                ac.ColorLine,
                np.array([[1., 2.], [np.nan, np.inf], [3., 4.]]),
            ),
            (
                ac.ColorImage,
                np.array([
                    [[1., 2.], [3., 4.]],
                    [[np.nan, 2.], [3., 4.]],
                    [[1., 2.], [3., np.inf]],
                ]),
            ),
        ],
        ids=['point', 'line', 'image'],
    )
    def test_non_finite_values_are_replaced_with_warning(
        self,
        color_type: type[ac.ColorPoint] | type[ac.ColorLine] | type[ac.ColorImage],
        values: npt.NDArray[np.floating],
    ) -> None:
        original = values.copy()
        expected = np.where(np.isfinite(values), values, 0.)
        std = np.full(values.shape, 0.1)

        with pytest.warns(NonFiniteValuesWarning, match='spectral_dist') as record:
            color = color_type(values, std, name='broken color')

        assert len(record) == 1
        assert 'broken color' in str(record[0].message)
        assert np.all(np.isfinite(color.spectral_dist))
        np.testing.assert_array_equal(color.spectral_dist, expected)
        # Sanitization must not rewrite the caller's array, and the reported
        # uncertainty is kept independently of the replaced values.
        np.testing.assert_array_equal(values, original)
        standard_deviation = color.standard_deviation
        assert standard_deviation is not None
        np.testing.assert_allclose(standard_deviation, std)

    def test_gamma_corrected_non_finite_input_is_replaced(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        std = np.array([0.01, 0.02, 0.03])
        encoded = np.array([0.5, np.nan, 0.1])

        with pytest.warns(NonFiniteValuesWarning) as record:
            color = ac.ColorPoint(
                encoded, std, color_system=system, gamma_corrected=True
            )

        assert len(record) == 1
        sanitized = np.array([0.5, 0., 0.1])
        np.testing.assert_allclose(color.spectral_dist, system.decode(sanitized))
        standard_deviation = color.standard_deviation
        assert standard_deviation is not None
        np.testing.assert_allclose(
            standard_deviation,
            srgb_decode_derivative(sanitized) * std,
            rtol=1e-12,
        )

    def test_render_replaces_non_finite_stored_values(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        color = ac.ColorPoint(
            [0.5, 0.25, 0.1], color_system=system, name='broken color'
        )
        # Direct mutation bypasses the constructor validation.
        color.spectral_dist[1] = np.nan
        settings = ac.DisplaySettings(
            color_system=system, maximize_brightness=False
        )

        with pytest.warns(NonFiniteValuesWarning, match='broken color') as record:
            rendered, uncertainty = color.render(settings)

        assert len(record) == 1
        assert uncertainty is None
        assert np.all(np.isfinite(rendered))
        np.testing.assert_allclose(
            rendered, system.encode(np.array([0.5, 0., 0.1]))
        )


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
        rendered, _ = converted.render(
            ac.DisplaySettings(color_system=target, maximize_brightness=False)
        )
        np.testing.assert_allclose(rendered, target.encode(expected))
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

    def test_gamma_corrected_input_propagates_standard_deviation(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([0.5, 0.25, 0.1])
        std = np.array([0.01, 0.02, 0.03])

        color = ac.ColorPoint(encoded, std, color_system=system, gamma_corrected=True)

        np.testing.assert_allclose(color.spectral_dist, system.decode(encoded))
        standard_deviation = color.standard_deviation
        assert standard_deviation is not None
        np.testing.assert_allclose(
            standard_deviation,
            srgb_decode_derivative(encoded) * std,
            rtol=1e-12,
        )

    def test_gamma_corrected_input_propagates_full_covariance(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([0.5, 0.25, 0.1])
        covariance = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03],
        ])

        color = ac.ColorPoint(
            encoded, covariance, color_system=system, gamma_corrected=True
        )

        actual = color.covariance_matrix
        assert actual is not None
        derivative = srgb_decode_derivative(encoded)
        expected = covariance * derivative[:, np.newaxis] * derivative[np.newaxis, :]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(actual, actual.T, rtol=1e-15)
        assert np.all(np.diag(actual) >= 0.)

    def test_gamma_corrected_line_keeps_compact_uncertainty(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([
            [0.5, 0.25],
            [0.25, 0.5],
            [0.1, 0.2],
        ])
        std = np.array([
            [0.01, 0.02],
            [0.02, 0.01],
            [0.03, 0.04],
        ])

        line = ac.ColorLine(encoded, std, color_system=system, gamma_corrected=True)

        assert line.uncertainty is not None
        assert not line.uncertainty.is_covariance
        standard_deviation = line.standard_deviation
        assert standard_deviation is not None
        np.testing.assert_allclose(
            standard_deviation,
            srgb_decode_derivative(encoded) * std,
            rtol=1e-12,
        )

    def test_gamma_corrected_line_propagates_covariance_per_pixel(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([
            [0.5, 0.25],
            [0.25, 0.5],
            [0.1, 0.2],
        ])
        base = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03],
        ])
        covariance = base[:, :, np.newaxis] * np.array([1., 2.])

        line = ac.ColorLine(
            encoded, covariance, color_system=system, gamma_corrected=True
        )

        actual = line.covariance_matrix
        assert actual is not None
        assert actual.shape == (3, 3, 2)
        derivative = srgb_decode_derivative(encoded)
        expected = (
            covariance
            * derivative[:, np.newaxis, :]
            * derivative[np.newaxis, :, :]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(actual, np.swapaxes(actual, 0, 1), rtol=1e-15)
        assert np.all(np.diagonal(actual, axis1=0, axis2=1) >= 0.)

    def test_gamma_corrected_line_broadcasts_shared_covariance(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([
            [0.5, 0.25],
            [0.25, 0.5],
            [0.1, 0.2],
        ])
        base = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03],
        ])

        line = ac.ColorLine(
            encoded, base, color_system=system, gamma_corrected=True
        )

        actual = line.covariance_matrix
        assert actual is not None
        assert actual.shape == (3, 3, 2)
        derivative = srgb_decode_derivative(encoded)
        expected = (
            base[:, :, np.newaxis]
            * derivative[:, np.newaxis, :]
            * derivative[np.newaxis, :, :]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)

    def test_gamma_corrected_line_keeps_shared_covariance_when_derivative_is_uniform(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([
            [0.5, 0.5],
            [0.25, 0.25],
            [0.1, 0.1],
        ])
        base = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03],
        ])

        line = ac.ColorLine(
            encoded, base, color_system=system, gamma_corrected=True
        )

        assert line.uncertainty is not None
        assert line.uncertainty.is_covariance
        assert line.uncertainty.covariance_input().shape == (3, 3)
        actual = line.covariance_matrix
        assert actual is not None
        derivative = srgb_decode_derivative(encoded[:, 0])
        expected_shared = base * derivative[:, np.newaxis] * derivative[np.newaxis, :]
        expected = np.broadcast_to(expected_shared[:, :, np.newaxis], (3, 3, 2))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)

    def test_gamma_corrected_input_ignores_correlations_when_disabled(self):
        system = ac.ColorSystem('sRGB', 'Illuminant E')
        encoded = np.array([0.5, 0.25, 0.1])
        covariance = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03],
        ])

        with ac.Config.override(ignore_correlations=True):
            color = ac.ColorPoint(
                encoded, covariance, color_system=system, gamma_corrected=True
            )

        assert color.uncertainty is not None
        assert not color.uncertainty.is_covariance
        standard_deviation = color.standard_deviation
        assert standard_deviation is not None
        np.testing.assert_allclose(
            standard_deviation,
            srgb_decode_derivative(encoded) * np.sqrt(np.diag(covariance)),
            rtol=1e-12,
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

    def test_render_default_path(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.ColorPoint.from_spectral_data(sun_tsis1_hsrs)
        raw = color.spectral_dist.copy()
        rendered, uncertainty = color.render()
        assert uncertainty is None
        np.testing.assert_allclose(
            rendered,
            [0.92652538, 0.99430624, 1.0],
            rtol=1e-7,
        )
        assert color.to_html() == '#ecfeff'
        np.testing.assert_allclose(color.spectral_dist, raw)

    def test_render_linear_settings(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.get_color(sun_tsis1_hsrs)
        rendered, _ = color.render(ac.LINEAR_SETTINGS)
        np.testing.assert_allclose(rendered, color.spectral_dist)

    def test_render_custom_color_system(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.ColorPoint.from_spectral_data(sun_tsis1_hsrs)
        rendered, _ = color.render(ac.DisplaySettings(color_system=ac.xyz_color_system))
        raw = color.spectral_dist
        raw_max = float(raw.max())
        np.testing.assert_allclose(rendered, raw / raw_max)

    def test_render_scale_factor_and_no_maximize(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.get_color(sun_tsis1_hsrs)
        settings = ac.DisplaySettings(maximize_brightness=False, scale_factor=0.5)
        rendered, _ = color.render(settings)
        np.testing.assert_allclose(
            rendered,
            ac.default_color_system.encode(color.spectral_dist * 0.5),
        )

    def test_render_uses_target_color_space_transfer_function(self):
        adobe = ac.ColorSystem('Adobe RGB')
        color = ac.ColorPoint(np.full(3, 0.5), color_system=adobe)
        rendered, _ = color.render(
            ac.DisplaySettings(color_system=adobe, maximize_brightness=False)
        )
        np.testing.assert_allclose(rendered, math.pow(0.5, 1./2.2), rtol=1e-15)

    def test_render_does_not_modify_object(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.get_color(sun_tsis1_hsrs)
        before = color.spectral_dist.copy()
        _ = color.render()
        _ = color.to_html()
        np.testing.assert_allclose(color.spectral_dist, before)

    def test_render_image(self):
        image = ac.ColorImage(
            np.ones((3, 2, 2)) * 0.5,
            color_system=ac.default_color_system,
        )
        rendered, _ = image.render(ac.DisplaySettings(maximize_brightness=False))
        expected = ac.default_color_system.encode(np.ones((3, 2, 2)) * 0.5)
        np.testing.assert_allclose(rendered, expected)
        assert rendered.shape == (3, 2, 2)

    def test_render_propagates_standard_deviation_through_transfer_function(self):
        system = ac.ColorSystem('sRGB')
        linear = np.array([0.5, 0.21404114, 0.03310477])
        std = np.array([0.05, 0.05, 0.05])
        color = ac.ColorPoint(linear, std, color_system=system)
        settings = ac.DisplaySettings(color_system=system, maximize_brightness=False)

        rendered, uncertainty = color.render(settings)

        assert uncertainty is not None
        assert not uncertainty.is_covariance
        np.testing.assert_allclose(rendered, system.encode(linear))
        np.testing.assert_allclose(
            uncertainty.standard_deviation,
            srgb_encode_derivative(linear) * std,
            rtol=1e-12,
        )

    def test_render_propagates_full_covariance_through_transfer_function(self):
        system = ac.ColorSystem('sRGB')
        linear = np.array([0.5, 0.21404114, 0.03310477])
        covariance = np.array([
            [0.0025, 0.0004, 0.0001],
            [0.0004, 0.0025, 0.0002],
            [0.0001, 0.0002, 0.0025],
        ])
        color = ac.ColorPoint(linear, covariance, color_system=system)
        settings = ac.DisplaySettings(color_system=system, maximize_brightness=False)

        _, uncertainty = color.render(settings)

        assert uncertainty is not None
        derivative = srgb_encode_derivative(linear)
        expected = covariance * derivative[:, np.newaxis] * derivative[np.newaxis, :]
        np.testing.assert_allclose(
            uncertainty.covariance_matrix, expected, rtol=1e-12, atol=0.
        )
        np.testing.assert_allclose(
            uncertainty.covariance_matrix,
            uncertainty.covariance_matrix.T,
            rtol=1e-15,
        )

    def test_render_propagates_uncertainty_through_brightness_normalization(self):
        system = ac.ColorSystem('sRGB')
        linear: npt.NDArray[np.floating] = np.array([0.5, 0.25, 0.1])
        std = np.array([0.01, 0.02, 0.03])
        color = ac.ColorPoint(linear, std, color_system=system)

        rendered, uncertainty = color.render(ac.DisplaySettings(color_system=system))

        assert uncertainty is not None
        maximum = float(linear.max())
        normalized = linear / maximum
        np.testing.assert_allclose(rendered, system.encode(normalized))
        np.testing.assert_allclose(
            uncertainty.standard_deviation,
            srgb_encode_derivative(normalized) * std / maximum,
            rtol=1e-12,
        )

    def test_render_propagates_uncertainty_through_scale_factor(self):
        system = ac.ColorSystem('sRGB')
        linear = np.array([0.5, 0.25, 0.1])
        std = np.array([0.01, 0.02, 0.03])
        color = ac.ColorPoint(linear, std, color_system=system)
        settings = ac.DisplaySettings(
            color_system=system,
            maximize_brightness=False,
            scale_factor=0.5,
        )

        rendered, uncertainty = color.render(settings)

        assert uncertainty is not None
        scaled = linear * 0.5
        np.testing.assert_allclose(rendered, system.encode(scaled))
        np.testing.assert_allclose(
            uncertainty.standard_deviation,
            srgb_encode_derivative(scaled) * std * 0.5,
            rtol=1e-12,
        )

    @pytest.mark.parametrize('with_uncertainty', [False, True])
    def test_render_warns_on_non_finite_normalization_factor(self, with_uncertainty: bool):
        system = ac.ColorSystem('sRGB')
        tiny = np.finfo(float).smallest_subnormal
        uncertainty = np.full(3, 1e-10) if with_uncertainty else None
        color = ac.ColorPoint(
            np.array([tiny, 0., 0.]),
            uncertainty,
            color_system=system,
            name='tiny color',
        )

        with pytest.warns(NonFiniteScaleWarning, match='tiny color'):
            rendered, rendered_uncertainty = color.render(
                ac.DisplaySettings(color_system=system)
            )

        assert not np.all(np.isfinite(rendered))
        assert rendered_uncertainty is None

    def test_render_drops_uncertainty_when_finite_factor_overflows(self):
        system = ac.default_color_system
        color = ac.ColorPoint(
            np.array([1e-300, 0.5e-300, 0.2e-300]),
            np.full(3, 1e-301),
            color_system=system,
            name='dark color',
        )

        with pytest.warns(NonFiniteScaleWarning, match='dark color'):
            rendered, uncertainty = color.render()

        assert uncertainty is None
        np.testing.assert_allclose(
            rendered,
            ac.default_color_system.encode(np.array([1., 0.5, 0.2])),
            rtol=1e-12,
        )

    def test_render_does_not_warn_when_only_uncertainty_scale_would_overflow(self):
        system = ac.default_color_system
        color = ac.ColorPoint(
            np.array([1e-300, 0.5e-300, 0.2e-300]),
            color_system=system,
            name='dark color',
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error', NonFiniteScaleWarning)
            rendered, uncertainty = color.render()

        assert uncertainty is None
        np.testing.assert_allclose(
            rendered,
            system.encode(np.array([1., 0.5, 0.2])),
            rtol=1e-12,
        )

    def test_render_warns_when_values_overflow_for_finite_factor(self):
        system = ac.default_color_system
        color = ac.ColorPoint(
            np.array([1e200, 2e200, 3e200]),
            color_system=system,
            name='huge color',
        )
        settings = ac.DisplaySettings(
            color_system=system,
            maximize_brightness=False,
            scale_factor=1e150,
        )

        with pytest.warns(NonFiniteScaleWarning, match='huge color'):
            rendered, uncertainty = color.render(settings)

        assert uncertainty is None
        assert not np.all(np.isfinite(rendered))

    def test_render_warns_and_drops_uncertainty_without_derivative(self):
        custom_space = ac.ColorSpace(
            name='Custom square',
            primaries=ac.ColorSystem.supported_color_spaces['sRGB'].primaries,
            white_point=ac.ColorSystem.supported_white_points['Illuminant E'],
            transfer_function=_SquareTransferFunction(),
        )
        system = ac.ColorSystem(custom_space)
        color = ac.ColorPoint(
            np.array([0.2, 0.3, 0.4]),
            np.full(3, 0.01),
            color_system=system,
            name='custom color',
        )

        with pytest.warns(MissingDerivativeWarning, match='custom color'):
            rendered, uncertainty = color.render(
                ac.DisplaySettings(color_system=system, maximize_brightness=False)
            )

        assert uncertainty is None
        np.testing.assert_allclose(rendered, color.spectral_dist**2)

    def test_gamma_corrected_input_drops_uncertainty_without_derivative(self):
        custom_space = ac.ColorSpace(
            name='Custom square',
            primaries=ac.ColorSystem.supported_color_spaces['sRGB'].primaries,
            white_point=ac.ColorSystem.supported_white_points['Illuminant E'],
            transfer_function=_SquareTransferFunction(),
        )
        system = ac.ColorSystem(custom_space)
        encoded = np.array([0.2, 0.3, 0.4])

        with pytest.warns(MissingDerivativeWarning, match='custom color'):
            color = ac.ColorPoint(
                encoded,
                np.full(3, 0.01),
                color_system=system,
                gamma_corrected=True,
                name='custom color',
            )

        assert color.uncertainty is None
        np.testing.assert_allclose(color.spectral_dist, np.sqrt(encoded))

    def test_render_line_keeps_compact_uncertainty(self):
        system = ac.ColorSystem('sRGB')
        linear = np.full((3, 4), 0.5)
        std = np.full((3, 4), 0.01)
        line = ac.ColorLine(linear, std, color_system=system)

        rendered, uncertainty = line.render(
            ac.DisplaySettings(color_system=system, maximize_brightness=False)
        )

        assert uncertainty is not None
        assert not uncertainty.is_covariance
        assert rendered.shape == (3, 4)
        assert uncertainty.standard_deviation.shape == (3, 4)
        np.testing.assert_allclose(
            uncertainty.standard_deviation,
            srgb_encode_derivative(linear) * std,
            rtol=1e-12,
        )

    def test_invalid_scale_factor(self):
        for value in (-1.0, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                _ = ac.DisplaySettings(scale_factor=value)


class TestGetColor:
    """ The one-call shortcut for linear color data. """

    def test_get_color_uses_recommended_system(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.get_color(sun_tsis1_hsrs)
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

    def test_get_color_can_return_xyz(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.get_color(sun_tsis1_hsrs, ac.xyz_color_system)
        assert color.color_system is ac.xyz_color_system

    def test_get_color_preserves_an_equal_custom_system(self, sun_tsis1_hsrs: ac.Spectrum):
        target = ac.ColorSystem('CIE 1931 XYZ', 'Illuminant E')
        color = ac.get_color(sun_tsis1_hsrs, target)
        assert color.color_system is target
        np.testing.assert_allclose(
            color.spectral_dist,
            ac.ColorPoint.from_spectral_data(sun_tsis1_hsrs).spectral_dist,
        )

    def test_grayscale_is_linear_luminance(self, sun_tsis1_hsrs: ac.Spectrum):
        color = ac.get_color(sun_tsis1_hsrs)
        expected = cast(
            float,
            color.to_color_system(ac.xyz_color_system).spectral_dist[1],
        )
        assert color.grayscale() == pytest.approx(expected)
