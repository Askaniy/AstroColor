import math
from typing import override

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac


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
        assert ac.transfer_functions.prophoto.encode(value) == pytest.approx(math.pow(value, 1./1.8))
        assert ac.transfer_functions.rec709.encode(value) == pytest.approx(
            1.099 * math.pow(value, 0.45) - 0.099
        )
        alpha = 1.09929682680944
        assert ac.transfer_functions.rec2020.encode(value) == pytest.approx(
            alpha * math.pow(value, 0.45) - (alpha - 1.)
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
