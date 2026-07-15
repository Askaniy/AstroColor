import numpy as np
import astrocolor as ac


# === Color System Tests ===

class TestColorSystem():

    def test_color_system_reversibility(self):
        """ Converting a color point between systems should be reversible to machine precision. """
        srgb = ac.ColorSystem('sRGB')
        xyz = ac.ColorSystem('CIE 1931 XYZ')
        color0 = ac.ColorPoint.from_spectral_data(ac.sun_CALSPEC)  # in XYZ
        color1 = color0.to_color_system(srgb)
        color2 = color1.to_color_system(xyz)
        np.testing.assert_allclose(color0.to_array(), color2.to_array(), rtol=1e-13)

    def test_adaptation_white_point(self):
        """ CIE 1931 RGB should have Illuminant E white point """
        rgb = ac.ColorSystem('CIE 1931 RGB')
        rgb_ = ac.ColorSystem('CIE 1931 RGB', adaptation_white_point='Illuminant E')
        color0 = ac.ColorPoint.from_spectral_data(ac.sun_CALSPEC)  # in XYZ
        color1 = color0.to_color_system(rgb)
        color2 = color1.to_color_system(rgb_)
        np.testing.assert_allclose(color1.to_array(), color2.to_array(), rtol=1e-13)
