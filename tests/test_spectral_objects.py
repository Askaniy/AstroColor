import numpy as np
import astrocolor as ac


# === Spectral Objects Creation & Properties Tests ===

class TestSpectralObjects():

    def test_spectrum_from_nm_float(self):
        spectrum = ac.Spectrum.monochromatic(555.5)
        np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(spectrum.mean_nm(), 555.5, rtol=1e-10)

    def test_spectrum_from_nm_integer(self):
        spectrum = ac.Spectrum.monochromatic(555)
        np.testing.assert_allclose(spectrum.integrate(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(spectrum.mean_nm(), 555, rtol=1e-10)

    def test_filter_edges(self):
        v = ac.Filter.get('Generic_Bessell.V')
        assert v.spectral_dist[0] == 0.
        assert v.spectral_dist[-1] == 0.

    def test_filter_edges_extrapolated(self):
        v = ac.Filter.get('Generic_Bessell.V')
        extrapolated_v = v.determine_at_wavelengths(ac.visible_range)
        assert extrapolated_v.spectral_dist[0] == 0.
        assert extrapolated_v.spectral_dist[-1] == 0.

    def test_filter_system_getitem(self):
        ubv = ac.FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V')
        v = ac.Filter.get('Generic_Bessell.V')
        np.testing.assert_equal(ubv[2].mean_nm(), v.mean_nm())
