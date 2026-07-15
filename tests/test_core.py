import numpy as np
import astrocolor as ac


# === Spectrum Creation & Properties Tests ===

class TestSpectrumCreation():

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


# === Filter & FilterSet Statistics Tests ===

class TestFilterStatistics():
    """ Tests for mean_nm() and std_of_nm() on filters, spectra, and their sets. """

    # - mean_nm() tests

    def test_mean_nm_spectra(self):
        np.testing.assert_allclose(ac.sun_CALSPEC.mean_nm(), 857.052056, rtol=0.01)
        np.testing.assert_allclose(ac.vega_CALSPEC.mean_nm(), 510.428463, rtol=0.01)

    def test_mean_nm_filter(self, filter_v):
        # 551.210 in SVO Filter Profile Service — our value is ~551.204
        np.testing.assert_allclose(filter_v.mean_nm(), 551.204273, rtol=0.01)

    def test_mean_nm_filter_set(self, filterset_ubv):
        np.testing.assert_allclose(
            filterset_ubv.mean_nm(), [360.507105, 441.301389, 551.204273], rtol=0.01
        )

    # - std_of_nm() tests

    def test_std_of_nm_spectra(self):
        np.testing.assert_allclose(ac.sun_CALSPEC.std_of_nm(), 468.978657, rtol=0.01)
        np.testing.assert_allclose(ac.vega_CALSPEC.std_of_nm(), 353.430263, rtol=0.01)

    def test_std_of_nm_filter(self, filter_v):
        np.testing.assert_allclose(filter_v.std_of_nm(), 36.354015, rtol=0.01)

    def test_std_of_nm_filter_set(self, filterset_ubv):
        np.testing.assert_allclose(
            filterset_ubv.std_of_nm(), [21.932217, 35.816641, 36.354015], rtol=0.01
        )
