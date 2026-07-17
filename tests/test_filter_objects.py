import numpy as np
import astrocolor as ac



# === FilterObjects Statistics Tests ===

class TestFilterStatistics():
    """ Tests for mean_nm() and std_of_nm() on filters, spectra, and their sets. """

    # - mean_nm() tests

    def test_mean_nm_spectra(self):
        np.testing.assert_allclose(ac.sun_CALSPEC.mean_nm(), 857.052056, rtol=0.01)
        np.testing.assert_allclose(ac.vega_CALSPEC.mean_nm(), 510.428463, rtol=0.01)

    def test_mean_nm_filter(self, v_filter):
        # 551.210 in SVO Filter Profile Service — our value is ~551.204
        np.testing.assert_allclose(v_filter.mean_nm(), 551.204273, rtol=0.01)

    def test_mean_nm_filter_set(self, ubv_filterset):
        np.testing.assert_allclose(
            ubv_filterset.mean_nm(), [360.507105, 441.301389, 551.204273], rtol=0.01
        )

    # - std_of_nm() tests

    def test_std_of_nm_spectra(self):
        np.testing.assert_allclose(ac.sun_CALSPEC.std_of_nm(), 468.978657, rtol=0.01)
        np.testing.assert_allclose(ac.vega_CALSPEC.std_of_nm(), 353.430263, rtol=0.01)

    def test_std_of_nm_filter(self, v_filter):
        np.testing.assert_allclose(v_filter.std_of_nm(), 36.354015, rtol=0.01)

    def test_std_of_nm_filter_set(self, ubv_filterset):
        np.testing.assert_allclose(
            ubv_filterset.std_of_nm(), [21.932217, 35.816641, 36.354015], rtol=0.01
        )


# === Filter Unification Operation Tests ===

class TestFilterUnification():

    def test_filter_filter(self):
        actual = ac.Filter.get('Generic_Bessell.B') | ac.Filter.get('Generic_Bessell.V')
        desired = ac.FilterSet.get('Generic_Bessell.B', 'Generic_Bessell.V')
        np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
        np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)

    def test_filterset_filter(self):
        actual = ac.FilterSet.get('Generic_Bessell.B', 'Generic_Bessell.V') | ac.Filter.get('Generic_Bessell.R')
        desired = ac.FilterSet.get('Generic_Bessell.B', 'Generic_Bessell.V', 'Generic_Bessell.R')
        np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
        np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)

    def test_filter_filterset(self):
        actual = ac.Filter.get('Generic_Bessell.B') | ac.FilterSet.get('Generic_Bessell.V', 'Generic_Bessell.R')
        desired = ac.FilterSet.get('Generic_Bessell.B', 'Generic_Bessell.V', 'Generic_Bessell.R')
        np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
        np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)

    def test_filterset_filterset(self):
        actual = ac.FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B') | ac.FilterSet.get('Generic_Bessell.V', 'Generic_Bessell.R')
        desired = ac.FilterSet.get('Generic_Bessell.U', 'Generic_Bessell.B', 'Generic_Bessell.V', 'Generic_Bessell.R')
        np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
        np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)
