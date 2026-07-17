import numpy as np
import astrocolor as ac



# === FilterObjects Statistics Tests ===

class TestFilterStatistics():
    """ Tests for mean_nm() and std_of_nm() on filters, spectra, and their sets. """

    # - mean_nm() tests

    def test_mean_nm_spectra(self):
        np.testing.assert_allclose(ac.sun_CALSPEC.mean_nm(), 858.101429, rtol=0.001)
        np.testing.assert_allclose(ac.vega_CALSPEC.mean_nm(), 523.296467, rtol=0.001)

    def test_mean_nm_bessell_filters(self):
        # Reference for mean wavelengths: SVO Filter Profile Service
        u_filter = ac.Filter.get('Generic_Bessell.U')
        b_filter = ac.Filter.get('Generic_Bessell.B')
        v_filter = ac.Filter.get('Generic_Bessell.V')
        r_filter = ac.Filter.get('Generic_Bessell.R')
        np.testing.assert_allclose(u_filter.mean_nm(), 360.507, rtol=0.001)
        np.testing.assert_allclose(b_filter.mean_nm(), 441.308, rtol=0.001)
        np.testing.assert_allclose(v_filter.mean_nm(), 551.210, rtol=0.001)
        np.testing.assert_allclose(r_filter.mean_nm(), 658.592, rtol=0.001)

    def test_mean_nm_bessell_filter_set(self, ubv_filterset):
        # Reference for effective wavelengths: Bessell 2005, Table 1
        np.testing.assert_allclose(
            (ubv_filterset * ac.vega_CALSPEC).mean_nm(), [366.3, 436.1, 544.8], rtol=0.01
        )

    def test_mean_nm_sloan_filters(self):
        # Reference for mean wavelengths: SVO Filter Profile Service
        g_filter = ac.Filter.get('SLOAN_SDSS.g').convert_for_photon_counter()
        r_filter = ac.Filter.get('SLOAN_SDSS.r').convert_for_photon_counter()
        np.testing.assert_allclose(g_filter.mean_nm(), 475.082, rtol=0.001)
        np.testing.assert_allclose(r_filter.mean_nm(), 620.429, rtol=0.001)

    # - std_of_nm() tests

    def test_std_of_nm_spectra(self):
        np.testing.assert_allclose(ac.sun_CALSPEC.std_of_nm(), 468.978657, rtol=0.01)
        np.testing.assert_allclose(ac.vega_CALSPEC.std_of_nm(), 466.044761, rtol=0.01)

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
