import numpy as np
import astrocolor as ac


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
