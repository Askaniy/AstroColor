import warnings
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.algebra import (
    div_error,
    get_standard_deviation,
    make_diagonal_covariance,
    mul_error,
    mul_value,
)
from astrocolor.config import Config
from astrocolor.errors import ErasingCorrelationsWarning


def reconstruct_for_test(photo: ac.Photospectrum | ac.PhotospectralSet):
    """ Reconstructs a photospectral object without attaching the raw photometry """
    return ac.spectral_reconstruction(
        photo,
        photo.filter_set.wavelength_nm,
        attach_photospectral_obj=False,
    )


# === Arithmetic Operations Tests ===


class TestAddiction:

    def test_spectrum(self):
        np.testing.assert_allclose(
            (ac.vega_calspec + ac.vega_calspec).spectral_dist,
            (ac.vega_calspec * 2).spectral_dist,
            rtol=0.01,
        )

    def test_same_filter_set_photospectra_add_elementwise(self, ubv_filterset: ac.FilterSet):
        photospectrum1 = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        photospectrum2 = ac.Photospectrum(ubv_filterset, [2.0, 1.0, 0.5])
        result = photospectrum1 + photospectrum2
        np.testing.assert_allclose(
            result.spectral_dist,
            photospectrum1.spectral_dist + photospectrum2.spectral_dist,
        )


class TestMultiplication:

    def test_0D_mul_value(self):
        assert mul_value(2, 2) == 4

    def test_1D_mul_value(self):
        np.testing.assert_allclose(mul_value([3, 3], 3), [9, 9])

    def test_0D_mul_error_both_have_errors(self):
        x = 10
        dx = 2
        y = 20
        dy = 5
        expected = (y * dx)**2 + (x * dy)**2
        result = mul_error(x, dx**2, y, dy**2)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_mul_error_only_x_has_error(self):
        x = 10
        dx = 2
        y = 20
        expected = (y * dx)**2
        result = mul_error(x, dx**2, y, None)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_mul_error_only_y_has_error(self):
        x = 10
        y = 20
        dy = 5
        expected = (x * dy)**2
        result = mul_error(x, None, y, dy**2)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_0D_mul_error_no_errors(self):
        result = mul_error(10, None, 20, None)
        assert result is None

    def test_1D_1D_mul_error_no_errors(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = mul_error(x, None, y, None)
        assert result is None

    def test_0D_1D_mul_error_no_errors(self):
        result = mul_error(3, None, np.array([1, 2, 3]), None)
        assert result is None

    def test_scalar_times_array_with_error(self):
        """ Multiplication of scalar by array with error """
        scalar = 3
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])  # diagonal covariance matrix
        result = mul_error(scalar, None, array, cov_array)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_array_times_scalar_with_error(self):
        """ Multiplication of array with error by scalar """
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])
        scalar = 5
        result = mul_error(array, cov_array, scalar, None)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_array_times_scalar_both_have_errors(self):
        """ Multiplication of array and scalar, both with errors """
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])
        scalar = 5
        scalar_var = 0.5
        result = mul_error(array, cov_array, scalar, scalar_var)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array + np.diag(array**2 * scalar_var)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_multiplication_with_correlation(self):
        """ Multiplication of two 1D arrays with correlation """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_x = np.array([[0.1, 0.02, 0.01],
                        [0.02, 0.2, 0.03],
                        [0.01, 0.03, 0.3]])
        cov_y = np.array([[0.4, 0.05, 0.02],
                        [0.05, 0.5, 0.06],
                        [0.02, 0.06, 0.6]])
        result = mul_error(x, cov_x, y, cov_y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        diag_y = np.diag(y)
        diag_x = np.diag(x)
        expected = diag_y @ cov_x @ diag_y + diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_only_x_has_error(self):
        """ Multiplication of arrays, only x has error """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_x = np.diag([0.1, 0.2, 0.3])
        result = mul_error(x, cov_x, y, None)
        assert isinstance(result, np.ndarray)
        diag_y = np.diag(y)
        expected = diag_y @ cov_x @ diag_y
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_only_y_has_error(self):
        """ Multiplication of arrays, only y has error """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_y = np.diag([0.4, 0.5, 0.6])
        result = mul_error(x, None, y, cov_y)
        assert isinstance(result, np.ndarray)
        diag_x = np.diag(x)
        expected = diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_incompatible_sizes_raises_error(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        with pytest.raises(ValueError):
            _ = mul_error(x, np.diag([0.1, 0.2, 0.3]), y, np.diag([0.4, 0.5]))

    def test_vector_covariance_input(self):
        """ Check that variance vector can be passed instead of matrix """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        var_x = np.array([0.1, 0.2, 0.3])
        var_y = np.array([0.4, 0.5, 0.6])
        result = mul_error(x, var_x, y, var_y)
        assert isinstance(result, np.ndarray)
        cov_x = np.diag(var_x)
        cov_y = np.diag(var_y)
        diag_y = np.diag(y)
        diag_x = np.diag(x)
        expected = diag_y @ cov_x @ diag_y + diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_multiplication_filter_spectrum_mean(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            (v_filter * ac.vega_calspec).mean_nm(), 544.601418, rtol=0.01
        )  # 544.543 in SVO Filter Profile Service
        np.testing.assert_allclose(
            (ubv_filterset * ac.vega_calspec).mean_nm(),
            [366.764603, 435.741381, 544.601418],
            rtol=0.01,
        )

    def test_multiplication_observation(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_calspec * 2, v_filter)[0],
            ac.get_photometry(ac.vega_calspec, v_filter)[0] * 2,
            rtol=0.01,
        )
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_calspec * 2, ubv_filterset).spectral_dist,
            ac.get_photometry(ac.vega_calspec, ubv_filterset * 2).spectral_dist,
            rtol=0.01,
        )

    def test_photospectrum_times_spectrum_reconstructs(self, ubv_filterset: ac.FilterSet):
        """ Multiplication is performed on reconstructed spectra, not on photometric integrals """
        nm = ubv_filterset.wavelength_nm
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], name='test photospectrum')
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3, name='test spectrum')
        result = photospectrum * spectrum
        expected = ac.get_photometry(
            reconstruct_for_test(photospectrum) * spectrum,
            ubv_filterset,
        )
        np.testing.assert_allclose(result.spectral_dist, expected.spectral_dist)
        old_result = photospectrum.spectral_dist * ac.get_photometry(spectrum, ubv_filterset).spectral_dist
        assert not np.allclose(result.spectral_dist, old_result)

    def test_photospectrum_times_photospectrum_reconstructs(self, ubv_filterset: ac.FilterSet):
        """ Same-filter photospectra are multiplied through their reconstructed spectra """
        photospectrum1 = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        photospectrum2 = ac.Photospectrum(ubv_filterset, [2.0, 1.0, 0.5])
        result = photospectrum1 * photospectrum2
        expected = ac.get_photometry(
            reconstruct_for_test(photospectrum1) * reconstruct_for_test(photospectrum2),
            ubv_filterset,
        )
        np.testing.assert_allclose(result.spectral_dist, expected.spectral_dist)
        old_result = photospectrum1.spectral_dist * photospectrum2.spectral_dist
        assert not np.allclose(result.spectral_dist, old_result)

    def test_photospectrum_times_photospectrum_different_filter_sets(self):
        """ The result uses the filter set of the left operand and stays finite """
        ub_filter_set = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B')
        bv_filter_set = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V')
        photospectrum1 = ac.Photospectrum(ub_filter_set, [0.5, 1.0])
        photospectrum2 = ac.Photospectrum(bv_filter_set, [1.0, 2.0])
        result = photospectrum1 * photospectrum2
        assert isinstance(result, ac.Photospectrum)
        assert result.filter_set == ub_filter_set
        assert result.spectral_dist.shape == photospectrum1.spectral_dist.shape
        assert np.all(np.isfinite(result.spectral_dist))

    def test_photospectral_set_times_spectrum_reconstructs(self, ubv_filterset: ac.FilterSet):
        """ Spatial dimensions are broadcast correctly """
        nm = ubv_filterset.wavelength_nm
        photospectral_set = ac.PhotospectralSet(
            ubv_filterset,
            [[0.5, 1.0], [1.0, 1.5], [2.0, 0.5]],
        )
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3)
        result = photospectral_set * spectrum
        expected = ac.get_photometry(
            reconstruct_for_test(photospectral_set) * spectrum,
            ubv_filterset,
        )
        assert isinstance(result, ac.PhotospectralSet)
        assert result.spectral_dist.shape == photospectral_set.spectral_dist.shape
        np.testing.assert_allclose(result.spectral_dist, expected.spectral_dist)

    def test_photospectrum_times_scalar_is_unchanged(self, ubv_filterset: ac.FilterSet):
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            (photospectrum * 2).spectral_dist,
            photospectrum.spectral_dist * 2,
        )

    def test_photospectrum_times_spectrum_keeps_1d_covariance(self, ubv_filterset: ac.FilterSet):
        """ 1D uncertainty is propagated through the reconstructed spectra """
        nm = ubv_filterset.wavelength_nm
        photospectrum = ac.Photospectrum(
            ubv_filterset,
            [0.5, 1.0, 2.0],
            uncertainty=[0.05, 0.1, 0.2],
        )
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ErasingCorrelationsWarning)
            result = photospectrum * spectrum
        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (len(ubv_filterset), len(ubv_filterset))


class TestDivision:

    def test_zero_division_error(self, v_filter: ac.Filter):
        np.testing.assert_equal((v_filter / 0).spectral_dist, np.full_like(v_filter.spectral_dist, np.inf))

    def test_zero_division_by_zero_error(self):
        np.testing.assert_equal((ac.SpectralSet.stub() / 0).spectral_dist, [[0.]])

    def test_photospectrum_divided_by_spectrum_reconstructs(self, ubv_filterset: ac.FilterSet):
        """ Division is also performed on reconstructed spectra """
        nm = ubv_filterset.wavelength_nm
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3)
        result = photospectrum / spectrum
        expected = ac.get_photometry(
            reconstruct_for_test(photospectrum) / spectrum,
            ubv_filterset,
        )
        np.testing.assert_allclose(result.spectral_dist, expected.spectral_dist)

    def test_filter_spectrum_mean(self, v_filter: ac.Filter, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            (v_filter / ac.vega_calspec).mean_nm(), 558.681024, rtol=0.001
        )
        np.testing.assert_allclose(
            (ubv_filterset / ac.vega_calspec).mean_nm(),
            [356.283866, 447.589411, 558.681024],
            rtol=0.01,
        )

    def test_spectrum_wavelength(self, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            (ac.sun_tsis1_hsrs / ac.sun_tsis1_hsrs.wavelength_nm).mean_nm(), 687.336338, rtol=0.001
        )
        np.testing.assert_allclose(
            (ubv_filterset / ubv_filterset.wavelength_nm).mean_nm(),
            [359.158258, 438.480057, 548.890305],
            rtol=0.01,
        )


class TestNormalization:

    def test_filter(self, v_filter: ac.Filter):
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_calspec, (v_filter * 2).normalized())[0],
            ac.get_photometry(ac.vega_calspec, v_filter)[0],
            rtol=0.01,
        )

    def test_filter_set(self, ubv_filterset: ac.FilterSet):
        np.testing.assert_allclose(
            ac.get_photometry(ac.vega_calspec, (ubv_filterset * 2).normalized()).spectral_dist,
            ac.get_photometry(ac.vega_calspec, ubv_filterset).spectral_dist,
            rtol=0.01,
        )


class TestCovariance:

    def test_spectral_set_std_array_roundtrip(self):
        nm = np.arange(500, 516, 5)
        std = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        obj = ac.SpectralSet(nm, np.ones((4, 2)), std)
        assert obj.covariance_matrix is not None
        assert obj.covariance_matrix.shape == (4, 4, 2)
        assert isinstance(obj.standard_deviation, np.ndarray)
        np.testing.assert_allclose(obj.standard_deviation, std)

    def test_photospectral_set_std_array_roundtrip(self, ubv_filterset: ac.FilterSet):
        std = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        obj = ac.PhotospectralSet(ubv_filterset, np.ones((3, 2)), std)
        assert obj.covariance_matrix is not None
        assert obj.covariance_matrix.shape == (3, 3, 2)
        assert isinstance(obj.standard_deviation, np.ndarray)
        np.testing.assert_allclose(obj.standard_deviation, std)

    def test_unsorted_covariance_is_reordered_on_both_axes(self):
        nm0 = np.arange(400, 431, 5)
        order = np.array([3, 1, 5, 0, 6, 2, 4])
        cov = np.arange(49.).reshape(7, 7) + np.eye(7)
        obj = ac.Spectrum(nm0[order], np.arange(1., 8.), cov)
        assert isinstance(obj.covariance_matrix, np.ndarray)
        np.testing.assert_allclose(
            obj.covariance_matrix,
            cov[np.ix_(np.argsort(nm0[order]), np.argsort(nm0[order]))],
        )

    def test_determine_at_trusted_wavelengths_keeps_covariance_shape(self):
        nm = np.arange(500, 601, 5)
        std = np.ones((nm.size, 2)) * 0.1
        obj = ac.SpectralSet(nm, np.ones((nm.size, 2)), std)
        output = obj.determine_at_trusted_wavelengths(np.arange(490, 611, 5))
        assert output.covariance_matrix is not None
        assert output.covariance_matrix.shape == (output.spectral_size, output.spectral_size, 2)
        assert output.standard_deviation is not None
        np.testing.assert_allclose(output.standard_deviation[1:-1], np.ones((output.spectral_size - 2, 2)) * 0.1)

    def test_cube_covariance_is_extrapolated_along_spatial_axes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Config, 'ignore_uncertainty_for_cubes', False)
        nm = np.arange(500, 601, 5)
        std = np.ones((nm.size, 2, 2)) * 0.1
        obj = ac.SpectralCube(nm, np.ones((nm.size, 2, 2)), std)
        output = obj.determine_at_trusted_wavelengths(np.arange(490, 611, 5))
        assert output.covariance_matrix is not None
        assert output.covariance_matrix.shape == (output.spectral_size, output.spectral_size, 2, 2)
        assert output.standard_deviation is not None
        assert output.standard_deviation.shape == output.spectral_dist.shape

    def test_mul_and_div_error_support_spatial_axes(self):
        x = np.arange(1., 7.).reshape(3, 2)
        y = np.full((3, 2), 2.)
        std_x = np.full((3, 2), 0.1)
        std_y = np.full((3, 2), 0.2)
        cov_x = make_diagonal_covariance(std_x)
        cov_y = make_diagonal_covariance(std_y)

        product = mul_error(x, cov_x, y, cov_y)
        assert product is not None
        assert product.shape == (3, 3, 2)
        expected_product_diag = (y * std_x)**2 + (x * std_y)**2
        np.testing.assert_allclose(get_standard_deviation(product)**2, expected_product_diag)

        quotient = div_error(x, cov_x, y, cov_y)
        assert quotient is not None
        assert quotient.shape == (3, 3, 2)
        expected_quotient_diag = (std_x / y)**2 + (x * std_y / y**2)**2
        np.testing.assert_allclose(get_standard_deviation(quotient)**2, expected_quotient_diag)

    def test_single_filter_photometry_error_is_scalar_and_convolved(self):
        nm = np.arange(500, 601, 5)
        cov = np.diag(np.full(nm.size, 0.01))
        spectrum = ac.Spectrum(nm, np.arange(1., nm.size + 1.), cov)
        bandpass = ac.Filter.get('Generic/Bessell.V')
        _, error = ac.get_photometry(spectrum, bandpass)
        target = ac.get_spectrometry(spectrum, bandpass.wavelength_nm, strictly=True)
        assert isinstance(target.covariance_matrix, np.ndarray)
        expected = target.nm_step**2 * cast(npt.NDArray[np.floating], np.einsum(
            'i,ij,j->', bandpass.spectral_dist, target.covariance_matrix, bandpass.spectral_dist
        ))
        assert error is not None
        assert np.ndim(error) == 0
        np.testing.assert_allclose(error, expected)

    def test_filter_set_photometry_keeps_spatial_covariance(self, ubv_filterset: ac.FilterSet):
        nm = ubv_filterset.wavelength_nm
        std = np.ones((nm.size, 2)) * 0.1
        spectral_set = ac.SpectralSet(nm, np.ones((nm.size, 2)) * 2, std)
        photospectral_set = ac.get_photometry(spectral_set, ubv_filterset)
        assert isinstance(photospectral_set, ac.PhotospectralSet)
        assert photospectral_set.covariance_matrix is not None
        assert photospectral_set.covariance_matrix.shape == (len(ubv_filterset), len(ubv_filterset), 2)
        assert isinstance(spectral_set.covariance_matrix, np.ndarray)
        expected = cast(npt.NDArray[np.floating], np.einsum(
            'ij,jk...,lk->il...',
            ubv_filterset.matrix,
            spectral_set.covariance_matrix,
            ubv_filterset.matrix,
        ))
        np.testing.assert_allclose(photospectral_set.covariance_matrix, expected)

    def test_same_filter_photometry_is_identity_with_covariance(self, ubv_filterset: ac.FilterSet):
        std = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        photospectral_set = ac.PhotospectralSet(ubv_filterset, np.ones((3, 2)), std)
        output = ac.get_photometry(photospectral_set, ubv_filterset)
        np.testing.assert_allclose(output.spectral_dist, photospectral_set.spectral_dist)
        assert isinstance(output.standard_deviation, np.ndarray)
        np.testing.assert_allclose(output.standard_deviation, std)
