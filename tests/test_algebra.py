import warnings
from typing import cast, overload

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.algebra import mul_value
from astrocolor.auxiliary import (
    interpolation_matrix,
    spectral_binning,
    spectral_binning_matrix,
)
from astrocolor.config import Config
from astrocolor.errors import (
    ErasingCorrelationsWarning,
    InconsistentUncertaintyShapeError,
)
from astrocolor.uncertainty import (
    Uncertainty,
    combine_covariance_terms,
    div_uncertainty,
    propagate_covariance,
)


def product_covariance(
    mean_x: npt.ArrayLike,
    cov_x: npt.ArrayLike | None,
    mean_y: npt.ArrayLike,
    cov_y: npt.ArrayLike | None
):
    """ Product-rule wrapper around the generic weighted covariance combiner. """
    return combine_covariance_terms(cov_x, mean_y, cov_y, mean_x)


@overload
def reconstruct(
    obj: ac.Spectrum | ac.Photospectrum,
    target_nm: npt.NDArray[np.integer],
    attach: bool = True
) -> ac.Spectrum:
    ...

@overload
def reconstruct(
    obj: ac.SpectralSet | ac.PhotospectralSet,
    target_nm: npt.NDArray[np.integer],
    attach: bool = True
) -> ac.SpectralSet:
    ...

def reconstruct(
    obj: ac.Spectrum | ac.SpectralSet | ac.Photospectrum | ac.PhotospectralSet,
    target_nm: npt.NDArray[np.integer],
    attach: bool = False
):
    """ Reconstructs an object on the target grid without keeping raw photometry. """
    return ac.get_spectrometry(
        obj,
        target_nm,
        strictly=True,
        attach_photospectral_obj=attach,
    )

def align_spectral_dims(left: ac.Spectrum | ac.SpectralSet, right: ac.Spectrum | ac.SpectralSet):
    """ Aligns the lower-dimensional spectral distribution for broadcasting. """
    left_value = left.spectral_dist
    right_value = right.spectral_dist
    if left.ndim < right.ndim:
        left_value = left_value.reshape(left_value.shape + (1,) * (right.ndim - left.ndim))
    elif right.ndim < left.ndim:
        right_value = right_value.reshape(right_value.shape + (1,) * (left.ndim - right.ndim))
    return left_value, right_value

def synthetic_product(left: ac.Spectrum | ac.SpectralSet, right: ac.Spectrum | ac.SpectralSet, filter_set: ac.FilterSet):
    """ Independent oracle: point-wise product of reconstructed spectra, then photometry. """
    left_value, right_value = align_spectral_dims(left, right)
    return filter_set.matrix @ (left_value * right_value)


# === Arithmetic Operations Tests ===


class TestAddiction:

    def test_spectrum(self):
        np.testing.assert_allclose(
            (ac.vega_calspec + ac.vega_calspec).spectral_dist,
            2 * ac.vega_calspec.spectral_dist,
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

    def test_0D_product_covariance_both_have_errors(self):
        x = 10
        dx = 2
        y = 20
        dy = 5
        expected = (y * dx)**2 + (x * dy)**2
        result = product_covariance(x, dx**2, y, dy**2)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_product_covariance_only_x_has_error(self):
        x = 10
        dx = 2
        y = 20
        expected = (y * dx)**2
        result = product_covariance(x, dx**2, y, None)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_product_covariance_only_y_has_error(self):
        x = 10
        y = 20
        dy = 5
        expected = (x * dy)**2
        result = product_covariance(x, None, y, dy**2)
        assert isinstance(result, (int, float))
        assert np.isclose(result, expected)

    def test_0D_0D_product_covariance_no_errors(self):
        result = product_covariance(10, None, 20, None)
        assert result is None

    def test_1D_1D_product_covariance_no_errors(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = product_covariance(x, None, y, None)
        assert result is None

    def test_0D_1D_product_covariance_no_errors(self):
        result = product_covariance(3, None, np.array([1, 2, 3]), None)
        assert result is None

    def test_scalar_times_array_with_error(self):
        """ Multiplication of scalar by array with error """
        scalar = 3
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])  # diagonal covariance matrix
        result = product_covariance(scalar, None, array, cov_array)
        assert isinstance(result, np.ndarray)
        expected = scalar**2 * cov_array
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, expected)

    def test_array_times_scalar_with_error(self):
        """ Multiplication of array with error by scalar """
        array = np.array([1, 2, 3])
        cov_array = np.diag([0.1, 0.2, 0.3])
        scalar = 5
        result = product_covariance(array, cov_array, scalar, None)
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
        result = product_covariance(array, cov_array, scalar, scalar_var)
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
        result = product_covariance(x, cov_x, y, cov_y)
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
        result = product_covariance(x, cov_x, y, None)
        assert isinstance(result, np.ndarray)
        diag_y = np.diag(y)
        expected = diag_y @ cov_x @ diag_y
        np.testing.assert_allclose(result, expected)

    def test_1d_arrays_only_y_has_error(self):
        """ Multiplication of arrays, only y has error """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cov_y = np.diag([0.4, 0.5, 0.6])
        result = product_covariance(x, None, y, cov_y)
        assert isinstance(result, np.ndarray)
        diag_x = np.diag(x)
        expected = diag_x @ cov_y @ diag_x
        np.testing.assert_allclose(result, expected)

    def test_incompatible_sizes_raises_error(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        with pytest.raises(ValueError):
            _ = product_covariance(x, np.diag([0.1, 0.2, 0.3]), y, np.diag([0.4, 0.5]))

    def test_vector_covariance_input(self):
        """ Check that variance vector can be passed instead of matrix """
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        var_x = np.array([0.1, 0.2, 0.3])
        var_y = np.array([0.4, 0.5, 0.6])
        result = product_covariance(x, var_x, y, var_y)
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


class TestPhotospectralMultiplication:

    def test_photospectrum_times_spectrum_matches_synthetic_product(self, ubv_filterset: ac.FilterSet):
        nm = ubv_filterset.wavelength_nm
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], name='test photospectrum')
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3, name='test spectrum')

        left = reconstruct(photospectrum, nm)
        right = reconstruct(spectrum, nm)
        expected = synthetic_product(left, right, ubv_filterset)

        result = photospectrum * spectrum

        assert isinstance(result, ac.Photospectrum)
        assert result.filter_set == ubv_filterset
        np.testing.assert_allclose(result.spectral_dist, expected)

        measured_product = photospectrum.spectral_dist * ac.get_photometry(spectrum, ubv_filterset).spectral_dist
        assert np.max(np.abs(result.spectral_dist - measured_product)) > 1e-2, (
            'Multiplication must be performed on reconstructed spectra, not on measured photometry'
        )

    def test_photospectrum_times_photospectrum_matches_synthetic_product(self, ubv_filterset: ac.FilterSet):
        nm = ubv_filterset.wavelength_nm
        photospectrum1 = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0], name='photospectrum 1')
        photospectrum2 = ac.Photospectrum(ubv_filterset, [2.0, 1.0, 0.5], name='photospectrum 2')

        left = reconstruct(photospectrum1, nm)
        right = reconstruct(photospectrum2, nm)
        expected = synthetic_product(left, right, ubv_filterset)

        result = photospectrum1 * photospectrum2

        np.testing.assert_allclose(result.spectral_dist, expected)

        measured_product = photospectrum1.spectral_dist * photospectrum2.spectral_dist
        np.testing.assert_allclose(measured_product, np.ones(len(ubv_filterset)))
        assert np.max(np.abs(result.spectral_dist - measured_product)) > 1e-2, (
            'The integral of the product must differ from the product of the integrals'
        )

    def test_photospectrum_times_photospectrum_with_different_filter_sets_uses_left_grid(self):
        ub_filter_set = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B')
        bv_filter_set = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V')
        target_nm = ub_filter_set.wavelength_nm
        photospectrum1 = ac.Photospectrum(ub_filter_set, [0.5, 1.0])
        photospectrum2 = ac.Photospectrum(bv_filter_set, [1.0, 2.0])

        left = reconstruct(photospectrum1, target_nm)
        right = reconstruct(photospectrum2, target_nm)
        expected = synthetic_product(left, right, ub_filter_set)

        result = photospectrum1 * photospectrum2

        assert isinstance(result, ac.Photospectrum)
        assert result.filter_set == ub_filter_set
        np.testing.assert_allclose(result.spectral_dist, expected)

    def test_photospectral_set_times_spectrum_reconstructs(self, ubv_filterset: ac.FilterSet):
        """ Spatial dimensions are broadcast correctly """
        nm = ubv_filterset.wavelength_nm
        photospectral_set = ac.PhotospectralSet(
            ubv_filterset,
            [[0.5, 1.0], [1.0, 1.5], [2.0, 0.5]],
        )
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3)

        left = reconstruct(photospectral_set, nm)
        right = reconstruct(spectrum, nm)
        expected = synthetic_product(left, right, ubv_filterset)

        result = photospectral_set * spectrum

        assert isinstance(result, ac.PhotospectralSet)
        assert result.spectral_dist.shape == photospectral_set.spectral_dist.shape
        np.testing.assert_allclose(result.spectral_dist, expected)

    def test_photospectrum_times_spectral_set_broadcasts_elementwise_product(self, ubv_filterset: ac.FilterSet):
        nm = ubv_filterset.wavelength_nm
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3)
        spectral_set = ac.SpectralSet(
            nm,
            np.column_stack((spectrum.spectral_dist, 2 * spectrum.spectral_dist)),
        )

        left = reconstruct(photospectrum, nm)
        right = reconstruct(spectral_set, nm)
        expected = synthetic_product(left, right, ubv_filterset)

        result = photospectrum * spectral_set

        assert isinstance(result, ac.PhotospectralSet)
        assert result.spectral_dist.shape == (len(ubv_filterset), spectral_set.spatial_size)
        np.testing.assert_allclose(result.spectral_dist, expected)

    def test_forcing_attach_flag_does_not_change_product(
        self,
        ubv_filterset: ac.FilterSet,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ Attachment is metadata only; it must not switch multiplication to raw photometry. """
        nm = ubv_filterset.wavelength_nm
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        spectrum = ac.Spectrum(nm, (nm / cast(int, nm[0]))**3)

        baseline = photospectrum * spectrum
        original_get_spectrometry = ac.measurements.get_spectrometry

        def force_attachment(
            target: ac.Spectrum | ac.Photospectrum,
            requested_wavelengths: npt.ArrayLike,
            strictly: bool = False,
            attach_photospectral_obj: bool = False,  # pyright: ignore[reportUnusedParameter]
        ):
            # Deliberately override the flag used by the production code.
            return original_get_spectrometry(
                target,
                requested_wavelengths,
                strictly=strictly,
                attach_photospectral_obj=True,
            )

        monkeypatch.setattr(ac.measurements, 'get_spectrometry', force_attachment)
        forced = photospectrum * spectrum

        assert forced.filter_set == baseline.filter_set
        np.testing.assert_allclose(forced.spectral_dist, baseline.spectral_dist)

    def test_multiplication_reconstructs_attached_operand_instead_of_extrapolating(
        self,
        ubv_filterset: ac.FilterSet,
    ):
        """ Re-reconstruction on the common grid wins over extrapolating the current spectrum. """
        nm = ubv_filterset.wavelength_nm
        narrow_nm = np.arange(400, 501, 5)
        photospectrum = ac.Photospectrum(ubv_filterset, [0.5, 1.0, 2.0])
        source = ac.Photospectrum(ubv_filterset, [1.0, 2.0, 0.5])
        reconstructed = ac.get_spectrometry(
            source,
            narrow_nm,
            strictly=True,
            attach_photospectral_obj=True,
        )

        left = reconstruct(photospectrum, nm)
        expected_reconstructed = synthetic_product(left, reconstruct(source, nm), ubv_filterset)

        # What using the current (narrow-range) spectrum and extrapolating it would give.
        detached = ac.get_spectrometry(
            source,
            narrow_nm,
            strictly=True,
            attach_photospectral_obj=False,
        )
        extrapolated = cast(ac.Spectrum, ac.get_spectrometry(
            detached,
            nm,
            strictly=True,
            attach_photospectral_obj=False,
        ))
        expected_extrapolated = synthetic_product(left, extrapolated, ubv_filterset)

        result = photospectrum * reconstructed

        np.testing.assert_allclose(result.spectral_dist, expected_reconstructed)
        assert np.max(np.abs(result.spectral_dist - expected_extrapolated)) > 1e-2, (
            'The attached source photometry was not reconstructed on the common target grid'
        )

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
            reconstruct(photospectrum, nm) / spectrum,
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

    def test_product_covariance_and_div_uncertainty_support_spatial_axes(self):
        x = np.arange(1., 7.).reshape(3, 2)
        y = np.full((3, 2), 2.)
        std_x = np.full((3, 2), 0.1)
        std_y = np.full((3, 2), 0.2)
        cov_x = Uncertainty.from_standard_deviation(std_x).covariance_matrix
        cov_y = Uncertainty.from_standard_deviation(std_y).covariance_matrix

        product = product_covariance(x, cov_x, y, cov_y)
        assert isinstance(product, np.ndarray)
        assert product.shape == (3, 3, 2)
        expected_product_diag = (y * std_x)**2 + (x * std_y)**2
        product_uncertainty = Uncertainty.from_covariance(
            product, (product.shape[0], *product.shape[2:])
        )
        np.testing.assert_allclose(product_uncertainty.standard_deviation**2, expected_product_diag)

        quotient = div_uncertainty(
            x, Uncertainty.from_standard_deviation(std_x),
            y, Uncertainty.from_standard_deviation(std_y),
        )
        assert quotient is not None
        expected_quotient_diag = (std_x / y)**2 + (x * std_y / y**2)**2
        np.testing.assert_allclose(quotient.standard_deviation**2, expected_quotient_diag)

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


def correlated_covariance(
    n: int,
    spatial_shape: tuple[int, ...] = (),
    seed: int = 0,
) -> np.ndarray:
    """A deterministic symmetric positive-definite covariance matrix."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n))
    covariance = matrix @ matrix.T / n + np.eye(n) * 0.1
    if spatial_shape:
        covariance = np.broadcast_to(
            covariance.reshape(n, n, *((1,) * len(spatial_shape))),
            (n, n, *spatial_shape),
        ).copy()
    return covariance


class TestCovarianceHandling:

    def test_determine_at_trusted_wavelengths_preserves_correlations_on_same_grid(self):
        nm = np.arange(500, 601, 5)
        covariance = correlated_covariance(nm.size)
        obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        with warnings.catch_warnings():
            warnings.simplefilter('error', ErasingCorrelationsWarning)
            output = obj.determine_at_trusted_wavelengths(nm)

        output_covariance = output.covariance_matrix
        assert output_covariance is not None
        np.testing.assert_allclose(output_covariance, covariance)

    def test_determine_at_trusted_wavelengths_keeps_old_block_when_extending(self):
        nm = np.arange(500, 601, 5)
        covariance = correlated_covariance(nm.size)
        obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        with warnings.catch_warnings():
            warnings.simplefilter('error', ErasingCorrelationsWarning)
            output = obj.determine_at_trusted_wavelengths(np.arange(490, 611, 5))

        old_indices = np.searchsorted(output.wavelength_nm, nm)
        assert output.covariance_matrix is not None
        assert output.covariance_matrix.shape == (output.spectral_size, output.spectral_size)
        np.testing.assert_allclose(
            output.covariance_matrix[np.ix_(old_indices, old_indices)],
            covariance,
        )

    def test_photometry_uses_full_covariance(self):
        nm = np.arange(470, 711, 5)
        covariance = correlated_covariance(nm.size)
        obj = ac.Spectrum(nm, np.ones(nm.size), covariance)
        bandpass = ac.Filter.get('Generic/Bessell.V')

        _, error = ac.get_photometry(obj, bandpass)

        target = ac.get_spectrometry(obj, bandpass.wavelength_nm, strictly=True)
        assert target.covariance_matrix is not None
        expected = cast(float, target.nm_step**2 * np.einsum(
            'i,ij,j->',
            bandpass.spectral_dist,
            target.covariance_matrix,
            bandpass.spectral_dist,
        ))
        assert error is not None
        np.testing.assert_allclose(float(error), expected)

    def test_constructor_interpolates_full_covariance(self):
        nm = np.array([500., 512., 524., 536., 548., 560., 572., 584., 596.])
        covariance = correlated_covariance(nm.size)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ErasingCorrelationsWarning)
            obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        operator = interpolation_matrix(nm, obj.wavelength_nm, obj.nm_step)
        expected = propagate_covariance(covariance, operator)
        covariance = obj.covariance_matrix
        assert covariance is not None
        np.testing.assert_allclose(covariance, expected)

    def test_constructor_bins_full_covariance(self):
        nm = np.arange(500., 600., 2.5)
        covariance = correlated_covariance(nm.size)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ErasingCorrelationsWarning)
            obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        operator = spectral_binning_matrix(
            nm, obj.wavelength_nm, obj.nm_step, np.diff(nm), nm.size
        )
        expected = propagate_covariance(covariance, operator)
        covariance = obj.covariance_matrix
        assert covariance is not None
        np.testing.assert_allclose(covariance, expected)

    def test_constructor_still_warns_for_spectral_downscaling(self):
        nm = np.array([500., 502., 510., 512., 520., 522., 530., 532., 540., 542.])
        covariance = correlated_covariance(nm.size)
        with pytest.warns(ErasingCorrelationsWarning):
            obj = ac.Spectrum(nm, np.ones(nm.size), covariance)
        assert obj.covariance_matrix is not None

    def test_emission_spectrum_preserves_cross_line_correlations(self):
        nm = np.array([500., 600.])
        covariance = np.array([[0.01, 0.008], [0.008, 0.04]])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ErasingCorrelationsWarning)
            obj = ac.Spectrum(nm, np.array([1., 2.]), covariance, is_emission_spectrum=True)

        first_kernel = ac.Spectrum.monochromatic(cast(float, nm[0]))
        last_kernel = ac.Spectrum.monochromatic(cast(float, nm[1]))
        nm_min, nm_max = last_kernel.get_extremal_grid_endpoints(
            (first_kernel.wavelength_nm[0], last_kernel.wavelength_nm[-1])
        )
        nm_uniform = last_kernel.uniform_grid(nm_min, nm_max)
        kernel_matrix = np.empty((nm_uniform.size, nm.size))
        for i in range(nm.size):
            line = ac.Spectrum.monochromatic(cast(float, nm[i])).determine_at_trusted_wavelengths(
                nm_uniform
            )
            kernel_matrix[:, i] = line.spectral_dist

        expected = propagate_covariance(covariance, kernel_matrix)
        assert obj.covariance_matrix is not None
        np.testing.assert_allclose(obj.covariance_matrix, expected)
        assert np.any(np.abs(obj.covariance_matrix - np.diag(np.diag(obj.covariance_matrix))) > 1e-12)

    def test_normalize_updates_covariance(self):
        nm = np.arange(500, 601, 5)
        std = np.ones(nm.size) * 0.1
        obj = ac.Spectrum(nm, np.ones(nm.size) * 2, std)
        expected = obj.normalized().covariance_matrix
        old_covariance = obj.covariance_matrix
        assert old_covariance is not None
        old_covariance = old_covariance.copy()

        obj.normalize()

        covariance = obj.covariance_matrix
        assert covariance is not None
        assert expected is not None
        np.testing.assert_allclose(covariance, expected)
        assert not np.allclose(covariance, old_covariance)

    def test_edges_to_zero_updates_covariance(self):
        nm = np.arange(500, 601, 5)
        std = np.ones(nm.size) * 0.1
        obj = ac.Spectrum(nm, np.ones(nm.size), std)
        old_covariance = obj.covariance_matrix
        assert old_covariance is not None
        old_covariance = old_covariance.copy()

        obj.edges_to_zero()

        covariance = obj.covariance_matrix
        assert covariance is not None
        assert covariance.shape == (obj.spectral_size, obj.spectral_size)
        np.testing.assert_allclose(covariance[1:-1, 1:-1], old_covariance)
        np.testing.assert_allclose(cast(npt.NDArray[np.floating], covariance[0]), 0)
        np.testing.assert_allclose(cast(npt.NDArray[np.floating], covariance[-1]), 0)

    def test_color_system_conversion_preserves_covariance(self):
        covariance = np.array([[0.01, 0.002, 0.001],
                               [0.002, 0.02, 0.003],
                               [0.001, 0.003, 0.03]])
        color = ac.ColorPoint(np.array([0.2, 0.3, 0.4]), covariance)
        new_system = ac.ColorSystem('sRGB', 'Illuminant E')

        converted = color.to_color_system(new_system)

        xyz_values, xyz_covariance = color.color_system.rgb_to_xyz(
            color.spectral_dist, color.covariance_matrix
        )
        _, expected_covariance = new_system.xyz_to_rgb(xyz_values, xyz_covariance)
        assert expected_covariance is not None
        covariance = converted.covariance_matrix
        assert covariance is not None
        np.testing.assert_allclose(covariance, expected_covariance)

    def test_color_image_upscale_keeps_covariance_shape(self):
        image = ac.ColorImage(
            np.ones((3, 2, 3)),
            np.ones((3, 3, 2, 3)) * 0.01,
        )
        upscaled = image.upscale(2)
        assert upscaled.spectral_dist.shape == (3, 4, 6)
        assert upscaled.covariance_matrix is not None
        assert upscaled.covariance_matrix.shape == (3, 3, 4, 6)

    def test_mixed_dimensional_addition_broadcasts_covariance(self):
        nm = np.arange(500, 601, 5)
        std = np.ones(nm.size) * 0.1
        spectral_set = ac.SpectralSet(nm, np.ones((nm.size, 2)))
        spectrum = ac.Spectrum(nm, np.ones(nm.size), std)

        result = spectral_set + spectrum

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (nm.size, nm.size, 2)
        np.testing.assert_allclose(np.diag(result.covariance_matrix[:, :, 0]), std**2)

    def test_mixed_dimensional_addition_broadcasts_into_cube(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Config, 'ignore_uncertainty_for_cubes', False)
        nm = np.arange(500, 601, 5)
        std = np.ones((nm.size, 2)) * 0.1
        spectral_cube = ac.SpectralCube(nm, np.ones((nm.size, 2, 2)))
        spectral_set = ac.SpectralSet(nm, np.ones((nm.size, 2)), std)

        result = spectral_cube + spectral_set

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (nm.size, nm.size, 2, 2)

    def test_invalid_covariance_shape_is_rejected(self):
        nm = np.arange(500, 601, 5)
        with pytest.raises(InconsistentUncertaintyShapeError):
            _ = ac.Spectrum(nm, np.ones(nm.size), np.ones((nm.size, nm.size + 1)))

    def test_shared_covariance_is_expanded_over_spatial_axes(self):
        nm = np.arange(500, 601, 5)
        covariance = np.eye(nm.size)
        obj = ac.SpectralSet(nm, np.ones((nm.size, 2)), covariance)
        assert obj.covariance_matrix is not None
        assert obj.covariance_matrix.shape == (nm.size, nm.size, 2)
        np.testing.assert_allclose(obj.covariance_matrix[:, :, 0], covariance)
        np.testing.assert_allclose(obj.covariance_matrix[:, :, 1], covariance)

    def test_spectral_binning_standard_deviation_uses_variances(self):
        nm0 = np.array([400., 405., 410., 415., 420.])
        nm1 = np.array([405, 410, 415])
        br0 = np.ones(nm0.size)
        std0 = np.array([0.1, 0.1, 0.2, 0.2, 0.2])
        diff = np.diff(nm0)

        _, std1 = spectral_binning(nm0, br0, std0, nm1, 5., diff)

        operator = spectral_binning_matrix(nm0, nm1, 5., diff, nm0.size)
        variance = cast(
            npt.NDArray[np.floating],
            np.einsum('ai,i...,ai->a...', operator, std0**2, operator),
        )
        expected = cast(npt.NDArray[np.floating], np.sqrt(variance))
        assert std1 is not None
        np.testing.assert_allclose(std1, expected)

    def test_constructor_interpolation_of_standard_deviation_uses_variances(self):
        nm = np.array([500., 512., 524., 536., 548., 560., 572., 584., 596.])
        std = np.linspace(0.1, 0.2, nm.size)
        obj = ac.Spectrum(nm, np.ones(nm.size), std)

        operator = interpolation_matrix(nm, obj.wavelength_nm, obj.nm_step)
        expected = np.sqrt(np.diag(
            propagate_covariance(
                Uncertainty.from_standard_deviation(std).covariance_matrix,
                operator,
            )
        ))
        assert obj.standard_deviation is not None
        np.testing.assert_allclose(obj.standard_deviation, expected)

    def test_color_image_batch_preserves_covariance(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Config, 'ignore_uncertainty_for_cubes', False)
        monkeypatch.setattr(Config, 'pixel_upper_limit', 2)
        nm = np.arange(400, 701, 5)
        std = np.ones((nm.size, 2, 2)) * 0.1
        cube = ac.SpectralCube(nm, np.ones((nm.size, 2, 2)), std)

        image = ac.ColorImage.from_spectral_data(cube)

        assert image.covariance_matrix is not None
        assert image.covariance_matrix.shape == (3, 3, 2, 2)

    def test_normalize_updates_covariance_for_spatial_objects(self):
        nm = np.arange(500, 601, 5)
        std = np.ones((nm.size, 2)) * 0.1
        obj = ac.SpectralSet(nm, np.ones((nm.size, 2)) * 2, std)
        expected = obj.normalized().covariance_matrix
        obj.normalize()
        assert expected is not None
        assert obj.covariance_matrix is not None
        np.testing.assert_allclose(obj.covariance_matrix, expected)

    def test_narrow_spectrum_preserves_full_covariance(self):
        nm = np.array([500., 502., 504.])
        covariance = correlated_covariance(nm.size)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ErasingCorrelationsWarning)
            obj = ac.Spectrum(nm, np.ones(nm.size), covariance)

        assert obj.covariance_matrix is not None
        assert obj.covariance_matrix.shape == (obj.spectral_size, obj.spectral_size)
        assert np.any(np.abs(obj.covariance_matrix - np.diag(np.diag(obj.covariance_matrix))) > 1e-12)
