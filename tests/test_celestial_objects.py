import math
import warnings
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import astrocolor as ac
from astrocolor.auxiliary import color_indices_parser, mag2irradiance
from astrocolor.celestial_objects import AlbedoInput, FilterInput, PhaseFunctionInput
from astrocolor.errors import (
    PhaseFunctionWarning,
    SpatialAxisWarning,
    SpectralAxisWarning,
)
from astrocolor.photometric_models import (
    HG,
    HG1G2,
    DefaultModel,
    Exponentials,
    Hapke,
    PhaseCoefficient,
    PhotometricModel,
)

V = ac.Filter.get('Generic/Bessell.V')
NM = np.arange(500, 601, 5)
SD = np.linspace(1., 2., NM.size)
BVR = ('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
SUN_NM = ac.sun_calspec.wavelength_nm
SUN_SD = ac.sun_calspec.spectral_dist


def solar_filter() -> ac.Filter:
    """ Full CALSPEC Solar reference spectrum used for the bond albedo. """
    return ac.Filter(SUN_NM, SUN_SD, name='Sun filter [CALSPEC]')


def photometry(obj: ac.Spectrum | ac.Photospectrum, bandpass: ac.Filter = V) -> float:
    value, _ = ac.get_photometry(obj, bandpass)
    return value


def color_index_std(photospectrum: ac.Photospectrum, first: int, second: int) -> float:
    """
    Propagates the photometry covariance into the standard deviation
    of a color index `-2.5 log10(F_first / F_second)`.
    """
    covariance = photospectrum.covariance_matrix
    assert covariance is not None
    cov = np.asarray(covariance, dtype=np.float64)
    flux = np.asarray(photospectrum.spectral_dist, dtype=np.float64)
    first_flux = cast(float, flux[first])
    second_flux = cast(float, flux[second])
    variance = (
        cast(float, cov[first, first]) / first_flux**2
        + cast(float, cov[second, second]) / second_flux**2
        - 2 * cast(float, cov[first, second]) / (first_flux * second_flux)
    )
    return float(2.5 / math.log(10) * math.sqrt(variance))


def reconstruct_bvri(
    photospectral: ac.Photospectrum | ac.PhotospectralSet,
    filter_set: ac.FilterSet,
) -> ac.Photospectrum | ac.PhotospectralSet:
    """
    Reconstructs the continuous spectrum on the BVRI grid and synthesizes
    photometry.

    The input photometry is already measured in these filters, so
    `get_photometry()` alone would return it unchanged. The explicit
    `get_spectrometry()` call forces the full processing cycle.
    """
    continuous = ac.get_spectrometry(
        photospectral,
        filter_set.wavelength_nm,
        strictly=True,
        attach_photospectral_obj=False,
    )
    return cast(
        ac.Photospectrum | ac.PhotospectralSet,
        ac.get_photometry(continuous, filter_set),
    )


class TestSpectralSources:

    def test_continuous_spectrum(self):
        obj = ac.CelestialObject('test', wavelength_nm=NM, spectral_dist=SD)
        assert obj.name == 'test'
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        np.testing.assert_allclose(obj.spectral_data_unscaled.wavelength_nm, NM)
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, SD)
        assert obj.spectral_data_geometric is None
        assert obj.spectral_data_spherical is None
        assert isinstance(obj.photometric_model, DefaultModel)
        assert obj.is_emissive is False

    def test_continuous_spectrum_with_uncertainty(self):
        obj = ac.CelestialObject('test', wavelength_nm=NM, spectral_dist=SD, uncertainty=0.1)
        assert obj.spectral_data_unscaled is not None
        covariance = obj.spectral_data_unscaled.covariance_matrix
        assert covariance is not None
        np.testing.assert_allclose(np.diag(covariance), np.full(NM.size, 0.01))

    def test_spectral_dist_with_std(self):
        sd = [[value, 0.1 * value] for value in cast(list[float], SD.tolist())]
        obj = ac.CelestialObject('test', wavelength_nm=NM, spectral_dist=sd)
        assert obj.spectral_data_unscaled is not None
        std = obj.spectral_data_unscaled.standard_deviation
        assert std is not None
        np.testing.assert_allclose(std, 0.1 * SD)

    def test_magnitudes(self):
        magnitudes = np.array([0., -1., 2.])
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=[500., 505., 510.],
            magnitudes=magnitudes
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        expected: npt.NDArray[np.floating] = np.asarray(mag2irradiance(magnitudes), dtype=np.float64)
        expected = expected / float(np.mean(expected))
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected)

    def test_magnitudes_with_uncertainty(self):
        magnitudes = np.array([[0., 0.1], [-1., 0.2], [2., 0.3]])
        obj = ac.CelestialObject('test', wavelength_nm=[500., 505., 510.], magnitudes=magnitudes)
        assert obj.spectral_data_unscaled is not None
        std = obj.spectral_data_unscaled.standard_deviation
        assert std is not None
        expected_irradiance: npt.NDArray[np.floating] = np.asarray(
            mag2irradiance(np.array([0., -1., 2.])),
            dtype=np.float64
        )
        expected_irradiance = expected_irradiance / float(np.mean(expected_irradiance))
        expected_std: npt.NDArray[np.floating] = (
            0.4 * math.log(10) * expected_irradiance * np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
        np.testing.assert_allclose(std, expected_std)

    def test_nm_range(self):
        obj = ac.CelestialObject(
            'test',
            nm_range={'start': 500, 'stop': 600, 'step': 10},
            sd_geometric=np.ones(11)
        )
        assert isinstance(obj.spectral_data_geometric, ac.Spectrum)
        np.testing.assert_equal(obj.spectral_data_geometric.wavelength_nm, np.arange(500, 601, 5))
        np.testing.assert_allclose(obj.spectral_data_geometric.spectral_dist, 1.)
        assert obj.spectral_data_unscaled is None

    def test_spectral_slope_power(self):
        obj = ac.CelestialObject(
            'test',
            spectral_slope={'start': 500, 'stop': 700, 'power': 2.}
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        nm = obj.spectral_data_unscaled.wavelength_nm
        sd = obj.spectral_data_unscaled.spectral_dist
        np.testing.assert_allclose(sd[nm == 600], 1.)
        np.testing.assert_allclose(sd[nm == 700], (700 / 600)**2)

    def test_spectral_slope_power_std(self):
        obj = ac.CelestialObject(
            'test',
            spectral_slope={'start': 500, 'stop': 700, 'power': [2., 0.5]}
        )
        assert obj.spectral_data_unscaled is not None
        nm = obj.spectral_data_unscaled.wavelength_nm
        std = obj.spectral_data_unscaled.standard_deviation
        assert std is not None
        std_at_700 = cast(float, std[nm == 700][0])
        ratio = 700 / 600
        expected_std_at_700 = ratio * ratio * (math.sqrt(ratio) - 1)
        np.testing.assert_allclose(std[nm == 600], 0.)
        np.testing.assert_allclose(std_at_700, expected_std_at_700)

    def test_spectral_slope_percent(self):
        obj = ac.CelestialObject(
            'test',
            spectral_slope={'start': 500, 'stop': 700, 'percent_per_100nm': 10.}
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        nm = obj.spectral_data_unscaled.wavelength_nm
        sd = obj.spectral_data_unscaled.spectral_dist
        np.testing.assert_allclose(sd[nm == 600], 1.)
        np.testing.assert_allclose(sd[nm == 500], 0.9)
        np.testing.assert_allclose(sd[nm == 700], 1.1)

    def test_spectral_slope_percent_std(self):
        obj = ac.CelestialObject(
            'test',
            spectral_slope={'start': 500, 'stop': 700, 'percent_per_100nm': [10., 2.]}
        )
        assert obj.spectral_data_unscaled is not None
        nm = obj.spectral_data_unscaled.wavelength_nm
        std = obj.spectral_data_unscaled.standard_deviation
        assert std is not None
        np.testing.assert_allclose(std[nm == 600], 0.)
        np.testing.assert_allclose(std[nm == 700], 0.02)

    def test_spectral_slope_percent_scales_with_range(self):
        # 10% per 100 nm over 400 nm gives ±20% at the endpoints.
        obj = ac.CelestialObject(
            'test',
            spectral_slope={'start': 500, 'stop': 900, 'percent_per_100nm': 10.}
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        nm = obj.spectral_data_unscaled.wavelength_nm
        sd = obj.spectral_data_unscaled.spectral_dist
        np.testing.assert_allclose(sd[nm == 500], 0.8)
        np.testing.assert_allclose(sd[nm == 700], 1.)
        np.testing.assert_allclose(sd[nm == 900], 1.2)

    def test_filters_with_filter_set(self):
        obj = ac.CelestialObject(
            'test',
            filter_set='Generic/Bessell',
            filters=['B', 'V', 'R'],
            spectral_dist=[0.1, 0.2, 0.3]
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Photospectrum)
        assert obj.spectral_data_unscaled.filter_set.name == 'Generic/Bessell'
        assert len(obj.spectral_data_unscaled.filter_set) == 3
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, [0.1, 0.2, 0.3])

    def test_filters_with_monochromatic_wavelengths(self):
        obj = ac.CelestialObject('test', filters=[500, 550, 600], spectral_dist=[1., 2., 3.])
        assert isinstance(obj.spectral_data_unscaled, ac.Photospectrum)
        assert len(obj.spectral_data_unscaled.filter_set) == 3

    def test_filters_with_numpy_wavelengths(self):
        obj = ac.CelestialObject(
            'test',
            filters=np.array([500, 550, 600]),
            spectral_dist=[1., 2., 3.]
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Photospectrum)
        assert len(obj.spectral_data_unscaled.filter_set) == 3

    def test_single_filter_string_rejected(self):
        with pytest.raises(TypeError, match='sequence'):
            _ = ac.CelestialObject('test', filters='Generic/Bessell.V', spectral_dist=[1.])

    def test_unsupported_filter_type(self):
        with pytest.raises(TypeError, match='filter name type'):
            _ = ac.CelestialObject(
                'test',
                filters=cast(FilterInput, [object()]),
                spectral_dist=[1.]
            )

    def test_color_indices(self):
        indices = {'B-V': 0.5, 'V-R': 0.2}
        obj = ac.CelestialObject(
            'test',
            filter_set='Generic/Bessell',
            color_indices=indices
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Photospectrum)
        assert obj.spectral_data_unscaled.filter_set.name == 'Generic/Bessell'
        _, expected_sd, expected_std = color_indices_parser(indices)
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected_sd)
        assert expected_std is None
        assert obj.spectral_data_unscaled.covariance_matrix is None

    def test_color_indices_with_std(self):
        indices = {'B-V': [0.5, 0.1], 'V-R': [0.2, 0.05]}
        obj = ac.CelestialObject(
            'test',
            filter_set='Generic/Bessell',
            color_indices=indices
        )
        assert obj.spectral_data_unscaled is not None
        _, expected_sd, _ = color_indices_parser(indices)
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected_sd)
        assert obj.spectral_data_unscaled.covariance_matrix is not None

    def test_calibration_vega(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            calibration_system='Vega'
        )
        vega_norm = ac.scale_to_match_value(ac.vega_calspec, V)
        expected = ac.Spectrum(NM, SD) * vega_norm
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected.spectral_dist)
        assert obj.spectral_data_unscaled.covariance_matrix is None

    def test_calibration_ab(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            calibration_system='AB'
        )
        expected = ac.Spectrum(NM, SD).convert_from_energy_spectral_density_per_frequency()
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected.spectral_dist)

    def test_reflecting_sunlight(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_reflecting_sunlight=True
        )
        sun_norm = ac.scale_to_match_value(ac.sun_tsis1_hsrs, V)
        expected = ac.Spectrum(NM, SD) / sun_norm
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected.spectral_dist)

    def test_emission_spectrum(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=[500, 600],
            spectral_dist=[1., 2.],
            is_emission_spectrum=True
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        np.testing.assert_allclose(obj.spectral_data_unscaled.integrate(), 3.)
        assert obj.is_emissive is True

    def test_empty_spectral_axis_warns(self):
        with pytest.warns(SpectralAxisWarning, match='test'):
            obj = ac.CelestialObject('test', spectral_dist=[1., 2., 3.])
        spectral_data, estimated = obj.get_spectral_data()
        assert isinstance(spectral_data, ac.Spectrum)
        assert estimated is None
        np.testing.assert_allclose(spectral_data.integrate(), 0.) # a stub object

    def test_empty_spatial_axis_warns(self):
        with pytest.warns(SpatialAxisWarning, match='test'):
            obj = ac.CelestialObject('test', wavelength_nm=NM)
        spectral_data, estimated = obj.get_spectral_data()
        assert isinstance(spectral_data, ac.Spectrum)
        assert estimated is None
        np.testing.assert_allclose(spectral_data.integrate(), 0.) # a stub object


class TestAlbedo:

    def test_geometric_albedo(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            geometric_albedo=['Generic/Bessell.V', [0.4, 0.02]]
        )
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is None
        assert photometry(obj.spectral_data_geometric) == pytest.approx(0.4)

    def test_spherical_albedo(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            spherical_albedo=['Generic/Bessell.V', 0.3]
        )
        assert obj.spectral_data_spherical is not None
        assert obj.spectral_data_geometric is None
        assert photometry(obj.spectral_data_spherical) == pytest.approx(0.3)

    def test_albedo_scales_both(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            albedo=[550, 0.25]
        )
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is not None
        monochromatic_550 = ac.Filter.monochromatic(550)
        assert photometry(obj.spectral_data_geometric, monochromatic_550) == pytest.approx(0.25)
        assert photometry(obj.spectral_data_spherical, monochromatic_550) == pytest.approx(0.25)

    def test_albedo_with_numpy_wavelength(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            geometric_albedo=[np.int64(550), 0.25]
        )
        assert obj.spectral_data_geometric is not None
        assert photometry(obj.spectral_data_geometric, ac.Filter.monochromatic(550)) == pytest.approx(0.25)

    def test_bond_albedo(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            bond_albedo=[0.25, 0.01]
        )
        assert obj.spectral_data_spherical is not None
        assert obj.spectral_data_geometric is None
        sun = solar_filter()
        assert photometry(obj.spectral_data_spherical, sun) == pytest.approx(0.25)

    def test_bond_albedo_with_photospectrum(self):
        obj = ac.CelestialObject(
            'test',
            filter_set='Generic/Bessell',
            color_indices={'B-V': [0.5, 0.1]},
            geometric_albedo=['Generic/Bessell.V', 0.3],
            bond_albedo=[0.2, 0.01]
        )
        assert isinstance(obj.spectral_data_spherical, ac.Photospectrum)
        sun = solar_filter()
        assert photometry(obj.spectral_data_spherical, sun) == pytest.approx(0.2)

    def test_sd_geometric(self):
        sd_geometric = [0.1, 0.2, 0.3]
        obj = ac.CelestialObject(
            'test',
            filters=BVR,
            sd_geometric=sd_geometric,
            std_geometric=0.01
        )
        assert obj.spectral_data_unscaled is None
        assert isinstance(obj.spectral_data_geometric, ac.Photospectrum)
        np.testing.assert_allclose(obj.spectral_data_geometric.spectral_dist, sd_geometric)
        std = obj.spectral_data_geometric.standard_deviation
        assert std is not None
        np.testing.assert_allclose(std, 0.01)

    def test_sd_spherical_with_geometric_scaling(self):
        sd_spherical = [0.1, 0.2, 0.3]
        obj = ac.CelestialObject(
            'test',
            filters=BVR,
            sd_spherical=sd_spherical,
            geometric_albedo=['Generic/Bessell.V', 0.5]
        )
        assert isinstance(obj.spectral_data_spherical, ac.Photospectrum)
        np.testing.assert_allclose(obj.spectral_data_spherical.spectral_dist, sd_spherical)
        assert obj.spectral_data_geometric is not None
        assert photometry(obj.spectral_data_geometric) == pytest.approx(0.5)

    def test_sd_geometric_and_sd_spherical(self):
        obj = ac.CelestialObject(
            'test',
            filters=BVR,
            sd_geometric=[0.1, 0.2, 0.3],
            sd_spherical=[0.01, 0.02, 0.03]
        )
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is not None
        np.testing.assert_allclose(obj.spectral_data_geometric.spectral_dist, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(obj.spectral_data_spherical.spectral_dist, [0.01, 0.02, 0.03])

    def test_supplied_albedos_are_not_overwritten_by_constraints(self):
        obj = ac.CelestialObject(
            'test',
            filters=BVR,
            sd_geometric=[0.1, 0.2, 0.3],
            sd_spherical=[0.01, 0.02, 0.03],
            geometric_albedo=['Generic/Bessell.V', 0.5],
            spherical_albedo=['Generic/Bessell.V', 0.1],
            bond_albedo=0.05
        )
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is not None
        np.testing.assert_allclose(obj.spectral_data_geometric.spectral_dist, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(obj.spectral_data_spherical.spectral_dist, [0.01, 0.02, 0.03])

    def test_is_geometric_albedo(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_geometric_albedo=True,
            spherical_albedo=['Generic/Bessell.V', 0.1]
        )
        assert obj.spectral_data_geometric is obj.spectral_data_unscaled
        assert obj.spectral_data_spherical is not None
        assert obj.spectral_data_geometric is not None
        np.testing.assert_allclose(obj.spectral_data_geometric.spectral_dist, SD)
        assert photometry(obj.spectral_data_spherical) == pytest.approx(0.1)

    def test_is_spherical_albedo(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_spherical_albedo=True,
            geometric_albedo=['Generic/Bessell.V', 0.2]
        )
        assert obj.spectral_data_spherical is obj.spectral_data_unscaled
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is not None
        np.testing.assert_allclose(obj.spectral_data_spherical.spectral_dist, SD)
        assert photometry(obj.spectral_data_geometric) == pytest.approx(0.2)

    def test_is_albedo(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_albedo=True
        )
        assert obj.spectral_data_geometric is obj.spectral_data_unscaled
        assert obj.spectral_data_spherical is obj.spectral_data_unscaled
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, SD)

    def test_invalid_albedo_length(self):
        with pytest.raises(ValueError, match='exactly two'):
            _ = ac.CelestialObject('test', wavelength_nm=NM, spectral_dist=SD, geometric_albedo=[0.5])

    def test_invalid_albedo_location(self):
        with pytest.raises(TypeError, match='Unsupported albedo location'):
            _ = ac.CelestialObject(
                'test',
                wavelength_nm=NM,
                spectral_dist=SD,
                geometric_albedo=cast(AlbedoInput, [object(), 0.5])
            )


class TestPhotometricModels:

    def test_phase_coefficient(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            phase_function=['phase coefficient', {'beta': [0.042, 0.008]}]
        )
        assert isinstance(obj.photometric_model, PhaseCoefficient)
        assert obj.photometric_model.phase_integral is not None

    def test_phase_function_with_filter(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            phase_function=['Generic/Bessell.V', 'HG', {'G': 0.2}]
        )
        assert isinstance(obj.photometric_model, HG)
        assert obj.photometric_model.filter_or_nm == 'Generic/Bessell.V'

    def test_phase_function_with_numpy_wavelength(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            phase_function=[np.int64(550), 'HG', {'G': 0.2}]
        )
        assert isinstance(obj.photometric_model, HG)
        assert obj.photometric_model.filter_or_nm == 550.

    @pytest.mark.parametrize(
        'phase_function, expected_type',
        (
            (['exponentials', {'A_1': 0.5, 'mu_1': 1.}], Exponentials),
            (['HG', {'G': 0.2}], HG),
            (['HG1G2', {'G_1': 0.5, 'G_2': 0.2}], HG1G2),
            (['Hapke', {'w': 0.5, 'bo': 1., 'h': 0.1, 'b': 0.2, 'c': 0.1, 'theta': 20.}], Hapke),
        )
    )
    def test_supported_models(
        self,
        phase_function: PhaseFunctionInput,
        expected_type: type[PhotometricModel]
    ) -> None:
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            phase_function=phase_function
        )
        assert isinstance(obj.photometric_model, expected_type)

    def test_model_albedo_does_not_scale_without_location(self):
        # A_1 = 0.5 makes Exponentials set its own geometric albedo, but there is
        # no filter/wavelength to relate it to the data, so no scaling should happen.
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            phase_function=['exponentials', {'A_1': 0.5, 'mu_1': 1.}]
        )
        assert obj.photometric_model.geometric_albedo is not None
        assert obj.spectral_data_geometric is None

    def test_phase_integral_override(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            phase_function=['HG', {'G': 0.2}],
            phase_integral=[0.6, 0.01]
        )
        phase_integral = obj.photometric_model.phase_integral
        assert phase_integral is not None
        assert phase_integral[0] == pytest.approx(0.6)
        assert phase_integral[1] == pytest.approx(0.01)

    def test_spherical_albedo_error_uses_variances(self):
        model = PhaseCoefficient(cast(dict[str, float | tuple[float]], {'beta': (0.1, 0.01)}))
        model.geometric_albedo = (0.2, 0.02)
        phase_integral = cast(tuple[float, float | None], model.phase_integral)
        spherical = model.spherical_albedo
        assert spherical is not None
        value, std = spherical
        assert phase_integral[1] is not None
        expected_variance = (0.02 * phase_integral[0])**2 + (phase_integral[1] * 0.2)**2
        assert value == pytest.approx(0.2 * phase_integral[0])
        assert std == pytest.approx(math.sqrt(expected_variance))

    def test_unsupported_model(self):
        with pytest.warns(PhaseFunctionWarning, match='is not supported'):
            obj = ac.CelestialObject(
                'test',
                wavelength_nm=NM,
                spectral_dist=SD,
                phase_function=['unknown', {}]
            )
        assert isinstance(obj.photometric_model, DefaultModel)

    def test_invalid_phase_function_length(self):
        with pytest.warns(PhaseFunctionWarning, match='Invalid number of'):
            obj = ac.CelestialObject(
                'test',
                wavelength_nm=NM,
                spectral_dist=SD,
                phase_function=['HG']
            )
        assert isinstance(obj.photometric_model, DefaultModel)


class TestGetSpectralData:

    def test_geometric_known(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            geometric_albedo=['Generic/Bessell.V', 0.4]
        )
        spectral_data, estimated = obj.get_spectral_data('geometric')
        assert spectral_data is obj.spectral_data_geometric
        assert estimated is False

    def test_spherical_known(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            spherical_albedo=['Generic/Bessell.V', 0.3]
        )
        spectral_data, estimated = obj.get_spectral_data('spherical')
        assert spectral_data is obj.spectral_data_spherical
        assert estimated is False

    def test_estimate_spherical(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            geometric_albedo=['Generic/Bessell.V', 0.4]
        )
        assert obj.spectral_data_geometric is not None
        geometric_in_V = photometry(obj.spectral_data_geometric)
        expected = geometric_in_V * (0.359 + 0.47 * geometric_in_V)
        spherical, estimated = obj.get_spectral_data('spherical')
        assert estimated is True
        assert spherical is not None
        assert photometry(spherical) == pytest.approx(expected)

    def test_estimate_geometric(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            spherical_albedo=['Generic/Bessell.V', 0.1]
        )
        assert obj.spectral_data_spherical is not None
        spherical_in_V = photometry(obj.spectral_data_spherical)
        expected = (math.sqrt(0.359**2 + 4 * 0.47 * spherical_in_V) - 0.359) / (2 * 0.47)
        geometric, estimated = obj.get_spectral_data('geometric')
        assert estimated is True
        assert geometric is not None
        assert photometry(geometric) == pytest.approx(expected)

    def test_exact_conversion_with_phase_integral(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            geometric_albedo=['Generic/Bessell.V', 0.4],
            phase_integral=0.5
        )
        spherical, estimated = obj.get_spectral_data('spherical')
        assert estimated is False
        assert spherical is not None
        assert photometry(spherical) == pytest.approx(0.2)

        spherical_obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            spherical_albedo=['Generic/Bessell.V', 0.2],
            phase_integral=0.5
        )
        geometric, estimated = spherical_obj.get_spectral_data('geometric')
        assert estimated is False
        assert geometric is not None
        assert photometry(geometric) == pytest.approx(0.4)

    def test_unscaled_fallback(self):
        obj = ac.CelestialObject('test', wavelength_nm=NM, spectral_dist=SD)
        spectral_data, estimated = obj.get_spectral_data('geometric')
        assert spectral_data is obj.spectral_data_unscaled
        assert estimated is None
        spectral_data, estimated = obj.get_spectral_data('spherical')
        assert spectral_data is obj.spectral_data_unscaled
        assert estimated is None

    def test_albedo_only_object(self):
        obj = ac.CelestialObject(
            'test',
            filters=BVR,
            sd_geometric=[0.1, 0.2, 0.3],
            phase_integral=0.5
        )
        geometric, estimated = obj.get_spectral_data('geometric')
        assert geometric is obj.spectral_data_geometric
        assert estimated is False
        spherical, estimated = obj.get_spectral_data('spherical')
        assert estimated is False
        assert spherical is not None
        assert obj.spectral_data_geometric is not None
        geometric_in_V = photometry(obj.spectral_data_geometric)
        assert photometry(spherical) == pytest.approx(geometric_in_V * 0.5)

    def test_invalid_mode(self):
        with pytest.warns(SpatialAxisWarning, match='test'):
            obj = ac.CelestialObject('test', wavelength_nm=NM)
        spectral_data, estimated = obj.get_spectral_data('invalid')  # pyright: ignore[reportArgumentType]
        assert isinstance(spectral_data, ac.Spectrum)
        assert estimated is None
        np.testing.assert_allclose(spectral_data.integrate(), 0.) # a stub object


class TestFlags:

    def test_is_emissive_false(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_emissive=False
        )
        assert obj.is_emissive is False

    def test_is_emissive_true(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_emissive=True
        )
        assert obj.is_emissive is True

    def test_emission_spectrum_implies_emissive(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=[500, 600],
            spectral_dist=[1., 2.],
            is_emission_spectrum=True
        )
        assert obj.is_emissive is True

    def test_is_reflecting_sunlight_false(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_reflecting_sunlight=False
        )
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, SD)

    def test_photon_counter_false(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_photon_counter=False
        )
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, SD)

    def test_photon_counter_true(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_photon_counter=True
        )
        expected = ac.Spectrum(NM, SD).convert_from_photon_spectral_density()
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected.spectral_dist)

    def test_photon_counter_converts_object_once(self):
        obj = ac.CelestialObject(
            'test',
            wavelength_nm=NM,
            spectral_dist=SD,
            is_albedo=True,
            is_photon_counter=True
        )
        assert obj.spectral_data_geometric is obj.spectral_data_unscaled
        assert obj.spectral_data_spherical is obj.spectral_data_unscaled
        expected = ac.Spectrum(NM, SD).convert_from_photon_spectral_density()
        assert obj.spectral_data_unscaled is not None
        np.testing.assert_allclose(obj.spectral_data_unscaled.spectral_dist, expected.spectral_dist)
        # A double conversion would produce a different (and much steeper) spectrum.
        assert not np.allclose(
            obj.spectral_data_unscaled.spectral_dist,
            expected.convert_from_photon_spectral_density().spectral_dist
        )


class TestDatabaseStyleInputs:
    """ Representative TrueColorTools entries (without `tags`/`file`). """

    def test_mercury_filters_and_spherical_albedo(self):
        obj = ac.CelestialObject(
            name='Mercury',
            filter_set='Generic/Bessell',
            filters=['U', 'B', 'V', 'R'],
            sd_geometric=[0.087, 0.105, 0.142, 0.172],
            spherical_albedo=['Generic/Bessell.V', 0.07]
        )
        assert isinstance(obj.spectral_data_geometric, ac.Photospectrum)
        assert obj.spectral_data_spherical is not None
        assert photometry(obj.spectral_data_spherical) == pytest.approx(0.07)

    def test_io_wavelength_range(self):
        obj = ac.CelestialObject(
            name='Io',
            nm_range={'start': 320, 'stop': 780, 'step': 20},
            sd_geometric=[
                0.11, 0.12, 0.14, 0.18, 0.26, 0.36, 0.47, 0.56, 0.64, 0.71, 0.73, 0.74,
                0.75, 0.76, 0.78, 0.81, 0.83, 0.86, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89
            ],
            phase_integral=0.59
        )
        assert isinstance(obj.spectral_data_geometric, ac.Spectrum)
        assert obj.spectral_data_geometric.wavelength_nm[0] == 320
        assert obj.spectral_data_geometric.wavelength_nm[-1] == 780

    def test_titan_is_albedo(self):
        obj = ac.CelestialObject(
            name='Titan',
            wavelength_nm=[520, 563, 595, 642, 685],
            spectral_dist=[0.093, 0.107, 0.116, 0.130, 0.139],
            is_albedo=True
        )
        assert obj.spectral_data_geometric is obj.spectral_data_unscaled
        assert obj.spectral_data_spherical is obj.spectral_data_unscaled

    def test_hyperion_color_indices_and_bond_albedo(self):
        obj = ac.CelestialObject(
            name='Hyperion',
            filter_set='Generic/Bessell',
            color_indices={'B-V': [0.798, 0.030], 'V-R': [0.438, 0.031]},
            calibration_system='Vega',
            is_reflecting_sunlight=True,
            geometric_albedo=['Generic/Bessell.V', [0.44, 0.13]],
            bond_albedo=[0.33, 0.05]
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Photospectrum)
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is not None
        assert photometry(obj.spectral_data_geometric) == pytest.approx(0.44)

    def test_uranus_rings_spectral_slope(self):
        obj = ac.CelestialObject(
            name='Rings of Uranus',
            spectral_slope={'start': 320, 'stop': 2050, 'power': 0.09},
            geometric_albedo=[550, 0.050],
            spherical_albedo=[550, [0.018, 0.003]]
        )
        assert isinstance(obj.spectral_data_unscaled, ac.Spectrum)
        assert obj.spectral_data_geometric is not None
        assert obj.spectral_data_spherical is not None
        monochromatic_550 = ac.Filter.monochromatic(550)
        assert photometry(obj.spectral_data_geometric, monochromatic_550) == pytest.approx(0.050)
        assert photometry(obj.spectral_data_spherical, monochromatic_550) == pytest.approx(0.018)

    def test_proteus_magnitudes(self):
        obj = ac.CelestialObject(
            name='Proteus',
            filter_set='Generic/Bessell',
            filters=['U', 'B', 'V'],
            magnitudes=[[5.81, 0.02], [5.09, 0.01], [4.36, 0.05]],
            calibration_system='Vega',
            is_reflecting_sunlight=True,
            geometric_albedo=['Generic/Bessell.V', [0.093, 0.001]],
            phase_function=['phase coefficient', {'beta': 0.039}]
        )
        assert isinstance(obj.photometric_model, PhaseCoefficient)

    def test_aurora_is_emissive(self):
        obj = ac.CelestialObject(
            name='Aurora',
            filter_set='Generic/Bessell',
            filters=['B', 'V', 'R'],
            spectral_dist=[1.3, 0.6, 0.8],
            is_emissive=True
        )
        assert obj.is_emissive is True
        assert obj.spectral_data_geometric is None
        spectral_data, estimated = obj.get_spectral_data('geometric')
        assert spectral_data is obj.spectral_data_unscaled
        assert estimated is None


class TestFullCycleRoundtrip:
    """
    Passes a database entry through the complete processing cycle:
    color indices -> Vega-calibrated reflected-sunlight photospectrum ->
    geometric albedo -> continuous Tikhonov reconstruction -> synthetic BVRI
    photometry -> Solar/Vega normalization -> color indices.

    The recovered colors must reproduce the input ones within the propagated
    uncertainty, which is expected to be only slightly larger than the input.
    """

    # Reconstructed colors of the current implementation, used as regression
    # anchors. A 5% drift is allowed so that accuracy improvements still pass.
    B_V_REFERENCE: float = 1.367130
    V_R_REFERENCE: float = 0.841156
    R_I_REFERENCE: float = 0.819074

    def test_gonggong_bvri_roundtrip(self):
        input_b_v, input_b_v_std = 1.38, 0.03
        input_v_r, input_v_r_std = 0.86, 0.02
        input_r_i, input_r_i_std = 0.79, 0.02
        color_indices = {
            'B-V': [input_b_v, input_b_v_std],
            'V-R': [input_v_r, input_v_r_std],
            'R-I': [input_r_i, input_r_i_std],
        }
        celestial_obj = ac.CelestialObject(
            name='(225088) Gonggong | Boehnhardt2014, Kiss2019',
            filter_set='Generic/Bessell',
            color_indices=color_indices,
            calibration_system='Vega',
            is_reflecting_sunlight=True,
            geometric_albedo=['Generic/Bessell.V', [0.14, 0.01]],
        )
        spectrum, estimated = celestial_obj.get_spectral_data('geometric')
        assert estimated is False
        assert isinstance(spectrum, ac.Photospectrum)

        filter_set = ac.FilterSet.get(
            'Generic/Bessell.B',
            'Generic/Bessell.V',
            'Generic/Bessell.R',
            'Generic/Bessell.I',
        )
        bvri_photometry = cast(ac.Photospectrum, reconstruct_bvri(spectrum, filter_set))
        # The V-band geometric albedo must survive the reconstruction.
        assert bvri_photometry.spectral_dist[1] == pytest.approx(0.14, abs=0.001)

        bvri_photometry *= ac.sun_tsis1_hsrs
        bvri_photometry /= ac.vega_calspec
        flux = np.asarray(bvri_photometry.spectral_dist, dtype=np.float64)
        b = cast(float, flux[0])
        v = cast(float, flux[1])
        r = cast(float, flux[2])
        i = cast(float, flux[3])
        b_v = -2.5 * math.log10(b / v)
        v_r = -2.5 * math.log10(v / r)
        r_i = -2.5 * math.log10(r / i)
        b_v_std = color_index_std(bvri_photometry, 0, 1)
        v_r_std = color_index_std(bvri_photometry, 1, 2)
        r_i_std = color_index_std(bvri_photometry, 2, 3)

        # The recovered colors stay within two reconstructed standard
        # deviations of the input values.
        assert abs(b_v - input_b_v) <= 2 * b_v_std
        assert abs(v_r - input_v_r) <= 2 * v_r_std
        assert abs(r_i - input_r_i) <= 2 * r_i_std

        # Regression anchors: the reconstructed colors must not drift by more
        # than 5% from the values recorded for the current implementation.
        assert b_v == pytest.approx(self.B_V_REFERENCE, rel=0.05)
        assert v_r == pytest.approx(self.V_R_REFERENCE, rel=0.05)
        assert r_i == pytest.approx(self.R_I_REFERENCE, rel=0.05)

        # The reconstruction is expected to be slightly less precise than the
        # input, but not drastically so.
        assert input_b_v_std <= b_v_std <= 2 * input_b_v_std
        assert input_v_r_std <= v_r_std <= 2 * input_v_r_std
        assert input_r_i_std <= r_i_std <= 2 * input_r_i_std

    def test_gonggong_bvri_monte_carlo(self):
        """
        Checks the uncertainty propagation statistically.

        One thousand realizations of the input BVRI photometry are drawn from
        its covariance matrix and passed through the same reconstruction and
        Solar/Vega normalization. The empirical color scatter must match the
        analytical uncertainties obtained from one deterministic pass.
        """
        # The 5% tolerance accommodates the ~1.4% standard error of 2500 samples.
        sample_count = 2_500
        # A fixed seed keeps the statistical test reproducible.
        rng = np.random.default_rng(42)
        input_b_v, input_b_v_std = 1.38, 0.03
        input_v_r, input_v_r_std = 0.86, 0.02
        input_r_i, input_r_i_std = 0.79, 0.02
        color_indices = {
            'B-V': [input_b_v, input_b_v_std],
            'V-R': [input_v_r, input_v_r_std],
            'R-I': [input_r_i, input_r_i_std],
        }
        celestial_obj = ac.CelestialObject(
            name='(225088) Gonggong | Boehnhardt2014, Kiss2019',
            filter_set='Generic/Bessell',
            color_indices=color_indices,
            calibration_system='Vega',
            is_reflecting_sunlight=True,
            geometric_albedo=['Generic/Bessell.V', [0.14, 0.01]],
        )
        spectrum, estimated = celestial_obj.get_spectral_data('geometric')
        assert estimated is False
        assert isinstance(spectrum, ac.Photospectrum)
        covariance = spectrum.covariance_matrix
        assert covariance is not None

        filter_set = ac.FilterSet.get(
            'Generic/Bessell.B',
            'Generic/Bessell.V',
            'Generic/Bessell.R',
            'Generic/Bessell.I',
        )

        # Analytical reference: one deterministic pass through the pipeline.
        deterministic = cast(ac.Photospectrum, reconstruct_bvri(spectrum, filter_set))
        deterministic *= ac.sun_tsis1_hsrs
        deterministic /= ac.vega_calspec
        reported_stds = (
            color_index_std(deterministic, 0, 1),
            color_index_std(deterministic, 1, 2),
            color_index_std(deterministic, 2, 3),
        )

        # Batched Monte Carlo: all realizations are reconstructed at once.
        samples = rng.multivariate_normal(
            spectrum.spectral_dist,
            covariance,
            size=sample_count,
        ).T
        photospectral_set = ac.PhotospectralSet(spectrum.filter_set, samples)
        mc_photometry = cast(ac.PhotospectralSet, reconstruct_bvri(photospectral_set, filter_set))
        mc_photometry *= ac.sun_tsis1_hsrs
        mc_photometry /= ac.vega_calspec
        mc_flux = np.asarray(mc_photometry.spectral_dist, dtype=np.float64)
        b = np.asarray(mc_flux[0], dtype=np.float64)
        v = np.asarray(mc_flux[1], dtype=np.float64)
        r = np.asarray(mc_flux[2], dtype=np.float64)
        i = np.asarray(mc_flux[3], dtype=np.float64)
        mc_b_v = -2.5 * np.log10(b / v)
        mc_v_r = -2.5 * np.log10(v / r)
        mc_r_i = -2.5 * np.log10(r / i)
        rows = (
            ('B-V', mc_b_v, input_b_v, self.B_V_REFERENCE, input_b_v_std, reported_stds[0]),
            ('V-R', mc_v_r, input_v_r, self.V_R_REFERENCE, input_v_r_std, reported_stds[1]),
            ('R-I', mc_r_i, input_r_i, self.R_I_REFERENCE, input_r_i_std, reported_stds[2]),
        )

        for name, values, target, reference, input_std, reported_std in rows:
            assert np.all(np.isfinite(values))
            empirical_mean = float(np.mean(values))
            empirical_std = float(np.std(values, ddof=1))
            # The mean stays within two recovered standard deviations.
            assert abs(empirical_mean - target) <= 2 * reported_std, name
            # The MC mean also stays within 5% of the regression anchor.
            assert empirical_mean == pytest.approx(reference, rel=0.05), name
            # The empirical scatter matches the propagated uncertainty to 5%.
            assert empirical_std == pytest.approx(reported_std, rel=0.05), name
            # The reconstruction is slightly less precise than the input.
            assert input_std <= empirical_std <= 2 * input_std, name


def test_no_unexpected_warnings():
    """ A normal database-style object should not emit warnings. """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = ac.CelestialObject(
            'test',
            filter_set='Generic/Bessell',
            filters=['B', 'V', 'R'],
            sd_geometric=[0.3, 0.4, 0.2],
            spherical_albedo=['Generic/Bessell.V', 0.1]
        )
