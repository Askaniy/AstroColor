from typing import cast

import numpy as np
import pytest

import astrocolor as ac
from astrocolor.errors import (
    InconsistentAxesError,
    NonFiniteFilterProfileError,
    NonFiniteFilterWavelengthError,
    NonPositiveFilterIntegralError,
)

# Ten profiles bundled with the package: CIE color matching functions,
# Bessell UBVRI and SDSS g/r. All are available offline.
BUNDLED_FILTER_IDS = (
    'CIE/1931_2deg.x',
    'CIE/1931_2deg.y',
    'CIE/1931_2deg.z',
    'Generic/Bessell.U',
    'Generic/Bessell.B',
    'Generic/Bessell.V',
    'Generic/Bessell.R',
    'Generic/Bessell.I',
    'SLOAN/SDSS.g',
    'SLOAN/SDSS.r',
)

# === FilterObjects Statistics Tests ===

# - general tests

def test_filter_edges():
    v = ac.Filter.get('Generic/Bessell.V')
    assert v.spectral_dist[0] == 0.
    assert v.spectral_dist[-1] == 0.

def test_filter_edges_extrapolated():
    v_filter = ac.Filter.get('Generic/Bessell.V')
    extrapolated_v = ac.get_spectrometry(v_filter, ac.visible_range)
    assert extrapolated_v.spectral_dist[0] == 0.
    assert extrapolated_v.spectral_dist[-1] == 0.

def test_filter_set_getitem():
    ubv = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B', 'Generic/Bessell.V')
    v = ac.Filter.get('Generic/Bessell.V')
    extracted = ubv[2]
    assert isinstance(extracted, ac.Filter)
    assert extracted == v
    assert hash(extracted) == hash(v)
    assert extracted.name == v.name
    np.testing.assert_array_equal(extracted.wavelength_nm, v.wavelength_nm)
    np.testing.assert_array_equal(extracted.spectral_dist, v.spectral_dist)
    assert ubv[-1] == v


def test_cached_filters_are_equal_hashable_copies():
    first = ac.Filter.get('Generic/Bessell.V')
    second = ac.Filter.get('Generic/Bessell.V')

    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert {first: 'V'}[second] == 'V'


def test_cached_filter_sets_are_equal_hashable_copies():
    first = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B', 'Generic/Bessell.V')
    second = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B', 'Generic/Bessell.V')

    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert {first: 'UBV'}[second] == 'UBV'

# - tests for mean values

def test_mean_nm_spectra(
    sun_calspec: ac.Spectrum,
    sun_tsis1_hsrs: ac.Spectrum,
    vega_calspec: ac.Spectrum,
):
    np.testing.assert_allclose(sun_calspec.mean_nm(), 858.101429, rtol=0.001)
    np.testing.assert_allclose(sun_tsis1_hsrs.mean_nm(), 945.233905, rtol=0.001)
    np.testing.assert_allclose(vega_calspec.mean_nm(), 523.296467, rtol=0.001)

def test_mean_nm_bessell_filters():
    # Reference for mean wavelengths: SVO Filter Profile Service
    u_filter = ac.Filter.get('Generic/Bessell.U')
    b_filter = ac.Filter.get('Generic/Bessell.B')
    v_filter = ac.Filter.get('Generic/Bessell.V')
    r_filter = ac.Filter.get('Generic/Bessell.R')
    i_filter = ac.Filter.get('Generic/Bessell.I')
    np.testing.assert_allclose(u_filter.mean_nm(), 360.507, rtol=0.001)
    np.testing.assert_allclose(b_filter.mean_nm(), 441.308, rtol=0.001)
    np.testing.assert_allclose(v_filter.mean_nm(), 551.210, rtol=0.001)
    np.testing.assert_allclose(r_filter.mean_nm(), 658.592, rtol=0.001)
    np.testing.assert_allclose(i_filter.mean_nm(), 805.988, rtol=0.001)

def test_mean_nm_bessell_filter_set(ubv_filterset: ac.FilterSet, vega_calspec: ac.Spectrum):
    # Reference for effective wavelengths: Bessell 2005, Table 1
    np.testing.assert_allclose(
        (ubv_filterset * vega_calspec).mean_nm(), [366.3, 436.1, 544.8], rtol=0.01
    )

def test_mean_nm_sloan_filters():
    # Reference for mean wavelengths: SVO Filter Profile Service
    g_filter = ac.Filter.get('SLOAN/SDSS.g').convert_for_photon_counter()
    r_filter = ac.Filter.get('SLOAN/SDSS.r').convert_for_photon_counter()
    np.testing.assert_allclose(g_filter.mean_nm(), 475.082, rtol=0.001)
    np.testing.assert_allclose(r_filter.mean_nm(), 620.429, rtol=0.001)

# - tests for standard deviations

def test_std_of_nm_spectra(
    sun_calspec: ac.Spectrum,
    sun_tsis1_hsrs: ac.Spectrum,
    vega_calspec: ac.Spectrum,
):
    np.testing.assert_allclose(sun_calspec.std_of_nm(), 469.40510, rtol=0.01)
    np.testing.assert_allclose(sun_tsis1_hsrs.std_of_nm(), 806.849085, rtol=0.01)
    np.testing.assert_allclose(vega_calspec.std_of_nm(), 473.17811, rtol=0.01)

def test_std_of_nm_filter(v_filter: ac.Filter):
    np.testing.assert_allclose(v_filter.std_of_nm(), 36.354015, rtol=0.01)

def test_std_of_nm_filter_set(ubv_filterset: ac.FilterSet):
    np.testing.assert_allclose(
        ubv_filterset.std_of_nm(), [21.932217, 35.816641, 36.354015], rtol=0.01
    )


# === Filter Unification Operation Tests ===

def test_filter_filter():
    actual = ac.Filter.get('Generic/Bessell.B') | ac.Filter.get('Generic/Bessell.V')
    desired = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V')
    np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
    np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)

def test_filterset_filter():
    actual = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V') | ac.Filter.get('Generic/Bessell.R')
    desired = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
    np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
    np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)

def test_filter_filterset():
    actual = ac.Filter.get('Generic/Bessell.B') | ac.FilterSet.get('Generic/Bessell.V', 'Generic/Bessell.R')
    desired = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
    np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
    np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)

def test_filterset_filterset():
    actual = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B') | ac.FilterSet.get('Generic/Bessell.V', 'Generic/Bessell.R')
    desired = ac.FilterSet.get('Generic/Bessell.U', 'Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
    np.testing.assert_allclose(actual.wavelength_nm, desired.wavelength_nm)
    np.testing.assert_allclose(actual.spectral_dist, desired.spectral_dist)


# === FilterSet Indexing and Round-trip Tests ===

def test_filter_set_roundtrip_is_exact():
    """ FilterSet[i] must be the same profile as Filter.get(i), not just close. """
    fs = ac.FilterSet.get(*BUNDLED_FILTER_IDS)
    for index, filter_id in enumerate(BUNDLED_FILTER_IDS):
        cached = ac.Filter.get(filter_id)
        extracted = fs[index]
        assert isinstance(extracted, ac.Filter)
        assert extracted == cached
        assert hash(extracted) == hash(cached)
        np.testing.assert_array_equal(extracted.wavelength_nm, cached.wavelength_nm)
        np.testing.assert_array_equal(extracted.spectral_dist, cached.spectral_dist)

def _filter_names(fs: ac.FilterSet) -> tuple[object, ...]:
    """ The public `name` attribute is typed as `object`; filter sets store tuples. """
    return cast(tuple[object, ...], fs.name)

def test_from_filters_does_not_renormalize():
    """ Packing normalized profiles must not change their last float64 bits. """
    fs = ac.FilterSet.get(*BUNDLED_FILTER_IDS)
    wavelength_nm = fs.wavelength_nm
    spectral_dist = fs.spectral_dist
    for index, filter_id in enumerate(BUNDLED_FILTER_IDS):
        cached = ac.Filter.get(filter_id)
        first = cast(int, cached.wavelength_nm[0])
        last = cast(int, cached.wavelength_nm[-1])
        mask = (wavelength_nm >= first) & (wavelength_nm <= last)
        np.testing.assert_array_equal(spectral_dist[mask, index], cached.spectral_dist)

def test_filter_set_slice_returns_filter_set(ubv_filterset: ac.FilterSet):
    subset = ubv_filterset[:2]
    assert isinstance(subset, ac.FilterSet)
    assert len(subset) == 2
    assert subset.name == _filter_names(ubv_filterset)[:2]
    expected = ac.FilterSet.from_filters([ubv_filterset[0], ubv_filterset[1]])
    assert subset == expected
    np.testing.assert_array_equal(subset.wavelength_nm, expected.wavelength_nm)
    np.testing.assert_array_equal(subset.spectral_dist, expected.spectral_dist)
    np.testing.assert_allclose(subset.integrate(), 1.)

def test_filter_set_slice_step_and_negative():
    fs = ac.FilterSet.get(*BUNDLED_FILTER_IDS)
    stepped = fs[::2]
    assert isinstance(stepped, ac.FilterSet)
    assert len(stepped) == 5
    assert stepped.name == _filter_names(fs)[::2]
    for position, source_index in enumerate(range(0, len(fs), 2)):
        assert stepped[position] == fs[source_index]

    reversed_set = fs[::-1]
    assert isinstance(reversed_set, ac.FilterSet)
    assert reversed_set.name == _filter_names(fs)[::-1]
    for position in range(len(fs)):
        assert reversed_set[position] == fs[-1 - position]

    tail = fs[-2:]
    assert isinstance(tail, ac.FilterSet)
    assert tail.name == _filter_names(fs)[-2:]

def test_filter_set_empty_slice():
    fs = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V')
    empty = fs[1:1]
    assert isinstance(empty, ac.FilterSet)
    assert len(empty) == 0
    assert empty.spectral_dist.shape == (0, 0)
    assert empty.wavelength_nm.size == 0

def test_filter_set_stub_index_returns_stub_filter():
    stub_set = ac.FilterSet.stub()
    profile = stub_set[0]
    assert isinstance(profile, ac.Filter)
    assert not profile.spectral_dist.any()

def test_filter_set_iteration_yields_filters():
    fs = ac.FilterSet.get(*BUNDLED_FILTER_IDS)
    profiles = list(fs)
    assert all(isinstance(profile, ac.Filter) for profile in profiles)
    assert [profile.name for profile in profiles] == list(_filter_names(fs))


# === Photon Counter Conversion Tests ===

SYNTHETIC_NM = np.arange(500, 526, 5, dtype=np.int64)
SYNTHETIC_SD = np.array([0., 0.2, 0.6, 1., 0.4, 0.])

def _synthetic_filter(name: object = 'Synthetic/Band') -> ac.Filter:
    """ A small asymmetric normalized profile for independent oracles. """
    return ac.Filter(SYNTHETIC_NM, SYNTHETIC_SD, name=name)

def test_photon_counter_profile_is_wavelength_weighted():
    """ The effective shape is lambda * T, not just a renormalized T. """
    profile = _synthetic_filter()
    converted = profile.convert_for_photon_counter()
    weighted = profile.spectral_dist * profile.wavelength_nm
    expected = weighted / (np.sum(weighted) * profile.nm_step)
    assert isinstance(converted, ac.Filter)
    np.testing.assert_allclose(converted.spectral_dist, expected)
    np.testing.assert_allclose(converted.integrate(), 1.)
    # A red-weighted profile has a longer effective wavelength.
    assert converted.mean_nm() > profile.mean_nm()

def test_photon_counter_photometry_matches_photon_integral():
    """ Photometry with a photon-counting profile equals the photon-weighted integral. """
    profile = _synthetic_filter()
    spectrum = ac.Spectrum(SYNTHETIC_NM, np.array([1., 2., 3., 4., 5., 6.]))
    value, error = ac.get_photometry(spectrum, profile.convert_for_photon_counter())
    weighted = profile.spectral_dist * profile.wavelength_nm
    expected = np.sum(spectrum.spectral_dist * weighted) / np.sum(weighted)
    assert error is None
    np.testing.assert_allclose(value, expected)

def test_photon_counter_roundtrip_returns_filter():
    profile = _synthetic_filter()
    roundtrip = profile.convert_for_photon_counter().convert_from_photon_spectral_density()
    assert isinstance(roundtrip, ac.Filter)
    np.testing.assert_allclose(roundtrip.spectral_dist, profile.spectral_dist, atol=1e-15)
    np.testing.assert_array_equal(roundtrip.wavelength_nm, profile.wavelength_nm)

def test_filter_set_photon_counter_per_column(ubv_filterset: ac.FilterSet):
    converted = ubv_filterset.convert_for_photon_counter()
    assert isinstance(converted, ac.FilterSet)
    np.testing.assert_allclose(converted.integrate(), np.ones(len(ubv_filterset)))
    weighted = ubv_filterset.spectral_dist * ubv_filterset.wavelength_nm[:, np.newaxis]
    expected = weighted / (np.sum(weighted, axis=0) * ubv_filterset.nm_step)
    np.testing.assert_allclose(converted.spectral_dist, expected)
    assert np.all(converted.mean_nm() > ubv_filterset.mean_nm())

def test_filter_set_photon_counter_on_square_grid():
    """ Wavelength weights must follow the spectral axis when it matches the filter count. """
    nm = np.array([400., 405., 410., 415.])
    profiles = np.array([
        [0., 0., 0., 0.],
        [0.2, 0.3, 0.4, 0.5],
        [0.4, 0.6, 0.8, 1.0],
        [0., 0., 0., 0.],
    ])
    filter_set = ac.FilterSet(nm, profiles)
    converted = filter_set.convert_for_photon_counter()
    expected = ac.FilterSet(nm, filter_set.spectral_dist * nm[:, np.newaxis])
    np.testing.assert_allclose(converted.spectral_dist, expected.spectral_dist)
    roundtrip = converted.convert_from_photon_spectral_density()
    np.testing.assert_allclose(roundtrip.spectral_dist, filter_set.spectral_dist)


def test_photospectral_conversion_uses_filter_profiles(ubv_filterset: ac.FilterSet):
    """ Photon/energy photometry conversion keeps the per-filter scale factors. """
    values = np.array([1., 2., 3.])
    photospectrum = ac.Photospectrum(ubv_filterset, values)
    profiles = ubv_filterset.spectral_dist
    wavelengths = ubv_filterset.wavelength_nm
    step = ubv_filterset.nm_step

    photon_scales = np.sum(profiles / wavelengths[:, np.newaxis], axis=0) * step
    photon_scales = photon_scales / photon_scales.mean()
    photon = photospectrum.convert_from_photon_spectral_density()
    np.testing.assert_allclose(photon.spectral_dist, values * photon_scales)

    frequency_scales = np.sum(profiles / wavelengths[:, np.newaxis]**2, axis=0) * step
    frequency_scales = frequency_scales / frequency_scales.mean()
    frequency = photospectrum.convert_from_energy_spectral_density_per_frequency()
    np.testing.assert_allclose(frequency.spectral_dist, values * frequency_scales)


def test_photospectral_conversion_on_square_spatial(ubv_filterset: ac.FilterSet):
    """ Per-filter scales must not flip to pixels when spatial size equals the filter count. """
    photospectral_set = ac.PhotospectralSet(ubv_filterset, np.ones((3, 3)))
    profiles = ubv_filterset.spectral_dist
    wavelengths = ubv_filterset.wavelength_nm
    scales = np.sum(profiles / wavelengths[:, np.newaxis], axis=0) * ubv_filterset.nm_step
    scales = scales / scales.mean()
    converted = photospectral_set.convert_from_photon_spectral_density()
    expected = np.broadcast_to(scales[:, np.newaxis], (3, 3))
    np.testing.assert_allclose(converted.spectral_dist, expected)


# === Filter Profile Validation Tests ===

# Stiles & Burch (1955) 2-degree R color-matching function sampled at 10 nm.
# Source: CVRL, "Stiles & Burch (1955) 2-deg, RGB CMFs"
# (Wyszecki & Stiles 1982, Table I(5.5.3), corrected as described by Stiles &
# Burch 1959). The function has a pronounced negative lobe between 450 and
# 520 nm, reaching about -0.30 near 505 nm. This is the motivating example for
# accepting negative throughput while still requiring a positive total integral.
STILES_BURCH_1955_R_NM = np.array([
    390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
    510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620,
    630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730,
], dtype=np.int64)

STILES_BURCH_1955_R = np.array([
    0.0018397, 0.0096264, 0.030803, 0.051662, 0.044287, 0.014763,
    -0.02913, -0.096224, -0.17486, -0.2378, -0.27727, -0.295,
    -0.26759, -0.14768, 0.10614, 0.41976, 0.79004, 1.2283,
    1.7476, 2.2724, 2.6725, 2.8717, 2.7601, 2.3743,
    1.8145, 1.2543, 0.78642, 0.4432, 0.23455, 0.12086,
    0.06026, 0.028114, 0.013305, 0.0065177, 0.0031742,
])


class TestFilterProfileValidation:
    """ Filter construction must not accept profiles that cannot be normalized. """

    def test_zero_profile_is_rejected(self):
        with pytest.raises(NonPositiveFilterIntegralError, match='positive'):
            _ = ac.Filter([500, 505, 510], [0., 0., 0.], name='zero')

    def test_zero_column_in_filter_set_is_rejected(self):
        profiles = np.array([
            [0., 0.],
            [1., 0.],
            [0., 0.],
        ])
        with pytest.raises(NonPositiveFilterIntegralError, match='positive'):
            _ = ac.FilterSet([500, 505, 510], profiles, name='zero column')

    def test_axes_error_precedes_non_finite_profile(self):
        with pytest.raises(InconsistentAxesError):
            _ = ac.Filter([500, 505], [1., np.nan, 3.])

    @pytest.mark.parametrize('wavelengths', [
        [500, np.nan, 510],
        [np.nan, np.inf],
    ])
    def test_non_finite_wavelengths_are_rejected(self, wavelengths: list[float]):
        values = np.ones(len(wavelengths))
        with pytest.raises(NonFiniteFilterWavelengthError, match='non-finite'):
            _ = ac.Filter(wavelengths, values, name='bad axis')
        with pytest.raises(NonFiniteFilterWavelengthError, match='non-finite'):
            _ = ac.FilterSet(wavelengths, np.ones((len(wavelengths), 2)), name='bad axis')

    def test_negative_integral_is_rejected(self):
        with pytest.raises(NonPositiveFilterIntegralError, match='positive'):
            _ = ac.Filter([500, 505, 510], [-0.2, -1., -0.2], name='negative area')

    @pytest.mark.parametrize('non_finite', [np.nan, np.inf, -np.inf])
    def test_non_finite_filter_is_rejected(self, non_finite: float):
        with pytest.raises(NonFiniteFilterProfileError, match='non-finite'):
            _ = ac.Filter([500, 505, 510], [0., non_finite, 1.], name='bad')

    @pytest.mark.parametrize('non_finite', [np.nan, np.inf, -np.inf])
    def test_non_finite_filter_set_is_rejected(self, non_finite: float):
        profiles = np.array([
            [0., 1.],
            [1., non_finite],
            [0., 0.],
        ])
        with pytest.raises(NonFiniteFilterProfileError, match='non-finite'):
            _ = ac.FilterSet([500, 505, 510], profiles, name='bad set')

    def test_zero_edged_empty_filter_set_stays_empty(self):
        empty = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V')[1:1]
        result = empty.zero_edged()
        assert isinstance(result, ac.FilterSet)
        assert result.spectral_size == 0

    def test_stub_bypasses_validation(self):
        stub = ac.Filter.stub()
        stub_set = ac.FilterSet.stub()
        assert isinstance(stub, ac.Filter)
        assert isinstance(stub_set, ac.FilterSet)
        np.testing.assert_array_equal(stub.spectral_dist, [0.])

    def test_stub_conversions_remain_zero(self):
        stub = ac.Filter.stub()
        for converted in (
            stub.convert_for_photon_counter(),
            stub.convert_from_photon_spectral_density(),
            stub.convert_from_energy_spectral_density_per_frequency(),
        ):
            assert isinstance(converted, ac.Filter)
            assert not converted.spectral_dist.any()
        assert not ac.FilterSet.stub().convert_for_photon_counter().spectral_dist.any()
        # A computed zero result is degenerate, not an invalid input profile.
        monochromatic = ac.Filter.monochromatic(555., intensity=0.)
        assert not monochromatic.spectral_dist.any()

    def test_stiles_burch_negative_integral_excerpt_is_rejected(self):
        # The dip alone has a negative integral even though the full function
        # does not: rejection is about the total area, not about negative points.
        excerpt = (STILES_BURCH_1955_R_NM >= 440) & (STILES_BURCH_1955_R_NM <= 530)
        with pytest.raises(NonPositiveFilterIntegralError, match='positive'):
            _ = ac.Filter(
                STILES_BURCH_1955_R_NM[excerpt],
                STILES_BURCH_1955_R[excerpt],
                name='Stiles & Burch (1955) 2-deg R excerpt',
            )

    def test_stiles_burch_red_negative_lobe_is_preserved(self):
        profile = ac.Filter(
            STILES_BURCH_1955_R_NM,
            STILES_BURCH_1955_R,
            name='Stiles & Burch (1955) 2-deg R',
        )

        assert np.isclose(profile.integrate(), 1.)
        assert profile.is_zero_edged()
        # Normalization changes the scale, not the sign or the position of the dip.
        dip_index = int(np.argmin(profile.spectral_dist))
        assert profile.spectral_dist[dip_index] < 0
        assert 490 <= profile.wavelength_nm[dip_index] <= 515
        assert profile.spectral_dist[profile.wavelength_nm == 590][0] > 0

    def test_bundled_filters_pass_validation(self):
        for filter_id in BUNDLED_FILTER_IDS:
            profile = ac.Filter.get(filter_id)
            assert np.all(np.isfinite(profile.spectral_dist))
            assert profile.integrate() > 0
