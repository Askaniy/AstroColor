from typing import cast

import numpy as np

import astrocolor as ac

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
