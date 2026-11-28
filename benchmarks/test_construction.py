"""
Benchmarks for object construction and normalization.
"""

import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import astrocolor as ac

pytestmark = pytest.mark.benchmark(group='construction')


@pytest.fixture(scope='session')
def nm() -> npt.NDArray[np.floating]:
    return np.arange(400., 701., 5.)


@pytest.fixture(scope='session')
def values_2d(nm: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    values = np.linspace(1., 2., nm.size)
    return np.repeat(values[:, np.newaxis], 100, axis=1)


@pytest.fixture(scope='session')
def values_cube(
    values_2d: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.repeat(values_2d[:, :, np.newaxis], 100, axis=2)


@pytest.fixture(scope='session')
def spectral_set(
    nm: npt.NDArray[np.floating],
    values_2d: npt.NDArray[np.floating],
) -> ac.SpectralSet:
    return ac.SpectralSet(nm, values_2d)


@pytest.fixture(scope='session')
def filter_profiles(v_filter: ac.Filter) -> npt.NDArray[np.floating]:
    return np.repeat(v_filter.spectral_dist[:, np.newaxis], 3, axis=1)


@pytest.fixture(scope='session')
def color_point() -> npt.NDArray[np.floating]:
    return np.array([0.5, 0.25, 0.1])


@pytest.fixture(scope='session')
def color_line(
    color_point: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.repeat(color_point[:, np.newaxis], 100, axis=1)


@pytest.fixture(scope='session')
def color_image(
    color_line: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.repeat(color_line[:, :, np.newaxis], 100, axis=2)


def test_spectrum_construction(
    benchmark: BenchmarkFixture,
    nm: npt.NDArray[np.floating],
    values_2d: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.Spectrum, nm, values_2d[:, 0])


def test_spectral_set_construction(
    benchmark: BenchmarkFixture,
    nm: npt.NDArray[np.floating],
    values_2d: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.SpectralSet, nm, values_2d)


def test_spectral_cube_construction(
    benchmark: BenchmarkFixture,
    nm: npt.NDArray[np.floating],
    values_cube: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.SpectralCube, nm, values_cube)


def test_filter_construction(
    benchmark: BenchmarkFixture,
    v_filter: ac.Filter,
) -> None:
    benchmark(ac.Filter, v_filter.wavelength_nm, v_filter.spectral_dist)


def test_filter_set_construction(
    benchmark: BenchmarkFixture,
    v_filter: ac.Filter,
    filter_profiles: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.FilterSet, v_filter.wavelength_nm, filter_profiles)


def test_color_point_construction(
    benchmark: BenchmarkFixture,
    color_point: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.ColorPoint, color_point)


def test_color_line_construction(
    benchmark: BenchmarkFixture,
    color_line: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.ColorLine, color_line)


def test_color_image_construction(
    benchmark: BenchmarkFixture,
    color_image: npt.NDArray[np.floating],
) -> None:
    benchmark(ac.ColorImage, color_image)


def test_normalize_spectral_set(
    benchmark: BenchmarkFixture,
    spectral_set: ac.SpectralSet,
) -> None:
    benchmark(spectral_set.normalized)
