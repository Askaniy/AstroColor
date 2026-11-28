"""
Benchmarks for cubes, spectral sets, photometry and color images.
"""

import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import astrocolor as ac

pytestmark = pytest.mark.benchmark(group='2D/3D')

WIDTH = 100
HEIGHT = 100


@pytest.fixture(scope='session')
def nm() -> npt.NDArray[np.floating]:
    """A 400-700 nm grid with 5 nm steps."""
    return np.arange(400., 701., 5.)


@pytest.fixture(scope='session')
def spectral_cube(nm: npt.NDArray[np.floating]) -> ac.SpectralCube:
    rng = np.random.default_rng(0)
    values = rng.random((nm.size, WIDTH, HEIGHT)) + 0.1
    return ac.SpectralCube(nm, values)


@pytest.fixture(scope='session')
def spectral_set(nm: npt.NDArray[np.floating]) -> ac.SpectralSet:
    rng = np.random.default_rng(1)
    values = rng.random((nm.size, WIDTH)) + 0.1
    return ac.SpectralSet(nm, values)


@pytest.fixture(scope='session')
def photospectral_cube(ubv_filter_set: ac.FilterSet) -> ac.PhotospectralCube:
    rng = np.random.default_rng(2)
    values = rng.random((len(ubv_filter_set), WIDTH, HEIGHT)) + 0.1
    return ac.PhotospectralCube(ubv_filter_set, values)


@pytest.fixture(scope='session')
def color_image(spectral_cube: ac.SpectralCube) -> ac.ColorImage:
    return ac.ColorImage.from_spectral_data(spectral_cube)


@pytest.fixture(scope='session')
def srgb_color_image(
    color_image: ac.ColorImage,
    srgb: ac.ColorSystem,
) -> ac.ColorImage:
    return color_image.to_color_system(srgb)


def test_add_cubes(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(lambda: spectral_cube + spectral_cube)


def test_multiply_cube_by_scalar(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(lambda: spectral_cube * 2)


def test_divide_cube_by_scalar(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(lambda: spectral_cube / 2)


def test_divide_cube_by_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(lambda: spectral_cube / spectral_cube)


def test_normalize_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(spectral_cube.normalized)


def test_flatten_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(spectral_cube.flatten)


def test_downscale_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(lambda: spectral_cube.downscale(1000))


def test_add_spectral_sets(
    benchmark: BenchmarkFixture,
    spectral_set: ac.SpectralSet,
) -> None:
    benchmark(lambda: spectral_set + spectral_set)


def test_photometry_v_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
    v_filter: ac.Filter,
) -> None:
    benchmark(lambda: ac.get_photometry(spectral_cube, v_filter))


def test_photometry_ubv_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
    ubv_filter_set: ac.FilterSet,
) -> None:
    benchmark(lambda: ac.get_photometry(spectral_cube, ubv_filter_set))


def test_photometry_v_set(
    benchmark: BenchmarkFixture,
    spectral_set: ac.SpectralSet,
    v_filter: ac.Filter,
) -> None:
    benchmark(lambda: ac.get_photometry(spectral_set, v_filter))


def test_reconstruct_photospectral_cube(
    benchmark: BenchmarkFixture,
    photospectral_cube: ac.PhotospectralCube,
    ubv_filter_set: ac.FilterSet,
) -> None:
    wavelengths = ubv_filter_set.wavelength_nm
    benchmark(lambda: ac.get_spectrometry(photospectral_cube, wavelengths))


def test_color_line_from_spectral_set(
    benchmark: BenchmarkFixture,
    spectral_set: ac.SpectralSet,
) -> None:
    benchmark(lambda: ac.ColorLine.from_spectral_data(spectral_set))


def test_color_image_from_spectral_cube(
    benchmark: BenchmarkFixture,
    spectral_cube: ac.SpectralCube,
) -> None:
    benchmark(lambda: ac.ColorImage.from_spectral_data(spectral_cube))


def test_color_image_to_srgb(
    benchmark: BenchmarkFixture,
    color_image: ac.ColorImage,
    srgb: ac.ColorSystem,
) -> None:
    benchmark(lambda: color_image.to_color_system(srgb))


def test_render_color_image(
    benchmark: BenchmarkFixture,
    srgb_color_image: ac.ColorImage,
) -> None:
    benchmark(srgb_color_image.render)
