"""
Benchmarks for uncertainty propagation: compact std versus full covariance.
"""

import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import astrocolor as ac

pytestmark = pytest.mark.benchmark(group='uncertainty')

REPRESENTATIONS = pytest.mark.parametrize(
    'representation',
    ['compact', 'full'],
    ids=['std', 'covariance'],
)


@pytest.fixture(scope='session')
def nm() -> npt.NDArray[np.floating]:
    return np.arange(400., 1001., 5.)


@pytest.fixture(scope='session')
def spectra(nm: npt.NDArray[np.floating]) -> dict[str, ac.Spectrum]:
    values = np.linspace(1., 2., nm.size)
    std = np.full(nm.size, 0.1)
    covariance = np.diag(std**2)
    return {
        'compact': ac.Spectrum(nm, values, std),
        'full': ac.Spectrum(nm, values, covariance),
    }


@pytest.fixture(scope='session')
def color_values() -> npt.NDArray[np.floating]:
    return np.array([0.2, 0.3, 0.4])


@pytest.fixture(scope='session')
def color_uncertainties() -> dict[str, npt.NDArray[np.floating]]:
    std = np.array([0.01, 0.02, 0.03])
    return {
        'compact': std,
        'full': np.diag(std**2),
    }


@pytest.fixture(scope='session')
def color_points(
    color_values: npt.NDArray[np.floating],
    color_uncertainties: dict[str, npt.NDArray[np.floating]],
) -> dict[str, ac.ColorPoint]:
    return {
        name: ac.ColorPoint(color_values, uncertainty)
        for name, uncertainty in color_uncertainties.items()
    }


@REPRESENTATIONS
def test_scalar_multiply(
    benchmark: BenchmarkFixture,
    spectra: dict[str, ac.Spectrum],
    representation: str,
) -> None:
    spectrum = spectra[representation]
    benchmark(lambda: spectrum * 2)


@REPRESENTATIONS
def test_scalar_division(
    benchmark: BenchmarkFixture,
    spectra: dict[str, ac.Spectrum],
    representation: str,
) -> None:
    spectrum = spectra[representation]
    benchmark(lambda: spectrum / 2)


@REPRESENTATIONS
def test_addition(
    benchmark: BenchmarkFixture,
    spectra: dict[str, ac.Spectrum],
    representation: str,
) -> None:
    spectrum = spectra[representation]
    benchmark(lambda: spectrum + spectrum)


@REPRESENTATIONS
def test_normalize(
    benchmark: BenchmarkFixture,
    spectra: dict[str, ac.Spectrum],
    representation: str,
) -> None:
    benchmark(spectra[representation].normalized)


@REPRESENTATIONS
def test_photometry_v(
    benchmark: BenchmarkFixture,
    spectra: dict[str, ac.Spectrum],
    v_filter: ac.Filter,
    representation: str,
) -> None:
    spectrum = spectra[representation]
    benchmark(lambda: ac.get_photometry(spectrum, v_filter))


@REPRESENTATIONS
def test_color_point_construction(
    benchmark: BenchmarkFixture,
    color_values: npt.NDArray[np.floating],
    color_uncertainties: dict[str, npt.NDArray[np.floating]],
    representation: str,
) -> None:
    uncertainty = color_uncertainties[representation]
    benchmark(ac.ColorPoint, color_values, uncertainty)


@REPRESENTATIONS
def test_color_point_to_srgb(
    benchmark: BenchmarkFixture,
    color_points: dict[str, ac.ColorPoint],
    srgb: ac.ColorSystem,
    representation: str,
) -> None:
    color_point = color_points[representation]
    benchmark(lambda: color_point.to_color_system(srgb))


@REPRESENTATIONS
def test_render_color_point(
    benchmark: BenchmarkFixture,
    color_points: dict[str, ac.ColorPoint],
    representation: str,
) -> None:
    benchmark(color_points[representation].render)
