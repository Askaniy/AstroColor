"""
Benchmarks for algebraic helpers on batched and covariance data.
"""

import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from astrocolor.algebra import div_value
from astrocolor.auxiliary import is_diagonal

pytestmark = pytest.mark.benchmark(group='algebra')


@pytest.fixture(scope='session')
def values_2d() -> npt.NDArray[np.floating]:
    values = np.linspace(1., 2., 61)
    return np.repeat(values[:, np.newaxis], 100, axis=1)


@pytest.fixture(scope='session')
def covariance_3() -> npt.NDArray[np.floating]:
    return np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]) * 1e-12


@pytest.fixture(scope='session')
def covariance_61() -> npt.NDArray[np.floating]:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(61, 61))
    return (matrix @ matrix.T / 61 + np.eye(61) * 0.1) * 1e-12


@pytest.fixture(scope='session')
def covariance_61x100(
    covariance_61: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.repeat(covariance_61[:, :, np.newaxis], 100, axis=2)


def test_div_value(
    benchmark: BenchmarkFixture,
    values_2d: npt.NDArray[np.floating],
) -> None:
    benchmark(div_value, values_2d, values_2d + 1.)


def test_is_diagonal_3x3(
    benchmark: BenchmarkFixture,
    covariance_3: npt.NDArray[np.floating],
) -> None:
    benchmark(is_diagonal, covariance_3)


def test_is_diagonal_61x61(
    benchmark: BenchmarkFixture,
    covariance_61: npt.NDArray[np.floating],
) -> None:
    benchmark(is_diagonal, covariance_61)


def test_is_diagonal_61x61x100(
    benchmark: BenchmarkFixture,
    covariance_61x100: npt.NDArray[np.floating],
) -> None:
    benchmark(is_diagonal, covariance_61x100)
