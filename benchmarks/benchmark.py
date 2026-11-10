"""
Performance benchmarks for AstroColor.

Run from the repository root:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --suite 2d3d
    uv run python benchmarks/benchmark.py --suite uncertainty
"""

import argparse
import time
from collections.abc import Callable
from statistics import median
from typing import cast

import numpy as np

import astrocolor as ac
from astrocolor.config import Config


def benchmark(
    function: Callable[[], object],
    repeats: int
) -> tuple[float, float]:
    """ Returns (best, median) runtime in milliseconds after one warm-up call. """
    _ = function()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = function()
        timings.append((time.perf_counter() - start) * 1000)
    return min(timings), median(timings)


def run_2d3d(repeats: int, seed: int) -> None:
    Config.ignore_uncertainty_for_cubes = True
    nm = np.arange(400., 701., 5.)
    width = height = 100
    rng = np.random.default_rng(seed)
    cube = ac.SpectralCube(nm, rng.random((nm.size, width, height)) + 0.1)
    spectral_set = ac.SpectralSet(nm, rng.random((nm.size, width)) + 0.1)
    v_filter = ac.Filter.get('Generic/Bessell.V')
    ubv_filter_set = ac.FilterSet.get(
        'Generic/Bessell.U', 'Generic/Bessell.B', 'Generic/Bessell.V'
    )
    photospectral_cube = ac.PhotospectralCube(
        ubv_filter_set, rng.random((len(ubv_filter_set), width, height)) + 0.1
    )

    workload: dict[str, Callable[[], object]] = {
        'cube + cube': lambda: cube + cube,
        'cube * scalar': lambda: cube * 2,
        'cube normalized': cube.normalized,
        'cube flatten': cube.flatten,
        'cube downscale(1000)': lambda: cube.downscale(1000),
        'spectral set + set': lambda: spectral_set + spectral_set,
        'photometry V (cube)': lambda: ac.get_photometry(cube, v_filter),
        'photometry UBV (cube)': lambda: ac.get_photometry(cube, ubv_filter_set),
        'photometry V (set)': lambda: ac.get_photometry(spectral_set, v_filter),
        'reconstruct PhotospectralCube': lambda: ac.get_spectrometry(
            photospectral_cube,
            ubv_filter_set.wavelength_nm,
            attach_photospectral_obj=False,
        ),
        'ColorImage(cube)': lambda: ac.ColorImage.from_spectral_data(cube),
    }

    print(f'2D/3D workload: cube {nm.size} × {width} × {height}, {repeats} repeats')
    print(f'{"operation":36s} {"best, ms":>10s} {"median, ms":>12s}')
    print('-' * 60)
    for name, function in workload.items():
        best, med = benchmark(function, repeats)
        print(f'{name:36s} {best:10.3f} {med:12.3f}')


def run_uncertainty(repeats: int) -> None:
    nm = np.arange(400., 1001., 5.)
    values = np.linspace(1., 2., nm.size)
    std = np.full(nm.size, 0.1)
    covariance = np.diag(std**2)

    compact = ac.Spectrum(nm, values, std)
    full = ac.Spectrum(nm, values, covariance)
    v_filter = ac.Filter.get('Generic/Bessell.V')

    workload: dict[str, tuple[Callable[[], object], Callable[[], object]]] = {
        'scalar multiply': (lambda: compact * 2, lambda: full * 2),
        'addition': (lambda: compact + compact, lambda: full + full),
        'normalize': (compact.normalized, full.normalized),
        'photometry V': (
            lambda: ac.get_photometry(compact, v_filter),
            lambda: ac.get_photometry(full, v_filter),
        ),
    }

    print(f'Uncertainty: spectrum length {nm.size}, {repeats} repeats')
    print(f'{"operation":36s} {"compact, ms":>12s} {"full, ms":>10s}')
    print('-' * 62)
    for name, (compact_function, full_function) in workload.items():
        compact_best, _ = benchmark(compact_function, repeats)
        full_best, _ = benchmark(full_function, repeats)
        print(f'{name:36s} {compact_best:12.3f} {full_best:10.3f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--suite',
        choices=('all', '2d3d', 'uncertainty'),
        default='all',
        help='benchmark suite to run (default: all)',
    )
    _ = parser.add_argument('--repeats', type=int, default=7, help='timed repeats per operation')
    _ = parser.add_argument('--seed', type=int, default=0, help='random seed for synthetic data')
    args = parser.parse_args()
    suite = cast(str, args.suite)
    repeats = cast(int, args.repeats)
    seed = cast(int, args.seed)

    if suite in ('all', '2d3d'):
        run_2d3d(repeats, seed)
    if suite == 'all':
        print()
    if suite in ('all', 'uncertainty'):
        run_uncertainty(repeats)


if __name__ == '__main__':
    main()
