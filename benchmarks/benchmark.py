"""
Performance benchmarks for AstroColor.

Run from the repository root:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --suite 2d3d
    uv run python benchmarks/benchmark.py --suite construction
    uv run python benchmarks/benchmark.py --suite uncertainty

Each operation is measured in `--rounds` independent rounds of `--repeats`
samples. A sample times a batch of calls, so fast operations are not dominated
by clock overhead; the batch is chosen to last at least `--target-ms`.

Save a baseline and compare a later run against it in one command:
    uv run python benchmarks/benchmark.py --json temp/before.json
    uv run python benchmarks/benchmark.py --baseline temp/before.json
"""

import argparse
import json
import time
from collections.abc import Callable
from datetime import datetime
from math import ceil, isnan
from pathlib import Path
from statistics import median
from typing import cast

import numpy as np

import astrocolor as ac
from astrocolor.algebra import div_value
from astrocolor.config import Config

Results = dict[str, dict[str, dict[str, float]]]

SUITE_METRICS: dict[str, dict[str, str]] = {
    '2d3d': {'best': 'best, ms', 'median': 'median, ms'},
    'construction': {'best': 'best, ms', 'median': 'median, ms'},
    'uncertainty': {'compact': 'compact, ms', 'full': 'full, ms'},
}


def benchmark(
    function: Callable[[], object],
    repeats: int,
    rounds: int = 1,
    target_ms: float = 5.,
) -> tuple[float, float]:
    """
    Returns (best, median) runtime in milliseconds after warm-up.

    `repeats` samples are taken per round and `rounds` independent rounds are
    summarized by the median of their medians. Each sample times a batch of
    calls so that fast operations do not drown in clock overhead: the batch is
    chosen to last at least `target_ms` milliseconds. A non-positive
    `target_ms` disables batching.
    """
    _ = function()
    start = time.perf_counter()
    _ = function()
    single_ms = (time.perf_counter() - start) * 1000
    batch = 1
    if target_ms > 0 and single_ms > 0:
        batch = max(1, min(10_000, ceil(target_ms / single_ms)))
    round_bests: list[float] = []
    round_medians: list[float] = []
    for _ in range(rounds):
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(batch):
                _ = function()
            timings.append((time.perf_counter() - start) * 1000 / batch)
        round_bests.append(min(timings))
        round_medians.append(median(timings))
    return min(round_bests), median(round_medians)


def run_2d3d(
    repeats: int,
    rounds: int,
    target_ms: float,
    seed: int,
) -> dict[str, dict[str, float]]:
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
    srgb = ac.default_color_system
    color_image = ac.ColorImage.from_spectral_data(cube)
    color_image_srgb = color_image.to_color_system(srgb)

    workload: dict[str, Callable[[], object]] = {
        'cube + cube': lambda: cube + cube,
        'cube * scalar': lambda: cube * 2,
        'cube / scalar': lambda: cube / 2,
        'cube / cube': lambda: cube / cube,
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
        ),
        'ColorLine(set)': lambda: ac.ColorLine.from_spectral_data(spectral_set),
        'ColorImage(cube)': lambda: ac.ColorImage.from_spectral_data(cube),
        'ColorImage to sRGB': lambda: color_image.to_color_system(srgb),
        'ColorImage render': color_image_srgb.render,
    }

    print(f'2D/3D workload: cube {nm.size} × {width} × {height}, {repeats} samples × {rounds} rounds')
    print(f'{"operation":36s} {"best, ms":>10s} {"median, ms":>12s}')
    print('-' * 60)
    results: dict[str, dict[str, float]] = {}
    for name, function in workload.items():
        best, med = benchmark(function, repeats, rounds, target_ms)
        results[name] = {'best': best, 'median': med}
        print(f'{name:36s} {best:10.3f} {med:12.3f}')
    return results


def run_construction(
    repeats: int,
    rounds: int,
    target_ms: float,
) -> dict[str, dict[str, float]]:
    """ Times object construction and the hot division/normalization paths. """
    nm = np.arange(400., 701., 5.)
    values = np.linspace(1., 2., nm.size)
    values_2d = np.repeat(values[:, np.newaxis], 100, axis=1)
    values_cube = np.repeat(values_2d[:, :, np.newaxis], 100, axis=2)
    spectral_set = ac.SpectralSet(nm, values_2d)
    v_filter = ac.Filter.get('Generic/Bessell.V')
    filter_nm = v_filter.wavelength_nm
    filter_profiles = np.repeat(v_filter.spectral_dist[:, np.newaxis], 3, axis=1)
    color_point = np.array([0.5, 0.25, 0.1])
    color_line = np.repeat(color_point[:, np.newaxis], 100, axis=1)
    color_image = np.repeat(color_line[:, :, np.newaxis], 100, axis=2)

    workload: dict[str, Callable[[], object]] = {
        'Spectrum construction': lambda: ac.Spectrum(nm, values),
        'SpectralSet construction': lambda: ac.SpectralSet(nm, values_2d),
        'SpectralCube construction': lambda: ac.SpectralCube(nm, values_cube),
        'Filter construction': lambda: ac.Filter(filter_nm, v_filter.spectral_dist),
        'FilterSet construction': lambda: ac.FilterSet(filter_nm, filter_profiles),
        'ColorPoint construction': lambda: ac.ColorPoint(color_point),
        'ColorLine construction': lambda: ac.ColorLine(color_line),
        'ColorImage construction': lambda: ac.ColorImage(color_image),
        'SpectralSet normalized': spectral_set.normalized,
        'div_value (61×100)': lambda: div_value(values_2d, values_2d + 1.),
    }

    print(f'Construction and division: spectra {nm.size} points, filters {filter_nm.size} points, 100×100 spatial, color 3×100×100, {repeats} samples × {rounds} rounds')
    print(f'{"operation":36s} {"best, ms":>10s} {"median, ms":>12s}')
    print('-' * 60)
    results: dict[str, dict[str, float]] = {}
    for name, function in workload.items():
        best, med = benchmark(function, repeats, rounds, target_ms)
        results[name] = {'best': best, 'median': med}
        print(f'{name:36s} {best:10.3f} {med:12.3f}')
    return results


def run_uncertainty(
    repeats: int,
    rounds: int,
    target_ms: float,
) -> dict[str, dict[str, float]]:
    nm = np.arange(400., 1001., 5.)
    values = np.linspace(1., 2., nm.size)
    std = np.full(nm.size, 0.1)
    covariance = np.diag(std**2)

    compact = ac.Spectrum(nm, values, std)
    full = ac.Spectrum(nm, values, covariance)
    v_filter = ac.Filter.get('Generic/Bessell.V')
    srgb = ac.default_color_system
    color_values = np.array([0.2, 0.3, 0.4])
    color_std = np.array([0.01, 0.02, 0.03])
    color_covariance = np.diag(color_std**2)
    compact_color = ac.ColorPoint(color_values, color_std)
    full_color = ac.ColorPoint(color_values, color_covariance)

    workload: dict[str, tuple[Callable[[], object], Callable[[], object]]] = {
        'scalar multiply': (lambda: compact * 2, lambda: full * 2),
        'scalar division': (lambda: compact / 2, lambda: full / 2),
        'addition': (lambda: compact + compact, lambda: full + full),
        'normalize': (compact.normalized, full.normalized),
        'photometry V': (
            lambda: ac.get_photometry(compact, v_filter),
            lambda: ac.get_photometry(full, v_filter),
        ),
        'ColorPoint construction': (
            lambda: ac.ColorPoint(color_values, color_std),
            lambda: ac.ColorPoint(color_values, color_covariance),
        ),
        'ColorPoint to sRGB': (
            lambda: compact_color.to_color_system(srgb),
            lambda: full_color.to_color_system(srgb),
        ),
        'ColorPoint render': (compact_color.render, full_color.render),
    }

    print(f'Uncertainty: spectrum length {nm.size}, color point (3 channels), {repeats} samples × {rounds} rounds')
    print(f'{"operation":36s} {"compact, ms":>12s} {"full, ms":>10s}')
    print('-' * 62)
    results: dict[str, dict[str, float]] = {}
    for name, (compact_function, full_function) in workload.items():
        compact_best, _ = benchmark(compact_function, repeats, rounds, target_ms)
        full_best, _ = benchmark(full_function, repeats, rounds, target_ms)
        results[name] = {'compact': compact_best, 'full': full_best}
        print(f'{name:36s} {compact_best:12.3f} {full_best:10.3f}')
    return results


def write_results(
    path: Path,
    results: Results,
    *,
    repeats: int,
    rounds: int,
    target_ms: float,
    seed: int,
) -> None:
    """ Writes measurements and run settings to a JSON file. """
    payload = {
        'meta': {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'repeats': repeats,
            'rounds': rounds,
            'target_ms': target_ms,
            'seed': seed,
        },
        'suites': results,
    }
    _ = path.write_text(json.dumps(payload, indent=2) + '\n')


def read_results(path: Path) -> Results:
    """ Loads measurements written by `write_results()`. """
    raw = cast(object, json.loads(path.read_text()))
    if not isinstance(raw, dict):
        raise SystemExit(f'{path}: expected a JSON object written by --json')
    suites = cast(dict[str, object], raw).get('suites')
    if not isinstance(suites, dict):
        raise SystemExit(f'{path}: no "suites" object found; was it written by --json?')
    return cast(Results, suites)


def compare_results(
    current: Results,
    baseline: Results,
    baseline_path: Path,
) -> None:
    """ Prints current timings next to their baseline and the ratio. """
    print(f'Comparison with {baseline_path} (ratio < 1 means the current run is faster):')
    header = f'{"suite / operation":44s} {"metric":12s} {"baseline":>9s} {"current":>9s} {"ratio":>8s}'
    print(header)
    print('-' * len(header))
    matched = 0
    unmatched: list[str] = []
    for suite, rows in current.items():
        labels = SUITE_METRICS[suite]
        baseline_rows = baseline.get(suite, {})
        for operation, metrics in rows.items():
            baseline_metrics = baseline_rows.get(operation)
            if baseline_metrics is None:
                unmatched.append(f'{suite} / {operation}')
                continue
            for key, label in labels.items():
                value = metrics.get(key)
                previous = baseline_metrics.get(key)
                if value is None or previous is None:
                    continue
                if previous == 0:
                    ratio_text = '    inf'
                else:
                    ratio = value / previous
                    ratio_text = '    nan' if isnan(ratio) else f'{ratio:7.2f}'
                name = f'{suite} / {operation}'
                print(f'{name:44s} {label:12s} {previous:9.4f} {value:9.4f} {ratio_text}x')
                matched += 1
    if not matched:
        print('No overlapping measurements: the baseline has no matching suites or operations.')
    if unmatched:
        print(f'Only in the current run: {len(unmatched)} operation(s).')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--suite',
        choices=('all', '2d3d', 'construction', 'uncertainty'),
        default='all',
        help='benchmark suite to run (default: all)',
    )
    _ = parser.add_argument(
        '--repeats', type=int, default=7,
        help='samples per round (default: 7)',
    )
    _ = parser.add_argument(
        '--rounds', type=int, default=3,
        help='independent measurement rounds per operation (default: 3)',
    )
    _ = parser.add_argument(
        '--target-ms', type=float, default=5.,
        help='minimum duration of one measured sample in ms; fast operations are batched (default: 5)',
    )
    _ = parser.add_argument('--seed', type=int, default=0, help='random seed for synthetic data')
    _ = parser.add_argument(
        '--json', type=Path, default=None,
        help='write results to this JSON file',
    )
    _ = parser.add_argument(
        '--baseline', type=Path, default=None,
        help='compare with a JSON file written by --json (use the same machine)',
    )
    args = parser.parse_args()
    suite = cast(str, args.suite)
    repeats = cast(int, args.repeats)
    rounds = cast(int, args.rounds)
    target_ms = cast(float, args.target_ms)
    seed = cast(int, args.seed)
    json_path = cast(Path | None, args.json)
    baseline_path = cast(Path | None, args.baseline)
    if repeats < 1 or rounds < 1:
        parser.error('--repeats and --rounds must be positive')

    results: Results = {}
    if suite in ('all', '2d3d'):
        results['2d3d'] = run_2d3d(repeats, rounds, target_ms, seed)
    if suite == 'all':
        print()
    if suite in ('all', 'construction'):
        results['construction'] = run_construction(repeats, rounds, target_ms)
    if suite == 'all':
        print()
    if suite in ('all', 'uncertainty'):
        results['uncertainty'] = run_uncertainty(repeats, rounds, target_ms)

    if json_path is not None:
        write_results(
            json_path,
            results,
            repeats=repeats,
            rounds=rounds,
            target_ms=target_ms,
            seed=seed,
        )
        print(f'\nResults written to {json_path}')
    if baseline_path is not None:
        print()
        compare_results(results, read_results(baseline_path), baseline_path)


if __name__ == '__main__':
    main()
