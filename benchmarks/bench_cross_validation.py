"""Benchmark and profile ``pysrf.cross_val_score``.

Two modes:

* timing (default) -- scaling table over matrix size, rank-grid width,
  fold count, and worker count, plus the parallel speedup.
* ``--profile`` -- a single cProfile run (forced ``n_jobs=1`` so the work
  is attributable) to surface the hot functions.

The cross-validation wrapper itself is thin: nearly all time is the
per-fold ``SRF.fit``. This harness exists to separate the wrapper
overhead (fold/rank fan-out, train-matrix construction, threadpool
setup) from the solver so regressions in either are visible.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from pysrf import cross_val_score


def make_low_rank_matrix(
    n: int, rank: int, noise_level: float = 0.05, seed: int = 0
) -> np.ndarray:
    """Low-rank symmetric similarity matrix with additive noise."""
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, rank)))
    s = w @ w.T
    s += noise_level * rng.standard_normal((n, n))
    s = (s + s.T) / 2
    return s


def _time_once(
    s: np.ndarray, ranks, n_folds: int, n_repeats: int, n_jobs: int
) -> float:
    """Wall-clock time of a single cross_val_score call."""
    start = time.perf_counter()
    cross_val_score(
        s,
        ranks=ranks,
        sampling_fraction=0.6,
        n_folds=n_folds,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
    )
    return time.perf_counter() - start


def _benchmark(
    s: np.ndarray,
    ranks,
    n_folds: int,
    n_repeats: int,
    n_jobs: int,
    n_warmup: int = 1,
    n_measure: int = 3,
) -> dict:
    """Median timing over repeated calls, after warmup."""
    for _ in range(n_warmup):
        _time_once(s, ranks, n_folds, n_repeats, n_jobs)
    times = np.zeros(n_measure)
    for i in range(n_measure):
        times[i] = _time_once(s, ranks, n_folds, n_repeats, n_jobs)
    n_fits = len(ranks) * n_folds * n_repeats
    return {
        "time_median": float(np.median(times)),
        "time_min": float(np.min(times)),
        "n_fits": n_fits,
        "per_fit_ms": float(np.median(times) / n_fits * 1e3),
    }


def _profile(s: np.ndarray, ranks, n_folds: int, n_repeats: int, top: int) -> None:
    """cProfile a single n_jobs=1 run and print the hottest frames."""
    profiler = cProfile.Profile()
    profiler.enable()
    cross_val_score(
        s,
        ranks=ranks,
        sampling_fraction=0.6,
        n_folds=n_folds,
        n_repeats=n_repeats,
        n_jobs=1,
    )
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(top)
    print(buffer.getvalue())


def _run_timing(args: argparse.Namespace) -> None:
    print("cross_val_score scaling (median of measured runs)")
    print("=" * 84)
    header = (
        f"{'n':>6} {'n_ranks':>8} {'folds':>6} {'reps':>5} "
        f"{'jobs':>5} {'fits':>5} {'total (s)':>11} {'per-fit (ms)':>13}"
    )
    print(header)
    print("-" * 84)

    for n in args.sizes:
        s = make_low_rank_matrix(n, rank=max(2, n // 20), seed=0)
        ranks = list(range(2, 2 + args.n_ranks))
        for n_jobs in args.jobs:
            stats = _benchmark(
                s,
                ranks,
                args.folds,
                args.repeats,
                n_jobs,
                n_warmup=args.warmup,
                n_measure=args.measure,
            )
            print(
                f"{n:6d} {args.n_ranks:8d} {args.folds:6d} {args.repeats:5d} "
                f"{n_jobs:5d} {stats['n_fits']:5d} "
                f"{stats['time_median']:11.3f} {stats['per_fit_ms']:13.1f}"
            )

    _print_parallel_speedup(args)


def _print_parallel_speedup(args: argparse.Namespace) -> None:
    """Sequential vs parallel on the largest size, to expose fan-out efficiency."""
    if 1 not in args.jobs or len(args.jobs) < 2:
        return
    n = max(args.sizes)
    s = make_low_rank_matrix(n, rank=max(2, n // 20), seed=0)
    ranks = list(range(2, 2 + args.n_ranks))
    serial = _benchmark(
        s, ranks, args.folds, args.repeats, 1, args.warmup, args.measure
    )
    print("\nparallel speedup at n=%d" % n)
    print("-" * 40)
    for n_jobs in args.jobs:
        if n_jobs == 1:
            continue
        par = _benchmark(
            s, ranks, args.folds, args.repeats, n_jobs, args.warmup, args.measure
        )
        speedup = serial["time_median"] / par["time_median"]
        print(f"  n_jobs={n_jobs:>3}: {speedup:5.2f}x  ({par['time_median']:.3f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[80, 160, 320])
    parser.add_argument("--n-ranks", type=int, default=5, help="width of the rank grid")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="cProfile a single n_jobs=1 run instead of timing",
    )
    parser.add_argument("--profile-n", type=int, default=160)
    parser.add_argument("--profile-top", type=int, default=20)
    args = parser.parse_args()

    if args.profile:
        s = make_low_rank_matrix(args.profile_n, rank=max(2, args.profile_n // 20))
        ranks = list(range(2, 2 + args.n_ranks))
        print(
            f"profiling cross_val_score: n={args.profile_n}, ranks={ranks}, "
            f"folds={args.folds}, repeats={args.repeats}, n_jobs=1"
        )
        print("=" * 84)
        _profile(s, ranks, args.folds, args.repeats, args.profile_top)
    else:
        _run_timing(args)


if __name__ == "__main__":
    main()
