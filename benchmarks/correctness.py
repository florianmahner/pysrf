"""Correctness test: all BSUM implementations vs original (ground truth).

Compares fast, blas, and blocked (multiple block sizes) against the original
Cython implementation across problem sizes and iteration counts.

Checks:
  1. W matrix element-wise agreement (max |W_test - W_orig|)
  2. Reconstruction quality agreement (||M - W@W.T||_F / ||M||_F)
  3. Whether differences grow, stay bounded, or diverge with iterations

Usage:
    python -u correctness.py
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

RESULTS_CSV = Path(__file__).parent / "correctness.csv"

CSV_COLUMNS = [
    "n", "rank", "max_iter", "impl",
    "max_w_diff", "mean_w_diff", "recon_orig", "recon_test", "recon_diff",
    "bit_identical",
]


def load_implementations():
    impls = {}
    from pysrf._bsum import update_w as original
    impls["original"] = original
    try:
        from pysrf._bsum_fast import update_w as fast
        impls["fast"] = fast
    except ImportError:
        pass
    try:
        from pysrf._bsum_fast_blas import update_w as blas
        impls["blas"] = blas
    except ImportError:
        pass
    try:
        from pysrf._bsum_blocked import update_w as blocked
        impls["blocked"] = blocked
    except ImportError:
        pass
    return impls


def make_data(n: int, rank: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    w_true = rng.rand(n, rank).astype(np.float64)
    m = w_true @ w_true.T
    m = (m + m.T) / 2
    w0 = rng.rand(n, rank).astype(np.float64) * np.sqrt(m.mean() / rank)
    return np.ascontiguousarray(m), np.ascontiguousarray(w0)


def compare(m, w0, impl_fn, orig_fn, max_iter: int, **kwargs):
    w_orig = orig_fn(m, w0.copy(), max_iter=max_iter, tol=0.0)
    w_test = impl_fn(m, w0.copy(), max_iter=max_iter, tol=0.0, **kwargs)

    norm_m = np.linalg.norm(m, "fro")
    max_w_diff = np.max(np.abs(w_test - w_orig))
    mean_w_diff = np.mean(np.abs(w_test - w_orig))
    recon_orig = np.linalg.norm(m - w_orig @ w_orig.T, "fro") / norm_m
    recon_test = np.linalg.norm(m - w_test @ w_test.T, "fro") / norm_m
    identical = np.array_equal(w_test, w_orig)

    return {
        "max_w_diff": max_w_diff,
        "mean_w_diff": mean_w_diff,
        "recon_orig": recon_orig,
        "recon_test": recon_test,
        "recon_diff": recon_test - recon_orig,
        "bit_identical": identical,
    }


def main():
    impls = load_implementations()
    orig_fn = impls["original"]

    sizes = [(500, 50), (1854, 50), (1854, 100), (5000, 50)]
    iter_counts = [1, 5, 10, 50, 100, 500]
    blocked_sizes = [1, 50, 100, 200]

    results = []

    print(f"{'n':>5} {'rank':>4} {'iter':>5} {'impl':>16} "
          f"{'max_W_diff':>12} {'mean_W_diff':>12} "
          f"{'recon_orig':>14} {'recon_test':>14} {'recon_diff':>12} "
          f"{'identical':>9}")
    print("-" * 125)

    for n, rank in sizes:
        m, w0 = make_data(n, rank)

        for max_iter in iter_counts:
            if n >= 5000 and max_iter > 100:
                continue

            t0 = time.perf_counter()

            for impl_name in ["fast", "blas"]:
                if impl_name not in impls:
                    continue
                result = compare(m, w0, impls[impl_name], orig_fn, max_iter)
                row = {
                    "n": n, "rank": rank, "max_iter": max_iter,
                    "impl": impl_name,
                    "max_w_diff": f"{result['max_w_diff']:.2e}",
                    "mean_w_diff": f"{result['mean_w_diff']:.2e}",
                    "recon_orig": f"{result['recon_orig']:.12f}",
                    "recon_test": f"{result['recon_test']:.12f}",
                    "recon_diff": f"{result['recon_diff']:+.2e}",
                    "bit_identical": result["bit_identical"],
                }
                results.append(row)
                print(
                    f"{n:>5} {rank:>4} {max_iter:>5} {impl_name:>16} "
                    f"{row['max_w_diff']:>12} {row['mean_w_diff']:>12} "
                    f"{row['recon_orig']:>14} {row['recon_test']:>14} "
                    f"{row['recon_diff']:>12} {str(result['bit_identical']):>9}",
                    flush=True,
                )

            if "blocked" in impls:
                for bs in blocked_sizes:
                    if bs > n:
                        continue
                    label = f"blocked(B={bs})"
                    result = compare(
                        m, w0, impls["blocked"], orig_fn, max_iter, block_size=bs
                    )
                    row = {
                        "n": n, "rank": rank, "max_iter": max_iter,
                        "impl": label,
                        "max_w_diff": f"{result['max_w_diff']:.2e}",
                        "mean_w_diff": f"{result['mean_w_diff']:.2e}",
                        "recon_orig": f"{result['recon_orig']:.12f}",
                        "recon_test": f"{result['recon_test']:.12f}",
                        "recon_diff": f"{result['recon_diff']:+.2e}",
                        "bit_identical": result["bit_identical"],
                    }
                    results.append(row)
                    print(
                        f"{n:>5} {rank:>4} {max_iter:>5} {label:>16} "
                        f"{row['max_w_diff']:>12} {row['mean_w_diff']:>12} "
                        f"{row['recon_orig']:>14} {row['recon_test']:>14} "
                        f"{row['recon_diff']:>12} {str(result['bit_identical']):>9}",
                        flush=True,
                    )

            elapsed = time.perf_counter() - t0
            if elapsed > 1:
                print(f"  ({elapsed:.1f}s)", flush=True)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to: {RESULTS_CSV}")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for impl_name in sorted({r["impl"] for r in results}):
        impl_rows = [r for r in results if r["impl"] == impl_name]
        all_identical = all(r["bit_identical"] for r in impl_rows)
        max_diff = max(float(r["max_w_diff"]) for r in impl_rows)
        max_recon_diff = max(abs(float(r["recon_diff"])) for r in impl_rows)

        print(f"\n{impl_name} vs original:")
        print(f"  Bit-identical:        {'YES' if all_identical else 'NO'}")
        print(f"  Max W element diff:   {max_diff:.2e}")
        print(f"  Max recon error diff: {max_recon_diff:.2e}")

        if all_identical:
            print(f"  VERDICT: EXACT MATCH")
        elif max_recon_diff < 1e-10:
            print(f"  VERDICT: Numerically equivalent (BLAS accumulation order only)")
        else:
            print(f"  VERDICT: DIFFERENCES DETECTED - investigate!")


if __name__ == "__main__":
    main()
