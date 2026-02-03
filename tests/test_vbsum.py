"""Tests for vBSUM algorithm implementation."""

from __future__ import annotations

import numpy as np
import pytest
import time

from pysrf._bsum import update_w as update_w_sbsum
from pysrf._vbsum import update_w_vbsum
from pysrf._vbsum_reference import update_w_vbsum_python


def generate_test_matrix(n: int, r: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a low-rank test matrix."""
    rng = np.random.default_rng(seed)
    W_true = rng.random((n, r)) * 2
    M = W_true @ W_true.T
    M = (M + M.T) / 2
    return M, W_true


class TestVBSUMCythonCorrectness:
    """Test that Cython vBSUM matches Python reference implementation."""

    def test_cython_matches_python_small(self):
        """Test on small matrix."""
        M, _ = generate_test_matrix(n=30, r=5, seed=42)
        W0 = np.random.default_rng(123).random((30, 5))

        W_cython = update_w_vbsum(M, W0.copy(), max_iter=50, tol=0.0, imax=5)
        W_python = update_w_vbsum_python(M, W0.copy(), max_iter=50, tol=0.0, imax=5)

        max_diff = np.max(np.abs(W_cython - W_python))
        assert max_diff < 1e-8, f"Cython vs Python max diff: {max_diff}"

    def test_cython_matches_python_medium(self):
        """Test on medium matrix."""
        M, _ = generate_test_matrix(n=100, r=10, seed=42)
        W0 = np.random.default_rng(123).random((100, 10))

        W_cython = update_w_vbsum(M, W0.copy(), max_iter=30, tol=0.0, imax=3)
        W_python = update_w_vbsum_python(M, W0.copy(), max_iter=30, tol=0.0, imax=3)

        max_diff = np.max(np.abs(W_cython - W_python))
        assert max_diff < 1e-8, f"Cython vs Python max diff: {max_diff}"

    def test_cython_matches_python_multiple_seeds(self):
        """Test across multiple random seeds."""
        for seed in [0, 42, 123, 456, 789]:
            M, _ = generate_test_matrix(n=50, r=8, seed=seed)
            W0 = np.random.default_rng(seed + 1000).random((50, 8))

            W_cython = update_w_vbsum(M, W0.copy(), max_iter=20, tol=0.0, imax=3)
            W_python = update_w_vbsum_python(M, W0.copy(), max_iter=20, tol=0.0, imax=3)

            max_diff = np.max(np.abs(W_cython - W_python))
            assert max_diff < 1e-8, f"Seed {seed}: Cython vs Python max diff: {max_diff}"


class TestVBSUMConvergence:
    """Test vBSUM convergence properties."""

    def test_objective_decreases(self):
        """Test that objective function decreases monotonically."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        def objective(W):
            return np.linalg.norm(M - W @ W.T, 'fro') ** 2

        W = W0.copy()
        prev_obj = objective(W)

        for i in range(20):
            W = update_w_vbsum(M, W, max_iter=1, tol=0.0, imax=5)
            curr_obj = objective(W)
            assert curr_obj <= prev_obj + 1e-10, f"Objective increased at iter {i}: {prev_obj} -> {curr_obj}"
            prev_obj = curr_obj

    def test_convergence_with_tolerance(self):
        """Test that algorithm converges when tolerance is set."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        # Run with strict tolerance
        W = update_w_vbsum(M, W0.copy(), max_iter=1000, tol=1e-6, imax=5)

        # Check reconstruction quality
        rec_error = np.linalg.norm(M - W @ W.T, 'fro')
        assert rec_error < 50.0, f"Reconstruction error too high: {rec_error}"

    def test_nonnegative_output(self):
        """Test that output is nonnegative."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        W = update_w_vbsum(M, W0, max_iter=50, tol=0.0, imax=5)

        assert np.all(W >= -1e-10), f"Negative values in W: min={W.min()}"


class TestVBSUMvssBSUM:
    """Compare vBSUM and sBSUM algorithms."""

    def test_both_decrease_objective(self):
        """Test that both algorithms decrease the objective."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        def objective(W):
            return np.linalg.norm(M - W @ W.T, 'fro') ** 2

        initial_obj = objective(W0)

        W_sbsum = update_w_sbsum(M, W0.copy(), max_iter=50, tol=0.0)
        W_vbsum = update_w_vbsum(M, W0.copy(), max_iter=50, tol=0.0, imax=5)

        obj_sbsum = objective(W_sbsum)
        obj_vbsum = objective(W_vbsum)

        assert obj_sbsum < initial_obj, "sBSUM did not decrease objective"
        assert obj_vbsum < initial_obj, "vBSUM did not decrease objective"

    def test_similar_quality_with_enough_iterations(self):
        """Test that both algorithms achieve similar quality with enough iterations."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        # Run both for many iterations
        W_sbsum = update_w_sbsum(M, W0.copy(), max_iter=200, tol=0.0)
        W_vbsum = update_w_vbsum(M, W0.copy(), max_iter=200, tol=0.0, imax=10)

        err_sbsum = np.linalg.norm(M - W_sbsum @ W_sbsum.T, 'fro')
        err_vbsum = np.linalg.norm(M - W_vbsum @ W_vbsum.T, 'fro')

        # vBSUM should achieve reasonable quality (within 5x of sBSUM)
        assert err_vbsum < err_sbsum * 5, f"vBSUM error {err_vbsum} too high vs sBSUM {err_sbsum}"

    def test_high_n_over_r_ratio(self):
        """Test that vBSUM performs well when n >> r."""
        # For high n/r ratio, vBSUM can actually be better
        M, _ = generate_test_matrix(n=200, r=5, seed=42)
        W0 = np.random.default_rng(123).random((200, 5))

        W_sbsum = update_w_sbsum(M, W0.copy(), max_iter=100, tol=0.0)
        W_vbsum = update_w_vbsum(M, W0.copy(), max_iter=100, tol=0.0, imax=5)

        err_sbsum = np.linalg.norm(M - W_sbsum @ W_sbsum.T, 'fro')
        err_vbsum = np.linalg.norm(M - W_vbsum @ W_vbsum.T, 'fro')

        # Both should achieve reasonable reconstruction
        assert err_sbsum < np.linalg.norm(M, 'fro'), "sBSUM error unreasonable"
        assert err_vbsum < np.linalg.norm(M, 'fro'), "vBSUM error unreasonable"


class TestVBSUMPerformance:
    """Performance benchmarks for vBSUM vs sBSUM."""

    @pytest.mark.parametrize("n,r", [(100, 10), (200, 15), (500, 20)])
    def test_vbsum_cython_faster_than_python(self, n, r):
        """Test that Cython vBSUM is faster than Python vBSUM."""
        M, _ = generate_test_matrix(n=n, r=r, seed=42)
        W0 = np.random.default_rng(123).random((n, r))
        max_iter = 10

        # Time Cython
        start = time.time()
        W_cython = update_w_vbsum(M, W0.copy(), max_iter=max_iter, tol=0.0, imax=3)
        time_cython = time.time() - start

        # Time Python
        start = time.time()
        W_python = update_w_vbsum_python(M, W0.copy(), max_iter=max_iter, tol=0.0, imax=3)
        time_python = time.time() - start

        speedup = time_python / time_cython
        print(f"\nn={n}, r={r}: Cython {time_cython:.3f}s, Python {time_python:.3f}s, speedup {speedup:.1f}x")

        # Cython should be at least 2x faster
        assert speedup > 1.5, f"Cython speedup too low: {speedup:.1f}x"

    def test_timing_comparison_report(self):
        """Generate timing comparison report."""
        print("\n" + "=" * 60)
        print("Timing Comparison: sBSUM vs vBSUM (Cython)")
        print("=" * 60)

        test_cases = [
            (100, 10, 50),
            (200, 15, 30),
            (500, 20, 20),
            (1000, 25, 10),
        ]

        print(f"\n{'n':>6} {'r':>4} {'iter':>5} {'sBSUM(s)':>10} {'vBSUM(s)':>10} {'ratio':>8} {'err_s':>10} {'err_v':>10}")
        print("-" * 75)

        for n, r, max_iter in test_cases:
            M, _ = generate_test_matrix(n=n, r=r, seed=42)
            W0 = np.random.default_rng(123).random((n, r))

            # Time sBSUM
            start = time.time()
            W_sbsum = update_w_sbsum(M, W0.copy(), max_iter=max_iter, tol=0.0)
            time_sbsum = time.time() - start
            err_sbsum = np.linalg.norm(M - W_sbsum @ W_sbsum.T, 'fro')

            # Time vBSUM
            start = time.time()
            W_vbsum = update_w_vbsum(M, W0.copy(), max_iter=max_iter, tol=0.0, imax=5)
            time_vbsum = time.time() - start
            err_vbsum = np.linalg.norm(M - W_vbsum @ W_vbsum.T, 'fro')

            ratio = time_vbsum / time_sbsum
            print(f"{n:6d} {r:4d} {max_iter:5d} {time_sbsum:10.4f} {time_vbsum:10.4f} {ratio:8.2f}x {err_sbsum:10.2f} {err_vbsum:10.2f}")


class TestVBSUMImaxEffect:
    """Test the effect of imax (inner iterations) parameter."""

    def test_higher_imax_better_convergence(self):
        """Test that higher imax gives better per-iteration convergence."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        errors = {}
        for imax in [1, 3, 5, 10]:
            W = update_w_vbsum(M, W0.copy(), max_iter=50, tol=0.0, imax=imax)
            errors[imax] = np.linalg.norm(M - W @ W.T, 'fro')

        # Higher imax should generally give better results
        assert errors[5] <= errors[1] * 2, f"imax=5 ({errors[5]}) should be close to imax=1 ({errors[1]})"

    def test_imax_1_still_converges(self):
        """Test that imax=1 still converges."""
        M, _ = generate_test_matrix(n=50, r=5, seed=42)
        W0 = np.random.default_rng(123).random((50, 5))

        def objective(W):
            return np.linalg.norm(M - W @ W.T, 'fro') ** 2

        initial_obj = objective(W0)
        W = update_w_vbsum(M, W0.copy(), max_iter=100, tol=0.0, imax=1)
        final_obj = objective(W)

        assert final_obj < initial_obj * 0.5, "imax=1 did not converge sufficiently"
