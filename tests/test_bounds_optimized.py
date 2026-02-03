"""Tests verifying optimized bounds implementation matches original."""

from __future__ import annotations

import numpy as np
import pytest

from pysrf import bounds
from pysrf import bounds_optimized


def generate_test_matrix(
    n: int, rank: int, noise_level: float = 0.1, seed: int = 42
) -> np.ndarray:
    """Generate a low-rank test matrix with noise."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((rank, n))
    S = U @ V
    S = (S + S.T) / 2
    S += noise_level * rng.standard_normal((n, n))
    S = (S + S.T) / 2
    return S


class TestPminBound:
    """Test pmin_bound function matches original implementation."""

    def test_small_matrix(self):
        """Test on small matrix."""
        S = generate_test_matrix(20, rank=3, seed=42)

        result_orig = bounds.pmin_bound(S, random_state=42, verbose=False)
        result_opt = bounds_optimized.pmin_bound(S, random_state=42, verbose=False)

        for i in range(4):
            assert np.isclose(
                result_orig[i], result_opt[i], rtol=1e-10
            ), f"Mismatch in return value {i}: {result_orig[i]} vs {result_opt[i]}"

    def test_medium_matrix(self):
        """Test on medium matrix."""
        S = generate_test_matrix(50, rank=5, seed=123)

        result_orig = bounds.pmin_bound(S, random_state=123, verbose=False)
        result_opt = bounds_optimized.pmin_bound(S, random_state=123, verbose=False)

        for i in range(4):
            assert np.isclose(
                result_orig[i], result_opt[i], rtol=1e-10
            ), f"Mismatch in return value {i}: {result_orig[i]} vs {result_opt[i]}"


class TestSolveVDE:
    """Test VDE solver."""

    def test_convergence(self):
        """Test VDE solver converges to similar solution."""
        S = generate_test_matrix(30, rank=4, seed=99)
        p = 0.5
        z = 1.0

        result_orig = bounds._solve_vde(S, p, z, max_iter=2000)
        result_opt = bounds_optimized._solve_vde(S, p, z, max_iter=1000)

        assert np.allclose(
            result_orig, result_opt, rtol=1e-4, atol=1e-6
        ), "VDE solutions differ significantly"


class TestLambdaBulkDyson:
    """Test lambda_bulk_dyson_raw function."""

    def test_small_matrix_various_p(self):
        """Test bulk edge estimation at various p values."""
        S = generate_test_matrix(30, rank=3, seed=55)

        from numpy.linalg import eigvalsh

        s2_max = np.max(eigvalsh(S**2))

        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result_orig = bounds.lambda_bulk_dyson_raw(S, p, ngrid=100)
            result_opt = bounds_optimized.lambda_bulk_dyson_raw(
                S, p, ngrid=40, s2_max=s2_max
            )

            rel_diff = abs(result_orig - result_opt) / (abs(result_orig) + 1e-10)
            assert (
                rel_diff < 0.2
            ), f"Bulk edge at p={p} differs: {result_orig} vs {result_opt} (rel_diff={rel_diff})"


class TestMonteCarloBulkEdge:
    """Test Monte Carlo bulk edge estimation."""

    def test_similar_distribution(self):
        """Test that Monte Carlo gives similar results with different trial counts."""
        S = generate_test_matrix(40, rank=4, seed=77)
        p = 0.6

        result_orig = bounds.monte_carlo_bulk_edge_raw(S, p, n_trials=400, seed=77)
        result_opt = bounds_optimized.monte_carlo_bulk_edge_raw(
            S, p, n_trials=150, seed=77
        )

        rel_diff = abs(result_orig - result_opt) / (abs(result_orig) + 1e-10)
        assert (
            rel_diff < 0.2
        ), f"Monte Carlo bulk edge differs: {result_orig} vs {result_opt}"


class TestPUpperOnlyK:
    """Test p_upper_only_k function."""

    def test_small_matrix_k1(self):
        """Test p_upper with k=1 on small matrix."""
        S = generate_test_matrix(30, rank=3, seed=88)

        from numpy.linalg import eigvalsh

        lam = np.sort(eigvalsh(S))[::-1]
        s2_max = np.max(eigvalsh(S**2))

        result_orig = bounds.p_upper_only_k(
            S, k=1, method="dyson", verbose=False, seed=88
        )
        result_opt = bounds_optimized.p_upper_only_k(
            S, k=1, method="dyson", verbose=False, seed=88, lam=lam, s2_max=s2_max
        )

        assert (
            abs(result_orig - result_opt) < 0.1
        ), f"p_upper differs: {result_orig} vs {result_opt}"

    def test_medium_matrix_k3(self):
        """Test p_upper with k=3 on medium matrix."""
        S = generate_test_matrix(50, rank=5, seed=99)

        from numpy.linalg import eigvalsh

        lam = np.sort(eigvalsh(S))[::-1]
        s2_max = np.max(eigvalsh(S**2))

        result_orig = bounds.p_upper_only_k(
            S, k=3, method="dyson", verbose=False, seed=99
        )
        result_opt = bounds_optimized.p_upper_only_k(
            S, k=3, method="dyson", verbose=False, seed=99, lam=lam, s2_max=s2_max
        )

        assert (
            abs(result_orig - result_opt) < 0.1
        ), f"p_upper differs: {result_orig} vs {result_opt}"


class TestEstimateSamplingBounds:
    """Test main estimate_sampling_bounds function."""

    def test_small_matrix(self):
        """Test on small matrix."""
        S = generate_test_matrix(40, rank=4, seed=111)

        pmin_orig, pmax_orig, S_noise_orig = bounds.estimate_sampling_bounds(
            S, method="dyson", verbose=False, random_state=111
        )
        pmin_opt, pmax_opt, S_noise_opt = bounds_optimized.estimate_sampling_bounds(
            S, method="dyson", verbose=False, random_state=111
        )

        assert (
            abs(pmin_orig - pmin_opt) < 0.05
        ), f"pmin differs: {pmin_orig} vs {pmin_opt}"
        assert (
            abs(pmax_orig - pmax_opt) < 0.15
        ), f"pmax differs: {pmax_orig} vs {pmax_opt}"
        assert np.allclose(
            S_noise_orig, S_noise_opt, rtol=1e-10
        ), "S_noise matrices differ"

    def test_medium_matrix(self):
        """Test on medium matrix."""
        S = generate_test_matrix(60, rank=5, seed=222)

        pmin_orig, pmax_orig, S_noise_orig = bounds.estimate_sampling_bounds(
            S, method="dyson", verbose=False, random_state=222
        )
        pmin_opt, pmax_opt, S_noise_opt = bounds_optimized.estimate_sampling_bounds(
            S, method="dyson", verbose=False, random_state=222
        )

        assert (
            abs(pmin_orig - pmin_opt) < 0.05
        ), f"pmin differs: {pmin_orig} vs {pmin_opt}"
        assert (
            abs(pmax_orig - pmax_opt) < 0.15
        ), f"pmax differs: {pmax_orig} vs {pmax_opt}"


class TestEstimateSamplingBoundsFast:
    """Test parallelized estimate_sampling_bounds_fast function."""

    def test_small_matrix(self):
        """Test fast version on small matrix."""
        S = generate_test_matrix(40, rank=4, seed=333)

        pmin_orig, pmax_orig, S_noise_orig = bounds.estimate_sampling_bounds_fast(
            S, method="dyson", verbose=False, random_state=333, n_jobs=1
        )
        (
            pmin_opt,
            pmax_opt,
            S_noise_opt,
        ) = bounds_optimized.estimate_sampling_bounds_fast(
            S, method="dyson", verbose=False, random_state=333, n_jobs=1
        )

        assert (
            abs(pmin_orig - pmin_opt) < 0.05
        ), f"pmin differs: {pmin_orig} vs {pmin_opt}"
        assert (
            abs(pmax_orig - pmax_opt) < 0.15
        ), f"pmax differs: {pmax_orig} vs {pmax_opt}"

    def test_consistency_with_serial(self):
        """Test that optimized version is internally consistent."""
        S = generate_test_matrix(50, rank=5, seed=444)

        pmin_serial, pmax_serial, _ = bounds_optimized.estimate_sampling_bounds(
            S, method="dyson", verbose=False, random_state=444
        )
        pmin_fast, pmax_fast, _ = bounds_optimized.estimate_sampling_bounds_fast(
            S, method="dyson", verbose=False, random_state=444, n_jobs=1
        )

        assert np.isclose(
            pmin_serial, pmin_fast, rtol=1e-10
        ), f"pmin differs between serial and fast: {pmin_serial} vs {pmin_fast}"
        assert np.isclose(
            pmax_serial, pmax_fast, rtol=1e-10
        ), f"pmax differs between serial and fast: {pmax_serial} vs {pmax_fast}"


class TestEdgeCache:
    """Test edge computation cache."""

    def test_cache_hit(self):
        """Test that cache returns same value for similar p."""
        cache = bounds_optimized._EdgeCache()

        cache.set(0.5, 1.234)

        assert cache.get(0.5, tol=1e-3) == 1.234
        assert cache.get(0.5001, tol=1e-3) == 1.234
        assert cache.get(0.4999, tol=1e-3) == 1.234

    def test_cache_miss(self):
        """Test that cache returns None for different p."""
        cache = bounds_optimized._EdgeCache()

        cache.set(0.5, 1.234)

        assert cache.get(0.6, tol=1e-3) is None
        assert cache.get(0.4, tol=1e-3) is None
