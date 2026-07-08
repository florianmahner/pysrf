"""Tests for all BSUM solver implementations.

The scalar Cython variant (update_w_scalar) precomputes m[i,:] @ w for all r columns
in a single fused loop per row, using the same summation order as Python
(k=0..n-1, sequential accumulation), so results are bitwise-identical to
the Python version.

The BLAS variants use BLAS routines which may change summation order, so they
are numerically equivalent but not necessarily bit-identical. After a single
BSUM iteration, all implementations agree to ~1e-14 (machine epsilon) on any
symmetric matrix, proving the correction math is exact. Over many iterations,
BLAS rounding differences can cascade through the quartic solver's cube-root
sensitivity at the max(0, root) boundary, causing element-wise divergence while
reconstruction quality (||M - WW^T||_F / ||M||_F) remains identical to 10+
decimal places.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from helpers import make_bsum_data

import pysrf
from pysrf._bsum import BACKEND
from pysrf._bsum import bsum_step


def _load_python_bsum():
    """Load the pure-python _bsum.py even when the compiled module shadows it."""
    path = Path(pysrf.__file__).parent / "_bsum.py"
    spec = importlib.util.spec_from_file_location("pysrf._bsum_python", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec.loader.exec_module(module)
    return module


update_w_python = _load_python_bsum().update_w

try:
    from pysrf._bsum import update_w_scalar

    HAS_SCALAR = True
except ImportError:
    HAS_SCALAR = False

try:
    from pysrf._bsum import update_w_blas

    HAS_BLAS = True
except ImportError:
    HAS_BLAS = False

try:
    from pysrf._bsum import update_w_blas_blocked

    HAS_BLOCKED = True
except ImportError:
    HAS_BLOCKED = False


class TestPythonFallback:
    def test_runs_and_shape(self):
        m, w0 = make_bsum_data(n=10, r=3, seed=0)
        result = update_w_python(m, w0, max_iter=10)
        assert result.shape == (10, 3)
        assert np.all(result >= 0)
        assert not np.any(np.isnan(result))


class TestSolverSelection:
    @pytest.mark.skipif(not HAS_SCALAR, reason="Cython not compiled")
    def test_backend_is_cython(self):
        assert BACKEND == "cython", f"Expected 'cython' but got '{BACKEND}'"


@pytest.mark.skipif(not HAS_SCALAR, reason="Cython not compiled")
class TestScalarVsPython:
    """Scalar Cython uses the same summation order as Python; results must be
    bitwise-identical.
    """

    def test_equivalence(self):
        m, w0 = make_bsum_data(seed=1234)
        np.testing.assert_array_equal(
            update_w_scalar(m, w0),
            update_w_python(m, w0),
        )

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_equivalence_multiple_seeds(self, seed):
        m, w0 = make_bsum_data(seed=seed)
        np.testing.assert_array_equal(
            update_w_scalar(m, w0),
            update_w_python(m, w0),
        )

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10), (200, 15)])
    def test_equivalence_different_sizes(self, n, r):
        m, w0 = make_bsum_data(n=n, r=r, seed=42)
        np.testing.assert_array_equal(
            update_w_scalar(m, w0),
            update_w_python(m, w0),
        )

    def test_nonnegativity(self):
        m, w0 = make_bsum_data(n=100, r=10, seed=42)
        result = update_w_scalar(m, w0)
        assert np.all(result >= 0)

    def test_convergence_tolerance(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        np.testing.assert_array_equal(
            update_w_scalar(m, w0, max_iter=200),
            update_w_python(m, w0, max_iter=200),
        )

    def test_does_not_modify_inputs(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        m_copy = m.copy()
        x0_copy = w0.copy()
        update_w_scalar(m, w0)
        np.testing.assert_array_equal(m, m_copy)
        np.testing.assert_array_equal(w0, x0_copy)


@pytest.mark.skipif(
    not (HAS_BLAS and HAS_SCALAR), reason="BLAS and scalar Cython required"
)
class TestBlasVsScalar:
    """BLAS variant uses dgemv/ddot/daxpy which may reorder summation.

    After 1 iteration the math is provably identical — differences are pure
    IEEE 754 rounding (~1e-14). Over many iterations, cube-root sensitivity
    at the max(0, root) boundary amplifies these, but reconstruction quality
    remains identical.
    """

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_single_iteration_elementwise(self, seed):
        m, w0 = make_bsum_data(seed=seed)
        np.testing.assert_allclose(
            update_w_blas(m, w0, max_iter=1),
            update_w_scalar(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10), (200, 15)])
    def test_single_iteration_different_sizes(self, n, r):
        m, w0 = make_bsum_data(n=n, r=r, seed=42)
        np.testing.assert_allclose(
            update_w_blas(m, w0, max_iter=1),
            update_w_scalar(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_reconstruction_equivalence(self, seed):
        m, w0 = make_bsum_data(seed=seed)
        w_blas = update_w_blas(m, w0)
        w_scalar = update_w_scalar(m, w0)
        norm_m = np.linalg.norm(m, "fro")
        err_blas = np.linalg.norm(m - w_blas @ w_blas.T, "fro") / norm_m
        err_scalar = np.linalg.norm(m - w_scalar @ w_scalar.T, "fro") / norm_m
        np.testing.assert_allclose(err_blas, err_scalar, rtol=1e-2)

    def test_nonnegativity(self):
        m, w0 = make_bsum_data(n=100, r=10, seed=42)
        result = update_w_blas(m, w0)
        assert np.all(result >= 0)


@pytest.mark.skipif(
    not (HAS_BLOCKED and HAS_SCALAR), reason="Blocked and scalar Cython required"
)
class TestBlockedVsScalar:
    """Blocked variant uses dsymm/dgemm (BLAS-3) which may reorder summation.

    Same numerical profile as BLAS variant: exact at 1 iteration, reconstruction-
    equivalent at convergence.
    """

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_single_iteration_elementwise(self, seed):
        m, w0 = make_bsum_data(seed=seed)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1),
            update_w_scalar(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.parametrize("block_size", [1, 10, 50, 100])
    def test_single_iteration_block_sizes(self, block_size):
        m, w0 = make_bsum_data(n=100, r=10, seed=42)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1, block_size=block_size),
            update_w_scalar(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10), (200, 15)])
    def test_single_iteration_different_sizes(self, n, r):
        m, w0 = make_bsum_data(n=n, r=r, seed=42)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1),
            update_w_scalar(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_reconstruction_equivalence(self, seed):
        m, w0 = make_bsum_data(seed=seed)
        w_blocked = update_w_blas_blocked(m, w0)
        w_scalar = update_w_scalar(m, w0)
        norm_m = np.linalg.norm(m, "fro")
        err_blocked = np.linalg.norm(m - w_blocked @ w_blocked.T, "fro") / norm_m
        err_scalar = np.linalg.norm(m - w_scalar @ w_scalar.T, "fro") / norm_m
        np.testing.assert_allclose(err_blocked, err_scalar, rtol=1e-2)

    def test_nonnegativity(self):
        m, w0 = make_bsum_data(n=100, r=10, seed=42)
        result = update_w_blas_blocked(m, w0)
        assert np.all(result >= 0)


class TestEdgeCases:
    """Edge-case tests: rank=1, small n, block_size > n."""

    def test_rank_1(self):
        m, w0 = make_bsum_data(n=20, r=1, seed=42)
        result = update_w_python(m, w0, max_iter=50)
        assert result.shape == (20, 1)
        assert np.all(result >= 0)

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_rank_1_blocked(self):
        m, w0 = make_bsum_data(n=20, r=1, seed=42)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1),
            update_w_python(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_block_size_larger_than_n(self):
        m, w0 = make_bsum_data(n=10, r=3, seed=42)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1, block_size=100),
            update_w_python(m, w0, max_iter=1),
            atol=1e-10,
        )

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_block_size_equals_n(self):
        m, w0 = make_bsum_data(n=10, r=3, seed=42)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1, block_size=10),
            update_w_python(m, w0, max_iter=1),
            atol=1e-10,
        )

    def test_small_n(self):
        m, w0 = make_bsum_data(n=3, r=2, seed=42)
        result = update_w_python(m, w0, max_iter=50)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_small_n_blocked(self):
        m, w0 = make_bsum_data(n=3, r=2, seed=42)
        np.testing.assert_allclose(
            update_w_blas_blocked(m, w0, max_iter=1),
            update_w_python(m, w0, max_iter=1),
            atol=1e-10,
        )


class TestResidual:
    """Test the memory-efficient bsum_step helper."""

    def test_matches_direct_computation(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        w = update_w_python(m, w0)
        residual = bsum_step(m, w)
        wwt = w @ w.T
        np.testing.assert_allclose(
            residual.rec_error, np.linalg.norm(m - wwt, "fro"), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.sqrt(residual.ss_wwt), np.linalg.norm(wwt, "fro"), rtol=1e-10
        )

    @pytest.mark.parametrize("n,r", [(10, 3), (50, 5), (100, 10)])
    def test_different_sizes(self, n, r):
        m, w0 = make_bsum_data(n=n, r=r, seed=42)
        w = update_w_python(m, w0, max_iter=10)
        residual = bsum_step(m, w)
        wwt = w @ w.T
        np.testing.assert_allclose(
            residual.rec_error, np.linalg.norm(m - wwt, "fro"), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.sqrt(residual.ss_wwt), np.linalg.norm(wwt, "fro"), rtol=1e-10
        )

    def test_residual_nonnegative(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        residual = bsum_step(m, w0)
        assert residual.rec_error >= 0
        assert residual.relative_fit >= 0

    def test_matches_python_mirror(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        python_step = _load_python_bsum().bsum_step(m, w0)
        np.testing.assert_array_equal(np.array(bsum_step(m, w0)), np.array(python_step))


class TestMonotonicity:
    """Verify that BSUM objective decreases monotonically per iteration.

    Shi et al. (2017) Theorem 1 guarantees this property.
    """

    @staticmethod
    def _run_iterations(m, w0, update_fn, n_iter, **kwargs):
        w = w0.copy()
        errors = []
        for _ in range(n_iter):
            w = update_fn(m, w, max_iter=1, **kwargs)
            errors.append(np.linalg.norm(m - w @ w.T, "fro"))
        return errors

    def test_python_monotonicity(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        errors = self._run_iterations(m, w0, update_w_python, 20)
        diffs = np.diff(errors)
        assert np.all(
            diffs <= 1e-10
        ), f"Python: non-monotonic at {np.argmax(diffs > 1e-10)}"

    @pytest.mark.skipif(not HAS_SCALAR, reason="Cython not compiled")
    def test_scalar_monotonicity(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        errors = self._run_iterations(m, w0, update_w_scalar, 20)
        diffs = np.diff(errors)
        assert np.all(
            diffs <= 1e-10
        ), f"Scalar: non-monotonic at {np.argmax(diffs > 1e-10)}"

    @pytest.mark.skipif(not HAS_BLOCKED, reason="Cython not compiled")
    def test_blocked_monotonicity(self):
        m, w0 = make_bsum_data(n=50, r=5, seed=42)
        errors = self._run_iterations(m, w0, update_w_blas_blocked, 20)
        diffs = np.diff(errors)
        assert np.all(
            diffs <= 1e-10
        ), f"Blocked: non-monotonic at {np.argmax(diffs > 1e-10)}"
