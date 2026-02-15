import numpy as np
import pytest


def test_cython_import():
    try:
        from pysrf._bsum import update_w

        assert callable(update_w)
    except ImportError:
        pytest.skip("Cython module not available")


def test_fast_cython_import():
    try:
        from pysrf._bsum_fast import update_w

        assert callable(update_w)
    except ImportError:
        pytest.skip("Fast Cython module not available")


def test_blas_cython_import():
    try:
        from pysrf._bsum_fast_blas import update_w

        assert callable(update_w)
    except ImportError:
        pytest.skip("BLAS Cython module not available")


def test_blocked_cython_import():
    try:
        from pysrf._bsum_blocked import update_w

        assert callable(update_w)
    except ImportError:
        pytest.skip("Blocked Cython module not available")


def test_cython_compilation():
    try:
        from pysrf._bsum import update_w
    except ImportError:
        pytest.skip("Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_fast_cython_compilation():
    try:
        from pysrf._bsum_fast import update_w
    except ImportError:
        pytest.skip("Fast Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_blas_cython_compilation():
    try:
        from pysrf._bsum_fast_blas import update_w
    except ImportError:
        pytest.skip("BLAS Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_blocked_cython_compilation():
    try:
        from pysrf._bsum_blocked import update_w
    except ImportError:
        pytest.skip("Blocked Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_python_fallback():
    from pysrf.model import update_w

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_cython_vs_python():
    try:
        from pysrf._bsum import update_w as update_w_cython
    except ImportError:
        pytest.skip("Cython module not available")

    from pysrf.model import update_w as update_w_python

    rng = np.random.default_rng(42)
    n, r = 10, 3
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result_cython = update_w_cython(m, x0.copy(), max_iter=50, tol=1e-6)
    result_python = update_w_python(m, x0.copy(), max_iter=50, tol=1e-6)

    np.testing.assert_allclose(result_cython, result_python, rtol=1e-5, atol=1e-5)


def test_fast_vs_original_cython():
    try:
        from pysrf._bsum import update_w as update_w_original
        from pysrf._bsum_fast import update_w as update_w_fast
    except ImportError:
        pytest.skip("Cython modules not available")

    rng = np.random.default_rng(42)
    n, r = 50, 5
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result_original = update_w_original(m, x0.copy(), max_iter=100, tol=0.0)
    result_fast = update_w_fast(m, x0.copy(), max_iter=100, tol=0.0)

    error = np.max(np.abs(result_original - result_fast))
    assert error < 1e-9, f"Fast vs original discrepancy: {error:.4e}"


def test_fallback_chain_prefers_blocked():
    try:
        from pysrf._bsum_blocked import update_w  # noqa: F401
    except ImportError:
        pytest.skip("Blocked Cython module not available")

    from pysrf.model import _update_w_source

    assert _update_w_source == "blocked_cython", (
        f"Expected 'blocked_cython' but got '{_update_w_source}'"
    )
