import numpy as np
import pytest
from pysrf.bounds import (
    pmin_bound,
    p_upper_only_k,
    estimate_p_bound,
    estimate_p_bound_fast,
    lambda_bulk_dyson_raw,
)


def generate_test_matrix(n=30, rank=5, random_state=42):
    rng = np.random.RandomState(random_state)
    w = rng.rand(n, rank)
    s = w @ w.T
    s = (s + s.T) / 2
    return s


def test_pmin_bound():
    s = generate_test_matrix(n=30, rank=5)

    pmin, _, _, _, _ = pmin_bound(s, random_state=42, verbose=False)

    assert isinstance(pmin, (float, np.floating))
    assert 0 <= pmin <= 1


def test_pmin_bound_verbose():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmin_bern, pmin_lower, pmin_alt, mc_norms = pmin_bound(
        s, random_state=42, verbose=False
    )

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmin_bern, (float, np.floating))
    assert isinstance(pmin_lower, (float, np.floating))
    assert isinstance(pmin_alt, (float, np.floating))
    assert isinstance(mc_norms, np.ndarray)


def test_lambda_bulk_dyson_raw():
    s = generate_test_matrix(n=30, rank=5)

    p = 0.5
    edge = lambda_bulk_dyson_raw(s, p)

    assert isinstance(edge, float)
    assert edge >= 0


def test_lambda_bulk_dyson_raw_edge_cases():
    s = generate_test_matrix(n=30, rank=5)

    edge_low = lambda_bulk_dyson_raw(s, 0.0)
    assert edge_low == 0.0

    edge_high = lambda_bulk_dyson_raw(s, 1.0)
    assert edge_high == 0.0


def test_p_upper_only_k():
    s = generate_test_matrix(n=30, rank=5)

    k = 5
    pmax = p_upper_only_k(s, k=k, method="dyson", verbose=False, seed=42)

    assert isinstance(pmax, (float, np.floating))
    assert 0 <= pmax <= 1


def test_p_upper_only_k_invalid_k():
    s = generate_test_matrix(n=30, rank=5)

    with pytest.raises(ValueError):
        p_upper_only_k(s, k=0)

    with pytest.raises(ValueError):
        p_upper_only_k(s, k=100)


def test_estimate_p_bound():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmax, s_noise = estimate_p_bound(s, verbose=False, random_state=42)

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmax, (float, np.floating))
    assert isinstance(s_noise, np.ndarray)
    assert s_noise.shape == s.shape


def test_estimate_p_bound_fast():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmax, s_noise = estimate_p_bound_fast(
        s, verbose=False, random_state=42, n_jobs=1
    )

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmax, (float, np.floating))
    assert isinstance(s_noise, np.ndarray)
    assert s_noise.shape == s.shape


def test_estimate_p_bound_parameters():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmax, s_noise = estimate_p_bound(
        s,
        gamma=1.1,
        eta=0.1,
        rho=0.9,
        method="dyson",
        omega=0.85,
        verbose=False,
        random_state=42,
    )

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmax, (float, np.floating))
