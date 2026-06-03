import numpy as np


def make_symmetric_matrix(n=30, rank=5, noise_level=0.01, seed=42):
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, rank)))
    s = w @ w.T
    s += noise_level * rng.standard_normal((n, n))
    s = (s + s.T) / 2
    return s


def make_missing_matrix(s, fraction=0.3, seed=42):
    rng = np.random.default_rng(seed)
    s_missing = s.copy()
    n = s.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_entries = len(triu_i)
    n_missing = int(n_entries * fraction)
    idx = rng.choice(n_entries, size=n_missing, replace=False)
    for k in idx:
        i, j = triu_i[k], triu_j[k]
        s_missing[i, j] = np.nan
        s_missing[j, i] = np.nan
    return s_missing


def make_bsum_data(n=100, r=10, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    w0 = rng.random((n, r))
    return m, w0


def make_cv_diagnostic_matrix(
    n: int,
    rank: int,
    nuisance_rank: int,
    nuisance_scale: float,
    noise: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dominant = rng.gamma(shape=1.4, scale=1.0, size=(n, rank))
    dominant /= np.linalg.norm(dominant, axis=0, keepdims=True)
    dominant *= np.linspace(7.0, 4.5, rank)

    nuisance = rng.gamma(shape=1.4, scale=1.0, size=(n, nuisance_rank))
    nuisance /= np.linalg.norm(nuisance, axis=0, keepdims=True)
    nuisance *= nuisance_scale * np.linspace(4.2, 2.0, nuisance_rank)

    factors = np.hstack([dominant, nuisance])
    signal = factors @ factors.T
    noise_matrix = rng.standard_normal((n, n))
    noise_matrix = (noise_matrix + noise_matrix.T) * 0.5
    s = signal + noise * np.std(signal) * noise_matrix
    s -= min(0.0, float(s.min()))
    return (s + s.T) * 0.5
