"""Eigenspace coherence for dimensionality estimation.

Estimates the number of signal dimensions (k*) in a symmetric
similarity matrix by measuring how stable each eigenspace
dimension is under random entry masking.

The method works by:
1. Computing a reference eigenspace from the full matrix
2. Repeatedly masking entries at rate p and re-computing eigenvectors
3. Measuring overlap (Iproj) between bootstrap and reference eigenvectors
4. Estimating scaled leakage (kappa) per dimension
5. Finding the changepoint where kappa jumps from signal to noise
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Matrix preparation
# ---------------------------------------------------------------------------


def _symmetrize(s: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix while preserving NaN semantics.

    For each pair (i, j):
    - average if both finite
    - copy the finite value if only one side is finite
    - keep NaN if both missing

    Diagonal NaNs are replaced with zero.
    """
    s = np.asarray(s, dtype=float)
    st = s.T
    a = np.isfinite(s)
    b = np.isfinite(st)

    out = np.full_like(s, np.nan, dtype=float)

    both = a & b
    out[both] = 0.5 * (s[both] + st[both])

    only_a = a & ~b
    out[only_a] = s[only_a]

    only_b = ~a & b
    out[only_b] = st[only_b]

    d = np.diag(out).copy()
    d[~np.isfinite(d)] = 0.0
    np.fill_diagonal(out, d)
    return out


def _observation_mask(
    s_sym: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build observation mask from a symmetrized matrix.

    Parameters
    ----------
    s_sym : (n, n) array
        Symmetrized similarity matrix (may contain NaN).

    Returns
    -------
    s_filled : (n, n) array
        Input with NaN replaced by zero.
    mask : (n, n) array
        Symmetric 0/1 observation mask with diagonal forced to 1.
    obs_rate : float
        Off-diagonal observation rate, clipped to [eps, 1].
    """
    n = s_sym.shape[0]
    mask = np.isfinite(s_sym).astype(float)
    mask = ((mask + mask.T) > 0).astype(float)
    np.fill_diagonal(mask, 1.0)

    s_filled = np.nan_to_num(s_sym, nan=0.0)

    if n <= 1:
        obs_rate = 1.0
    else:
        iu = np.triu_indices(n, k=1)
        obs_rate = float(mask[iu].mean()) if iu[0].size else 1.0
    obs_rate = float(np.clip(obs_rate, eps, 1.0))

    return s_filled, mask, obs_rate
