"""Cross-validation for symmetric matrix completion.

Mirrors ``update_pysrf/src/experiment_cvrank_at_pstar.py`` (the
reference experiment) line-for-line:

- ``_mask_missing_entries``: exact-count subsample of upper-triangle
  off-diagonal positions, with the diagonal kept observed.
- ``_split_observed_into_folds``: random partition of the observed
  pool into ``n_folds`` symmetric off-diagonal masks.
- Per fold: ``holdout = V | M_outer``; ``train_mask = ~holdout &
  off_diag``; ``val_mask = V & off_diag``. The diagonal is never in
  training or validation.
- ``_fit_score``: build NaN-masked ``x_train`` from ``train_mask``,
  fit SRF, score V-MSE on upper-triangle of ``val_mask`` with finite
  checks on both ``S`` and the reconstruction.
"""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from .coherence._sampling_fraction import _adaptive_cap
from .model import SRF


def cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: list[int],
    sampling_fraction: float,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: int = 0,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
    srf_kwargs: dict | None = None,
) -> pd.DataFrame:
    """K-fold cross-validation of SRF rank.

    Pre-masks observed entries at an outer probability inflated to
    ``sampling_fraction * k/(k-1)`` (capped at
    ``max(0.95, 1 - 2000 / N_pairs)``), partitions the observed pool
    into ``n_folds`` disjoint symmetric folds, and for every (fold,
    rank) pair fits an :class:`SRF` on training entries and scores
    reconstruction MSE on the held-out entries. With ``n_repeats > 1``
    the whole procedure repeats with a different outer mask realisation
    each rep. Diagonal entries are never in training or validation.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n, n)
    ranks : list of int
    sampling_fraction : float
        Per-fold training density, typically from
        ``estimate_rank(s).sampling_fraction``.
    n_folds : int, default=5
    n_repeats : int, default=1
    random_state : int, default=0
    n_jobs : int, default=-1
    missing_values : float or None, default=np.nan
    srf_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to :class:`SRF`. Must not
        contain ``rank``, ``bounds``, ``missing_values``, or
        ``random_state`` (those are set per fit).

    Returns
    -------
    curve : DataFrame
        Columns ``rep``, ``fold``, ``rank``, ``val_mse``.
    """
    if not (0.0 < sampling_fraction < 1.0):
        raise ValueError("sampling_fraction must be in (0, 1)")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    srf_kwargs = dict(srf_kwargs) if srf_kwargs else {}
    reserved = {"rank", "bounds", "missing_values", "random_state"} & srf_kwargs.keys()
    if reserved:
        raise ValueError(
            f"srf_kwargs must not contain {sorted(reserved)}; these are set per fit"
        )

    s = np.asarray(similarity_matrix)
    n = s.shape[0]
    bounds = (float(np.nanmin(s)), float(np.nanmax(s)))
    off_diag = ~np.eye(n, dtype=bool)
    p_outer = _outer_mask_probability(sampling_fraction, n_folds, n)

    seeds = np.random.SeedSequence(random_state).generate_state(n_repeats)

    jobs = []  # (rep, fold_idx, rank, train_mask, val_mask, fit_seed)
    for rep, rep_seed in enumerate(seeds):
        rng = check_random_state(int(rep_seed))
        m_outer = _mask_missing_entries(s, p_outer, rng, missing_values)
        val_folds = _split_observed_into_folds(m_outer, n_folds, rng)
        if val_folds is None:
            continue
        for fold_idx, v in enumerate(val_folds):
            holdout = v | m_outer
            train_mask = (~holdout) & off_diag
            val_mask = v & off_diag
            for rank in ranks:
                fit_seed = (
                    7919 * (rep + 1) + 101 * fold_idx + 13 * int(rank)
                ) & 0xFFFFFFFF
                jobs.append((rep, fold_idx, int(rank),
                              train_mask, val_mask, fit_seed))

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(s, tm, vm, rank, bounds, seed, srf_kwargs)
        for (_, _, rank, tm, vm, seed) in jobs
    )
    return pd.DataFrame([
        {"rep": rep, "fold": fold_idx, "rank": rank, "val_mse": score}
        for (rep, fold_idx, rank, _, _, _), score in zip(jobs, scores)
    ])


def _outer_mask_probability(sampling_fraction: float, n_folds: int, n: int) -> float:
    """Inflated, capped outer-mask probability. Warns if cap binds."""
    cap = _adaptive_cap(n)
    unclipped = sampling_fraction * n_folds / (n_folds - 1)
    if unclipped > cap:
        effective = cap * (n_folds - 1) / n_folds
        warnings.warn(
            f"Inflated outer mask {unclipped:.3f} for n_folds={n_folds} "
            f"exceeds the cap {cap:.3f}. Each fold will train at "
            f"{effective:.3f} instead of the requested {sampling_fraction:.3f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return min(unclipped, cap)


def _mask_missing_entries(
    x: np.ndarray,
    observed_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None,
) -> np.ndarray:
    """Symmetric outer mask. True = missing.

    Subsamples exactly ``int(observed_fraction * N_pairs)`` off-diagonal
    upper-triangle positions to keep observed; the rest are masked.
    Diagonal is always observed. Ported from
    ``update_pysrf/src/symmnmf/cross_validation.mask_missing_entries``.
    """
    if missing_values is None or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    ):
        observed = np.isfinite(x)
    else:
        observed = np.not_equal(x, missing_values)
    triu_i, triu_j = np.triu_indices_from(x, k=1)
    valid = np.where(observed[triu_i, triu_j])[0]

    n_keep = int(observed_fraction * len(valid))
    if n_keep == 0:
        return np.ones_like(x, dtype=bool)

    keep = rng.choice(valid, size=n_keep, replace=False)
    missing = np.ones_like(x, dtype=bool)
    missing[triu_i[keep], triu_j[keep]] = False
    missing[triu_j[keep], triu_i[keep]] = False
    np.fill_diagonal(missing, False)
    return missing


def _split_observed_into_folds(
    m_outer: np.ndarray,
    n_folds: int,
    rng: np.random.RandomState,
) -> list[np.ndarray] | None:
    """Random partition of observed off-diagonal pairs into ``n_folds`` symmetric masks.

    Returns ``None`` if the observed pool has fewer than ``n_folds``
    entries.
    """
    n = m_outer.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    observed = ~m_outer[triu_i, triu_j]
    valid = np.where(observed)[0]
    if len(valid) < n_folds:
        return None
    perm = rng.permutation(len(valid))
    folds = []
    for group in np.array_split(perm, n_folds):
        mask = np.zeros((n, n), dtype=bool)
        ii, jj = triu_i[valid[group]], triu_j[valid[group]]
        mask[ii, jj] = True
        mask[jj, ii] = True
        folds.append(mask)
    return folds


def _fit_score(
    s: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    rank: int,
    bounds: tuple[float, float],
    seed: int,
    srf_kwargs: dict,
) -> float:
    """Fit SRF on training entries, score V-MSE on upper-triangle of val_mask."""
    x_train = np.full_like(s, np.nan)
    x_train[train_mask] = s[train_mask]
    est = SRF(rank=rank, bounds=bounds, missing_values=np.nan,
              random_state=seed, **srf_kwargs)
    est.fit(x_train)
    s_hat = est.reconstruct()

    n = s.shape[0]
    iu = np.triu_indices(n, k=1)
    m = val_mask[iu] & np.isfinite(s[iu]) & np.isfinite(s_hat[iu])
    if not m.any():
        return float("nan")
    return float(np.mean((s[iu][m] - s_hat[iu][m]) ** 2))
