"""Cross-validation for symmetric matrix completion."""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import warnings
from typing import Generator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from .coherence._sampling_fraction import adaptive_cap
from .model import SRF


def cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: list[int],
    sampling_fraction: float,
    n_folds: int = 5,
    random_state: int = 0,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
) -> pd.DataFrame:
    """K-fold cross-validation of SRF rank.

    Pre-masks the observed entries of ``similarity_matrix`` at an
    outer probability inflated to ``sampling_fraction * k/(k-1)``
    (capped at ``max(0.95, 1 - 2000 / N_pairs)``), partitions the kept
    entries into ``n_folds`` disjoint symmetric folds, and for every
    (fold, rank) pair fits an :class:`SRF` on training entries and
    scores reconstruction MSE on the held-out entries. Diagonal entries
    are kept in training and never validated.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n, n)
        Symmetric similarity matrix. Missing entries marked according
        to ``missing_values``.
    ranks : list of int
        Candidate ranks to evaluate.
    sampling_fraction : float
        Per-fold training density. Typically
        ``estimate_rank(s).sampling_fraction`` from
        :func:`pysrf.estimate_rank`.
    n_folds : int, default=5
    random_state : int, default=0
    n_jobs : int, default=-1
    missing_values : float or None, default=np.nan

    Returns
    -------
    curve : DataFrame
        Long-format with columns ``rank``, ``fold``, ``val_mse``.
        Aggregate with ``curve.groupby("rank")["val_mse"].mean()``
        for the CV curve.
    """
    if not (0.0 < sampling_fraction < 1.0):
        raise ValueError("sampling_fraction must be in (0, 1)")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    s = np.asarray(similarity_matrix)
    bounds = (float(np.nanmin(s)), float(np.nanmax(s)))
    splits = list(_kfold_entry_splits(
        s, sampling_fraction, n_folds, random_state, missing_values,
    ))

    jobs = [
        (rank, fold_idx, train_mask, val_mask)
        for fold_idx, (train_mask, val_mask) in enumerate(splits)
        for rank in ranks
    ]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(
            s, train_mask, val_mask, int(rank), bounds,
            random_state + 7919 * fold_idx + 13 * int(rank),
        )
        for (rank, fold_idx, train_mask, val_mask) in jobs
    )
    return pd.DataFrame([
        {"rank": int(rank), "fold": fold_idx, "val_mse": score}
        for (rank, fold_idx, _, _), score in zip(jobs, scores)
    ])


def _kfold_entry_splits(
    s: np.ndarray,
    sampling_fraction: float,
    n_folds: int,
    random_state: int | None,
    missing_values: float | None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield ``(train_mask, val_mask)`` for ``n_folds``-fold partition CV.

    Pre-masks at the inflated outer probability (capped via
    :func:`adaptive_cap`), then partitions the kept upper-triangle
    entries into disjoint folds. Diagonals are always in training and
    never in validation.
    """
    rng = check_random_state(random_state)
    n = s.shape[0]
    cap = adaptive_cap(n)
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
    p_outer = min(unclipped, cap)

    rows, cols, eligible = _eligible_pair_positions(s, missing_values)
    pool = eligible[rng.uniform(size=eligible.size) < p_outer]
    fold_positions = [np.asarray(g, dtype=int)
                       for g in np.array_split(rng.permutation(pool), n_folds)]
    diag_in_training = _observed_diagonal_indices(s, missing_values)

    for val_positions in fold_positions:
        train_positions = np.setdiff1d(pool, val_positions, assume_unique=False)
        train_mask = np.zeros(s.shape, dtype=bool)
        val_mask = np.zeros(s.shape, dtype=bool)
        train_mask[rows[train_positions], cols[train_positions]] = True
        train_mask[cols[train_positions], rows[train_positions]] = True
        val_mask[rows[val_positions], cols[val_positions]] = True
        val_mask[cols[val_positions], rows[val_positions]] = True
        train_mask[diag_in_training, diag_in_training] = True
        yield train_mask, val_mask


def _fit_score(
    s: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    rank: int,
    bounds: tuple[float, float],
    seed: int,
) -> float:
    """Fit SRF at ``rank`` on training entries, score V-MSE on validation."""
    x_train = np.full_like(s, np.nan)
    x_train[train_mask] = s[train_mask]
    est = SRF(rank=rank, bounds=bounds, missing_values=np.nan, random_state=seed)
    est.fit(x_train)
    rec = est.reconstruct()
    if not val_mask.any():
        return float("nan")
    return float(np.mean((s[val_mask] - rec[val_mask]) ** 2))


def _eligible_pair_positions(
    s: np.ndarray, missing_values: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle indices of originally-observed entries."""
    if missing_values is None or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    ):
        observed = np.isfinite(s)
    else:
        observed = np.not_equal(s, missing_values)
    rows, cols = np.triu_indices_from(s, k=1)
    eligible = np.where(observed[rows, cols])[0]
    return rows, cols, eligible


def _observed_diagonal_indices(
    s: np.ndarray, missing_values: float | None,
) -> np.ndarray:
    """Indices of originally-observed diagonal entries."""
    diag = np.diag(s)
    if missing_values is None or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    ):
        observed = np.isfinite(diag)
    else:
        observed = np.not_equal(diag, missing_values)
    return np.where(observed)[0]
