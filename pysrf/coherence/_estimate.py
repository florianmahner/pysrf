"""Public function and dataclass for rank estimation."""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ._bootstrap import (
    bootstrap_coherence,
    observation_mask,
    reference_eigenpairs,
    symmetrize,
)
from ._rank_selection import changepoint, leakage_profile
from ._sampling_fraction import detectability_floor, invert_recovery, recovery_curve


@dataclass(frozen=True)
class RankEstimate:
    """Output of :func:`estimate_rank`.

    Attributes
    ----------
    rank : int
        Estimated number of signal dimensions.
    sampling_fraction : float
        Minimum per-fold training density needed to recover the signal
        eigenspace within ``recovery_tolerance``. Suitable as the
        ``sampling_fraction`` argument of :func:`pysrf.cross_val_score`.
    eigenvalues : ndarray of shape (max_rank,)
        Top-``max_rank`` reference eigenvalues, descending.
    leakage : ndarray of shape (max_rank,)
        Per-dimension instability score. Signal dimensions have small
        leakage; noise dimensions have large leakage.
    sampling_grid : ndarray of shape (P,)
        Sampling probabilities at which the bootstrap was evaluated.
    recovery_raw : ndarray of shape (P,)
        Empirical recovery deficit at each ``sampling_grid`` point.
    recovery_monotone : ndarray of shape (P,)
        Non-increasing projection of ``recovery_raw``. The estimator
        inverts this curve at ``recovery_tolerance`` to get
        ``sampling_fraction``.
    detectability_floor : float
        Random-matrix lower bound on ``sampling_fraction``. If
        ``sampling_fraction`` equals this value, the signal is at the
        detection limit and the rank may be unstable.
    n_features_in : int
        Size of the input matrix.
    """

    rank: int
    sampling_fraction: float
    eigenvalues: np.ndarray
    leakage: np.ndarray
    sampling_grid: np.ndarray
    recovery_raw: np.ndarray
    recovery_monotone: np.ndarray
    detectability_floor: float
    n_features_in: int


def estimate_rank(
    s: np.ndarray,
    recovery_tolerance: float = 0.10,
    max_rank: int | None = None,
    sampling_grid: np.ndarray | None = None,
    n_bootstrap: int = 20,
    high_band_quantile: float = 0.85,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> RankEstimate:
    """Estimate the number of signal dimensions of a symmetric similarity matrix.

    The method bootstraps the top eigenspace under random off-diagonal
    masking, finds the F-statistic changepoint of the per-dimension
    leakage profile to pick the rank, and inverts the recovery curve
    at ``recovery_tolerance`` (floored by a random-matrix
    detectability bound) to pick the calibrated sampling fraction.

    Parameters
    ----------
    s : array-like of shape (n, n)
        Symmetric similarity matrix. Missing entries may be marked
        with NaN.
    recovery_tolerance : float, default=0.10
        Maximum fraction of top-rank spectral mass allowed to be
        missing at the calibrated sampling fraction.
    max_rank : int or None, default=None
        Largest candidate rank to test. Defaults to ``min(n // 4, 100)``.
    sampling_grid : array-like or None, default=None
        Strictly-increasing sampling probabilities in (0, 1] at which
        to evaluate eigenspace stability. Defaults to
        ``np.linspace(0.05, 0.95, 20)``.
    n_bootstrap : int, default=20
    high_band_quantile : float, default=0.85
        Quantile of ``sampling_grid`` defining the high-p band used
        to aggregate the per-rank leakage score.
    random_state : int, default=0
    n_jobs : int or None, default=None
        Number of parallel workers. ``None`` uses ``cpu_count - 1``.

    Returns
    -------
    estimate : RankEstimate
    """
    s_sym = symmetrize(np.asarray(s, dtype=np.float64))
    n = s_sym.shape[0]

    k_max = _resolve_max_rank(n, max_rank)
    grid = _resolve_sampling_grid(sampling_grid)
    jobs = _resolve_n_jobs(n_jobs)
    seed = int(random_state)

    s_filled, mask, observed_rate = observation_mask(s_sym)
    eigenvalues, u_ref = reference_eigenpairs(s_filled, mask, observed_rate, k_max, seed)

    overlap, projected = bootstrap_coherence(
        s_filled, mask, u_ref, k_max, grid,
        n_bootstrap=n_bootstrap, random_state=seed, n_jobs=jobs,
    )
    overlap_median = np.median(overlap, axis=2)
    projected_median = np.median(projected, axis=2)

    leakage = leakage_profile(overlap_median, grid, high_band_quantile)
    rank = changepoint(leakage)

    p_sorted, recovery_raw, recovery_monotone = recovery_curve(
        grid, eigenvalues, projected_median, rank
    )
    raw_fraction = invert_recovery(p_sorted, recovery_monotone, recovery_tolerance)
    floor = detectability_floor(eigenvalues, rank)
    sampling_fraction = float(max(raw_fraction, floor))

    return RankEstimate(
        rank=int(rank),
        sampling_fraction=sampling_fraction,
        eigenvalues=eigenvalues,
        leakage=leakage,
        sampling_grid=p_sorted,
        recovery_raw=recovery_raw,
        recovery_monotone=recovery_monotone,
        detectability_floor=float(floor),
        n_features_in=int(n),
    )


def _resolve_max_rank(n: int, max_rank: int | None) -> int:
    if max_rank is None:
        return max(min(n // 4, 100), 2)
    return int(max_rank)


def _resolve_sampling_grid(sampling_grid: np.ndarray | None) -> np.ndarray:
    if sampling_grid is None:
        return np.linspace(0.05, 0.95, 20)
    return np.asarray(sampling_grid, dtype=np.float64)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    cpu = os.cpu_count() or 2
    if n_jobs is None:
        return max(1, cpu - 1)
    return max(1, min(int(n_jobs), cpu - 1))
