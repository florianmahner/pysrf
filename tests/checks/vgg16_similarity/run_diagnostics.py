from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pysrf import SRF
from pysrf._common import (
    observation_mask,
    replace_missing_with_nan,
    symmetrize_observations,
)
from pysrf.coherence import CVCalibration, calibrate_cross_validation
from pysrf.cross_validation import (
    _cv_seeds,
    _entrywise_splits,
    _summarize,
    _training_fraction,
    _validation_mse,
    get_fold_fraction,
)

DATA_PATH = ROOT / "tests" / "vgg16_similarity.npy"
OUT_ROOT = Path(__file__).resolve().parent
DEFAULT_RANKS = (50, 100, 200, 300, 400)
DEFAULT_SAMPLING_GRID = np.linspace(0.05, 0.95, 20)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _write_progress_event(out_dir: Path, payload: dict) -> None:
    progress_dir = out_dir / "progress_events"
    progress_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        **payload,
    }
    stem = f"{time.time_ns()}_{os.getpid()}_{payload.get('event', 'event')}"
    _write_json(progress_dir / f"{stem}.json", event)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_similarity() -> np.ndarray:
    s = np.load(DATA_PATH)
    return symmetrize_observations(replace_missing_with_nan(s, np.nan))


def _calibration_scalars(calibration: CVCalibration, seconds: float, seed: int) -> dict:
    return {
        "seed": int(seed),
        "seconds": float(seconds),
        "spectral_cutoff": int(calibration.spectral_cutoff),
        "sampling_fraction": float(calibration.sampling_fraction),
        "detectability_floor": float(calibration.detectability_floor),
        "n_features_in": int(calibration.n_features_in),
    }


def _save_calibration_arrays(
    calibration: CVCalibration, out_dir: Path, label: str
) -> None:
    pd.DataFrame(
        {
            "dimension": np.arange(1, calibration.eigvals.size + 1),
            "eigenvalue": calibration.eigvals,
            "leakage": calibration.leakage,
        }
    ).to_csv(out_dir / f"{label}_spectrum_and_leakage.csv", index=False)
    pd.DataFrame(
        {
            "sampling_fraction_grid": calibration.sampling_grid,
            "signal_loss_raw": calibration.signal_loss_raw,
            "signal_loss_monotone": calibration.signal_loss_monotone,
        }
    ).to_csv(out_dir / f"{label}_signal_loss.csv", index=False)


def run_matrix_overview() -> None:
    out_dir = OUT_ROOT / "matrix_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    s = _load_similarity()
    n = s.shape[0]
    diag = np.diag(s)
    triu = np.triu_indices(n, k=1)
    offdiag = s[triu]
    row_offdiag_mean = (s.sum(axis=1) - diag) / (n - 1)

    t0 = time.perf_counter()
    eigvals = np.linalg.eigvalsh(s)[::-1]
    eig_seconds = time.perf_counter() - t0
    cumulative_trace = np.cumsum(eigvals) / eigvals.sum()

    rank_records = []
    for rank in DEFAULT_RANKS:
        rank_records.append(
            {
                "rank": rank,
                "trace_mass": float(cumulative_trace[rank - 1]),
                "eigenvalue": float(eigvals[rank - 1]),
            }
        )

    summary = {
        "data_path": DATA_PATH,
        "shape": s.shape,
        "dtype": str(s.dtype),
        "finite_count": int(np.isfinite(s).sum()),
        "symmetry_max_abs": float(np.max(np.abs(s - s.T))),
        "diag_min": float(diag.min()),
        "diag_median": float(np.median(diag)),
        "diag_mean": float(diag.mean()),
        "diag_max": float(diag.max()),
        "offdiag_min": float(offdiag.min()),
        "offdiag_median": float(np.median(offdiag)),
        "offdiag_mean": float(offdiag.mean()),
        "offdiag_std": float(offdiag.std()),
        "offdiag_max": float(offdiag.max()),
        "diag_to_offdiag_mean_ratio": float(diag.mean() / offdiag.mean()),
        "diag_to_offdiag_median_ratio": float(np.median(diag) / np.median(offdiag)),
        "diag_row_offdiag_mean_corr": float(np.corrcoef(diag, row_offdiag_mean)[0, 1]),
        "min_eigenvalue": float(eigvals[-1]),
        "max_eigenvalue": float(eigvals[0]),
        "positive_eigenvalue_count_1e_8": int(np.sum(eigvals > 1e-8)),
        "eigendecomposition_seconds": float(eig_seconds),
    }

    _write_json(out_dir / "matrix_summary.json", summary)
    pd.DataFrame(
        {
            "dimension": np.arange(1, n + 1),
            "eigenvalue": eigvals,
            "cumulative_trace_mass": cumulative_trace,
        }
    ).to_csv(out_dir / "eigenvalues.csv", index=False)
    pd.DataFrame(rank_records).to_csv(
        out_dir / "trace_mass_at_requested_ranks.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("VGG16 Linear-Kernel Similarity Overview", fontsize=14)

    ax = axes[0, 0]
    image = ax.imshow(
        s, cmap="viridis", interpolation="nearest", vmax=np.quantile(s, 0.995)
    )
    ax.set_title("Similarity matrix")
    ax.set_xlabel("item")
    ax.set_ylabel("item")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    ax.semilogy(np.arange(1, n + 1), eigvals, color="#176D6A", linewidth=1.8)
    for rank in DEFAULT_RANKS:
        ax.axvline(rank, color="#6B7280", alpha=0.35, linewidth=1)
    ax.set_title("PSD spectrum")
    ax.set_xlabel("dimension")
    ax.set_ylabel("eigenvalue")

    ax = axes[1, 0]
    ax.plot(np.arange(1, n + 1), cumulative_trace, color="#4B5EAA", linewidth=2)
    for rank in DEFAULT_RANKS:
        ax.axvline(rank, color="#6B7280", alpha=0.35, linewidth=1)
    ax.set_ylim(0, 1.01)
    ax.set_title("Cumulative trace mass")
    ax.set_xlabel("rank")
    ax.set_ylabel("trace mass")

    ax = axes[1, 1]
    ax.hist(offdiag, bins=80, color="#8A8F98", alpha=0.75, label="off diagonal")
    ax.axvline(float(diag.mean()), color="#A64B2A", linewidth=2, label="diag mean")
    ax.axvline(
        float(offdiag.mean()), color="#176D6A", linewidth=2, label="offdiag mean"
    )
    ax.set_title("Similarity value scale")
    ax.set_xlabel("similarity")
    ax.set_ylabel("count")
    ax.legend(frameon=False)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "matrix_overview.png", dpi=160)
    plt.close(fig)
    print(f"matrix overview saved: {out_dir}")


def _run_calibration(
    s: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
    n_jobs: int,
    max_eigenpairs: int | None,
) -> tuple[CVCalibration, float]:
    t0 = time.perf_counter()
    calibration = calibrate_cross_validation(
        s,
        max_eigenpairs=max_eigenpairs,
        sampling_grid=DEFAULT_SAMPLING_GRID,
        n_bootstrap=n_bootstrap,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return calibration, time.perf_counter() - t0


def run_sampling_calibration(
    *,
    seed: int,
    primary_bootstrap: int,
    estimate_bootstrap: int,
    estimate_seeds: tuple[int, ...],
    n_jobs: int,
    max_eigenpairs: int | None,
) -> None:
    out_dir = OUT_ROOT / "sampling_fraction_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _load_similarity()

    _write_progress_event(
        out_dir,
        {
            "event": "primary_calibration_started",
            "seed": int(seed),
            "n_bootstrap": int(primary_bootstrap),
            "max_eigenpairs": max_eigenpairs,
        },
    )
    primary, primary_seconds = _run_calibration(
        s,
        seed=seed,
        n_bootstrap=primary_bootstrap,
        n_jobs=n_jobs,
        max_eigenpairs=max_eigenpairs,
    )
    _write_progress_event(
        out_dir,
        {
            "event": "primary_calibration_finished",
            "seed": int(seed),
            "n_bootstrap": int(primary_bootstrap),
            "spectral_cutoff": int(primary.spectral_cutoff),
            "sampling_fraction": float(primary.sampling_fraction),
            "seconds": float(primary_seconds),
        },
    )
    _save_calibration_arrays(primary, out_dir, "primary")
    primary_payload = {
        **_calibration_scalars(primary, primary_seconds, seed),
        "n_bootstrap": int(primary_bootstrap),
        "max_eigenpairs": max_eigenpairs,
        "sampling_grid": DEFAULT_SAMPLING_GRID,
    }
    _write_json(out_dir / "primary_calibration.json", primary_payload)

    estimate_rows = []
    curve_rows = []
    for estimate_seed in estimate_seeds:
        _write_progress_event(
            out_dir,
            {
                "event": "estimate_calibration_started",
                "seed": int(estimate_seed),
                "n_bootstrap": int(estimate_bootstrap),
            },
        )
        calibration, seconds = _run_calibration(
            s,
            seed=estimate_seed,
            n_bootstrap=estimate_bootstrap,
            n_jobs=n_jobs,
            max_eigenpairs=max_eigenpairs,
        )
        _write_progress_event(
            out_dir,
            {
                "event": "estimate_calibration_finished",
                "seed": int(estimate_seed),
                "n_bootstrap": int(estimate_bootstrap),
                "spectral_cutoff": int(calibration.spectral_cutoff),
                "sampling_fraction": float(calibration.sampling_fraction),
                "seconds": float(seconds),
            },
        )
        estimate_rows.append(
            {
                **_calibration_scalars(calibration, seconds, estimate_seed),
                "n_bootstrap": int(estimate_bootstrap),
            }
        )
        for p, raw, monotone in zip(
            calibration.sampling_grid,
            calibration.signal_loss_raw,
            calibration.signal_loss_monotone,
        ):
            curve_rows.append(
                {
                    "seed": int(estimate_seed),
                    "sampling_fraction_grid": float(p),
                    "signal_loss_raw": float(raw),
                    "signal_loss_monotone": float(monotone),
                }
            )

    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(out_dir / "sampling_fraction_estimates.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        out_dir / "sampling_fraction_estimate_curves.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("VGG16 CV Sampling-Fraction Calibration", fontsize=14)

    dims = np.arange(1, primary.eigvals.size + 1)
    ax = axes[0, 0]
    ax.semilogy(dims, primary.eigvals, color="#176D6A", linewidth=2)
    ax.axvline(primary.spectral_cutoff, color="#A64B2A", linestyle="--", linewidth=2)
    ax.set_title("Primary spectrum")
    ax.set_xlabel("dimension")
    ax.set_ylabel("eigenvalue")

    ax = axes[0, 1]
    ax.plot(dims, primary.leakage, color="#4B5EAA", linewidth=2)
    ax.axvline(primary.spectral_cutoff, color="#A64B2A", linestyle="--", linewidth=2)
    ax.set_title("Primary leakage")
    ax.set_xlabel("dimension")
    ax.set_ylabel("leakage")

    ax = axes[1, 0]
    if curve_rows:
        curves = pd.DataFrame(curve_rows)
        for _, curve in curves.groupby("seed"):
            ax.plot(
                curve["sampling_fraction_grid"],
                curve["signal_loss_monotone"],
                color="#8A8F98",
                alpha=0.5,
                linewidth=1.4,
            )
    ax.plot(
        primary.sampling_grid,
        primary.signal_loss_monotone,
        color="#176D6A",
        linewidth=2.6,
        label=f"primary p={primary.sampling_fraction:.3f}",
    )
    ax.axhline(
        0.10, color="#C59A2D", linestyle=":", linewidth=1.8, label="loss tolerance"
    )
    ax.axvline(
        primary.detectability_floor, color="#6B7280", linestyle="--", linewidth=1.4
    )
    ax.axvline(primary.sampling_fraction, color="#A64B2A", linestyle="--", linewidth=2)
    ax.set_title("Signal-loss estimates")
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("monotone signal loss")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    if not estimates.empty:
        y = np.zeros(len(estimates))
        ax.scatter(
            estimates["sampling_fraction"],
            y,
            color="#8A8F98",
            s=50,
            label="repeat estimates",
        )
        for _, row in estimates.iterrows():
            ax.text(
                row["sampling_fraction"],
                0.04,
                str(int(row["seed"])),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axvline(
        primary.sampling_fraction, color="#A64B2A", linewidth=2.5, label="primary"
    )
    ax.set_ylim(-0.2, 0.35)
    ax.set_yticks([])
    ax.set_title("Sampling-fraction variability")
    ax.set_xlabel("selected sampling fraction")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "sampling_fraction_calibration.png", dpi=160)
    plt.close(fig)

    print(
        "primary calibration: "
        f"spectral_cutoff={primary.spectral_cutoff}, "
        f"sampling_fraction={primary.sampling_fraction:.4f}, "
        f"seconds={primary_seconds:.1f}"
    )
    print(f"sampling calibration saved: {out_dir}")


def _masked_offdiag_mse(s: np.ndarray, s_hat: np.ndarray, mask: np.ndarray) -> float:
    triu = np.triu_indices(s.shape[0], k=1)
    scored = mask[triu] & np.isfinite(s[triu]) & np.isfinite(s_hat[triu])
    if not scored.any():
        return float("nan")
    residual = s[triu][scored] - s_hat[triu][scored]
    return float(np.mean(residual * residual))


def _history_frame(
    history: dict[str, list], repeat: int, fold: int, rank: int
) -> pd.DataFrame:
    frame = pd.DataFrame(history)
    frame.insert(0, "outer_iteration", np.arange(1, len(frame) + 1))
    frame.insert(0, "candidate_rank", int(rank))
    frame.insert(0, "fold", int(fold))
    frame.insert(0, "repeat", int(repeat))
    return frame


def _fit_rank_curve_job(
    s: np.ndarray,
    split,
    rank: int,
    seed: int,
    fit_kwargs: dict,
    embeddings_dir: Path,
    histories_dir: Path,
    partial_results_dir: Path,
    progress_out_dir: Path,
    pin_threads: bool,
) -> dict:
    job_id = f"rank_{rank}_repeat_{split.repeat}_fold_{split.fold}"
    _write_progress_event(
        progress_out_dir,
        {
            "event": "rank_curve_fit_started",
            "job_id": job_id,
            "candidate_rank": int(rank),
            "repeat": int(split.repeat),
            "fold": int(split.fold),
            "fit_seed": int(seed),
        },
    )
    limit = threadpool_limits(limits=1) if pin_threads else nullcontext()
    try:
        with limit:
            train = np.full(s.shape, np.nan, dtype=np.float64)
            train[split.train_mask] = s[split.train_mask]

            t0 = time.perf_counter()
            est = SRF(
                rank=rank,
                missing_values=np.nan,
                random_state=seed,
                **fit_kwargs,
            ).fit(train)
            seconds = time.perf_counter() - t0

            s_hat = est.reconstruct()
            val_mse = _validation_mse(s, s_hat, split.validation_mask)
            train_mse = _masked_offdiag_mse(s, s_hat, split.train_mask)
            all_offdiag_mse = _masked_offdiag_mse(s, s_hat, observation_mask(s))

            stem = f"rank_{rank}_repeat_{split.repeat}_fold_{split.fold}"
            embedding_path = embeddings_dir / f"{stem}.npy"
            history_path = histories_dir / f"{stem}_history.csv"
            np.save(embedding_path, est.w_)
            _history_frame(est.history_, split.repeat, split.fold, rank).to_csv(
                history_path,
                index=False,
            )

            history = est.history_
            evar = np.asarray(history.get("evar", []), dtype=float)
            rec_error = np.asarray(history.get("rec_error", []), dtype=float)
            converged = np.asarray(history.get("converged", [False]), dtype=bool)
            tail = min(10, evar.size)
            evar_gain_last_tail = float(evar[-1] - evar[-tail]) if tail else float("nan")
            rec_error_drop_last_tail = (
                float(rec_error[-tail] - rec_error[-1]) if tail else float("nan")
            )

        result = {
            "repeat": int(split.repeat),
            "fold": int(split.fold),
            "candidate_rank": int(rank),
            "fit_seed": int(seed),
            "val_mse": float(val_mse),
            "train_mse": float(train_mse),
            "all_offdiag_mse": float(all_offdiag_mse),
            "fit_seconds": float(seconds),
            "n_iter": int(est.n_iter_),
            "last_evar": float(evar[-1]) if evar.size else float("nan"),
            "last_rec_error": float(rec_error[-1]) if rec_error.size else float("nan"),
            "evar_gain_last_10": evar_gain_last_tail,
            "rec_error_drop_last_10": rec_error_drop_last_tail,
            "converged": bool(converged[-1]) if converged.size else False,
            "embedding_path": str(embedding_path.relative_to(OUT_ROOT)),
            "history_path": str(history_path.relative_to(OUT_ROOT)),
        }
        partial_results_dir.mkdir(parents=True, exist_ok=True)
        _write_json(partial_results_dir / f"{stem}_result.json", result)
        _write_progress_event(
            progress_out_dir,
            {
                "event": "rank_curve_fit_finished",
                "job_id": job_id,
                "candidate_rank": int(rank),
                "repeat": int(split.repeat),
                "fold": int(split.fold),
                "val_mse": float(val_mse),
                "fit_seconds": float(seconds),
                "last_evar": result["last_evar"],
                "converged": result["converged"],
            },
        )
        return result
    except Exception as exc:
        _write_progress_event(
            progress_out_dir,
            {
                "event": "rank_curve_fit_failed",
                "job_id": job_id,
                "candidate_rank": int(rank),
                "repeat": int(split.repeat),
                "fold": int(split.fold),
                "error": repr(exc),
            },
        )
        raise


def _load_primary_sampling_fraction() -> float:
    path = OUT_ROOT / "sampling_fraction_calibration" / "primary_calibration.json"
    if not path.exists():
        raise FileNotFoundError(
            "Run the sampling calibration first, or use --section all."
        )
    return float(_read_json(path)["sampling_fraction"])


def run_linear_rank_curve(
    *,
    ranks: tuple[int, ...],
    seed: int,
    n_folds: int,
    n_repeats: int,
    n_jobs: int,
    max_outer: int,
    max_inner: int,
    tol: float,
) -> None:
    out_dir = OUT_ROOT / "linear_kernel_rank_curve"
    embeddings_dir = out_dir / "embeddings"
    histories_dir = out_dir / "fit_histories"
    partial_results_dir = out_dir / "partial_results"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    partial_results_dir.mkdir(parents=True, exist_ok=True)

    s = _load_similarity()
    n = s.shape[0]
    sampling_fraction = _load_primary_sampling_fraction()
    fold_fraction = get_fold_fraction(sampling_fraction, n_folds, n)
    split_seeds, fit_seeds = _cv_seeds(seed, n_repeats, n_folds, len(ranks))
    splits = _entrywise_splits(
        observation_mask(s),
        fold_fraction=fold_fraction,
        n_folds=n_folds,
        split_seeds=split_seeds,
    )

    fit_kwargs = {
        "max_outer": int(max_outer),
        "max_inner": int(max_inner),
        "tol": float(tol),
        "check_input": False,
    }
    jobs = [
        (split, rank, int(fit_seeds[split.repeat, split.fold, rank_index]))
        for split in splits
        for rank_index, rank in enumerate(ranks)
    ]
    worker_count = min(max(1, int(n_jobs)), len(jobs))
    pin_threads = worker_count > 1
    _write_progress_event(
        out_dir,
        {
            "event": "rank_curve_started",
            "n_jobs_total": len(jobs),
            "n_worker_jobs": int(worker_count),
            "ranks": ranks,
            "n_folds": int(n_folds),
            "n_repeats": int(n_repeats),
            "sampling_fraction": float(sampling_fraction),
            "fit_kwargs": fit_kwargs,
        },
    )
    t0 = time.perf_counter()
    if worker_count == 1:
        rows = [
            _fit_rank_curve_job(
                s,
                split,
                rank,
                fit_seed,
                fit_kwargs,
                embeddings_dir,
                histories_dir,
                partial_results_dir,
                out_dir,
                pin_threads,
            )
            for split, rank, fit_seed in jobs
        ]
    else:
        rows = Parallel(n_jobs=worker_count, prefer="processes")(
            delayed(_fit_rank_curve_job)(
                s,
                split,
                rank,
                fit_seed,
                fit_kwargs,
                embeddings_dir,
                histories_dir,
                partial_results_dir,
                out_dir,
                pin_threads,
            )
            for split, rank, fit_seed in jobs
        )
    total_seconds = time.perf_counter() - t0
    _write_progress_event(
        out_dir,
        {
            "event": "rank_curve_finished",
            "n_jobs_total": len(jobs),
            "total_seconds": float(total_seconds),
        },
    )

    fold_scores = pd.DataFrame(rows).sort_values(
        ["repeat", "fold", "candidate_rank"], kind="stable"
    )
    fold_scores.to_csv(out_dir / "fold_scores.csv", index=False)

    rank_scores = _summarize(fold_scores)
    diagnostics = (
        fold_scores.groupby("candidate_rank", as_index=False)
        .agg(
            train_mse_mean=("train_mse", "mean"),
            all_offdiag_mse_mean=("all_offdiag_mse", "mean"),
            fit_seconds_total=("fit_seconds", "sum"),
            fit_seconds_mean=("fit_seconds", "mean"),
            last_evar_mean=("last_evar", "mean"),
            evar_gain_last_10_mean=("evar_gain_last_10", "mean"),
            rec_error_drop_last_10_mean=("rec_error_drop_last_10", "mean"),
            converged_fraction=("converged", "mean"),
        )
        .sort_values("candidate_rank", kind="stable")
    )
    rank_scores = rank_scores.merge(diagnostics, on="candidate_rank", how="left")
    rank_scores.to_csv(out_dir / "rank_scores.csv", index=False)

    history_frames = [pd.read_csv(path) for path in histories_dir.glob("*_history.csv")]
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            out_dir / "fit_histories.csv",
            index=False,
        )

    metadata = {
        "data_path": DATA_PATH,
        "ranks": ranks,
        "seed": int(seed),
        "n_folds": int(n_folds),
        "n_repeats": int(n_repeats),
        "sampling_fraction": float(sampling_fraction),
        "fold_pool_fraction": float(fold_fraction),
        "training_fraction": float(_training_fraction(fold_fraction, n_folds)),
        "validation_fraction": float(fold_fraction / n_folds),
        "fit_kwargs": fit_kwargs,
        "n_jobs": int(worker_count),
        "total_seconds": float(total_seconds),
    }
    _write_json(out_dir / "rank_curve_metadata.json", metadata)

    _plot_rank_curve(out_dir, fold_scores, rank_scores)
    print(f"rank curve saved: {out_dir}")
    print(rank_scores.to_string(index=False))


def _plot_rank_curve(
    out_dir: Path, fold_scores: pd.DataFrame, rank_scores: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("VGG16 Linear-Kernel SRF Rank Curve", fontsize=14)

    ax = axes[0, 0]
    n_folds = int(fold_scores["fold"].max()) + 1
    fold_id = fold_scores["repeat"] * n_folds + fold_scores["fold"]
    jitter = (fold_id - fold_id.mean()) * 2.2
    ax.scatter(
        fold_scores["candidate_rank"] + jitter,
        fold_scores["val_mse"],
        color="#8A8F98",
        s=35,
        alpha=0.65,
        linewidths=0,
        label="folds",
    )
    ax.errorbar(
        rank_scores["candidate_rank"],
        rank_scores["val_mse_mean"],
        yerr=rank_scores["val_mse_sem"],
        color="#4B5EAA",
        marker="o",
        linewidth=2.5,
        capsize=4,
        label="mean +/- SEM",
    )
    best_rank = int(
        rank_scores.loc[rank_scores["val_mse_mean"].idxmin(), "candidate_rank"]
    )
    ax.axvline(
        best_rank,
        color="#A64B2A",
        linestyle="--",
        linewidth=2,
        label=f"best={best_rank}",
    )
    ax.set_title("Validation MSE")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(
        rank_scores["candidate_rank"],
        rank_scores["train_mse_mean"],
        color="#176D6A",
        marker="o",
        linewidth=2,
        label="train",
    )
    ax.plot(
        rank_scores["candidate_rank"],
        rank_scores["val_mse_mean"],
        color="#4B5EAA",
        marker="o",
        linewidth=2,
        label="validation",
    )
    ax.set_title("Train/validation gap")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(
        rank_scores["candidate_rank"],
        rank_scores["last_evar_mean"],
        color="#176D6A",
        marker="o",
        linewidth=2,
    )
    ax.set_title("Mean final explained variance")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("final evar")

    ax = axes[1, 1]
    ax.bar(
        rank_scores["candidate_rank"].astype(str),
        rank_scores["evar_gain_last_10_mean"],
        color="#C59A2D",
        alpha=0.85,
    )
    ax.set_title("Mean evar gain over final 10 iterations")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("evar gain")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "rank_curve.png", dpi=160)
    plt.close(fig)

    history_path = out_dir / "fit_histories.csv"
    if history_path.exists():
        histories = pd.read_csv(history_path)
        summary = (
            histories.groupby(["candidate_rank", "outer_iteration"], as_index=False)
            .agg(evar_mean=("evar", "mean"), rec_error_mean=("rec_error", "mean"))
            .sort_values(["candidate_rank", "outer_iteration"])
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for rank, group in summary.groupby("candidate_rank"):
            axes[0].plot(
                group["outer_iteration"], group["evar_mean"], label=f"rank {rank}"
            )
            axes[1].plot(
                group["outer_iteration"], group["rec_error_mean"], label=f"rank {rank}"
            )
        axes[0].set_title("Mean CV fit evar by iteration")
        axes[0].set_xlabel("outer iteration")
        axes[0].set_ylabel("evar")
        axes[1].set_title("Mean CV fit reconstruction error")
        axes[1].set_xlabel("outer iteration")
        axes[1].set_ylabel("rec error")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.savefig(out_dir / "fit_convergence.png", dpi=160)
        plt.close(fig)


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / x.sum() - (n + 1)) / n)


def _normalized_columns(w: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(w, axis=0)
    return w / np.maximum(norms, 1e-12)


def _embedding_info(path: Path) -> tuple[int, int, int]:
    parts = path.stem.split("_")
    return int(parts[1]), int(parts[3]), int(parts[5])


def run_item_specificity() -> None:
    out_dir = OUT_ROOT / "psd_raw_item_specificity"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_progress_event(out_dir, {"event": "item_specificity_started"})
    embeddings_dir = OUT_ROOT / "linear_kernel_rank_curve" / "embeddings"
    if not embeddings_dir.exists():
        raise FileNotFoundError("Run the rank curve first, or use --section all.")

    s = _load_similarity()
    n = s.shape[0]
    triu = np.triu_indices(n, k=1)
    factor_rows = []
    model_rows = []
    embeddings_by_rank: dict[int, list[tuple[int, int, np.ndarray]]] = {}

    for path in sorted(embeddings_dir.glob("rank_*_repeat_*_fold_*.npy")):
        rank, repeat, fold = _embedding_info(path)
        w = np.load(path)
        embeddings_by_rank.setdefault(rank, []).append((repeat, fold, w))

        s_hat = w @ w.T
        diag_residual = np.diag(s) - np.diag(s_hat)
        offdiag_residual = s[triu] - s_hat[triu]
        model_rows.append(
            {
                "candidate_rank": rank,
                "repeat": repeat,
                "fold": fold,
                "diag_mse": float(np.mean(diag_residual * diag_residual)),
                "offdiag_mse": float(np.mean(offdiag_residual * offdiag_residual)),
                "diag_corr": float(np.corrcoef(np.diag(s), np.diag(s_hat))[0, 1]),
                "diag_mean_hat": float(np.diag(s_hat).mean()),
                "offdiag_mean_hat": float(s_hat[triu].mean()),
            }
        )

        for factor in range(w.shape[1]):
            column = w[:, factor]
            l1 = float(column.sum())
            l2_sq = float(np.dot(column, column))
            if l2_sq <= 0:
                effective_support = 0.0
                top1_l2_share = 0.0
                top5_l2_share = 0.0
                top10_l2_share = 0.0
            else:
                effective_support = l1 * l1 / l2_sq
                weights = np.sort(column * column)[::-1]
                top1_l2_share = float(weights[:1].sum() / l2_sq)
                top5_l2_share = float(weights[:5].sum() / l2_sq)
                top10_l2_share = float(weights[:10].sum() / l2_sq)
            factor_rows.append(
                {
                    "candidate_rank": rank,
                    "repeat": repeat,
                    "fold": fold,
                    "factor": factor,
                    "l1_sum": l1,
                    "l2_norm": float(np.sqrt(l2_sq)),
                    "max_loading": float(column.max()),
                    "max_loading_share_l1": float(column.max() / max(l1, 1e-12)),
                    "effective_support": float(effective_support),
                    "normalized_effective_support": float(effective_support / n),
                    "top1_l2_share": top1_l2_share,
                    "top5_l2_share": top5_l2_share,
                    "top10_l2_share": top10_l2_share,
                    "gini": _gini(column),
                }
            )

    factor_metrics = pd.DataFrame(factor_rows)
    model_metrics = pd.DataFrame(model_rows)
    factor_metrics.to_csv(out_dir / "factor_concentration_metrics.csv", index=False)
    model_metrics.to_csv(out_dir / "model_reconstruction_metrics.csv", index=False)

    stability_rows = []
    matched_rows = []
    for rank, entries in sorted(embeddings_by_rank.items()):
        for left_index in range(len(entries)):
            repeat_a, fold_a, w_a = entries[left_index]
            a = _normalized_columns(w_a)
            for right_index in range(left_index + 1, len(entries)):
                repeat_b, fold_b, w_b = entries[right_index]
                b = _normalized_columns(w_b)
                sim = a.T @ b
                row_ind, col_ind = linear_sum_assignment(-sim)
                matched = sim[row_ind, col_ind]
                pair_label = f"r{repeat_a}f{fold_a}_vs_r{repeat_b}f{fold_b}"
                stability_rows.append(
                    {
                        "candidate_rank": rank,
                        "pair": pair_label,
                        "mean_matched_cosine": float(np.mean(matched)),
                        "median_matched_cosine": float(np.median(matched)),
                        "q10_matched_cosine": float(np.quantile(matched, 0.10)),
                        "q90_matched_cosine": float(np.quantile(matched, 0.90)),
                    }
                )
                for value in matched:
                    matched_rows.append(
                        {
                            "candidate_rank": rank,
                            "pair": pair_label,
                            "matched_cosine": float(value),
                        }
                    )

    stability = pd.DataFrame(stability_rows)
    matched = pd.DataFrame(matched_rows)
    stability.to_csv(out_dir / "fold_factor_stability_summary.csv", index=False)
    matched.to_csv(out_dir / "fold_factor_stability_matched_cosines.csv", index=False)

    factor_summary = (
        factor_metrics.groupby("candidate_rank", as_index=False)
        .agg(
            median_effective_support=("effective_support", "median"),
            q10_effective_support=("effective_support", lambda x: x.quantile(0.10)),
            singleton_like_fraction_10_items=(
                "effective_support",
                lambda x: float(np.mean(x <= 10)),
            ),
            median_top10_l2_share=("top10_l2_share", "median"),
            median_gini=("gini", "median"),
        )
        .sort_values("candidate_rank")
    )
    stability_summary = (
        stability.groupby("candidate_rank", as_index=False)
        .agg(
            mean_matched_cosine=("mean_matched_cosine", "mean"),
            median_matched_cosine=("median_matched_cosine", "mean"),
            q10_matched_cosine=("q10_matched_cosine", "mean"),
        )
        .sort_values("candidate_rank")
        if not stability.empty
        else pd.DataFrame()
    )
    factor_summary.to_csv(out_dir / "factor_concentration_summary.csv", index=False)
    if not stability_summary.empty:
        stability_summary.to_csv(out_dir / "factor_stability_by_rank.csv", index=False)

    _plot_item_specificity(out_dir, factor_metrics, model_metrics, matched)
    _write_progress_event(out_dir, {"event": "item_specificity_finished"})
    print(f"item-specificity diagnostics saved: {out_dir}")


def _plot_item_specificity(
    out_dir: Path,
    factor_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    matched: pd.DataFrame,
) -> None:
    ranks = sorted(factor_metrics["candidate_rank"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Raw PSD VGG16 Item-Specificity Diagnostics", fontsize=14)

    ax = axes[0, 0]
    data = [
        factor_metrics.loc[
            factor_metrics["candidate_rank"] == rank,
            "normalized_effective_support",
        ]
        for rank in ranks
    ]
    ax.boxplot(data, labels=[str(rank) for rank in ranks], showfliers=False)
    ax.set_title("Factor effective support")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("effective support / n items")

    ax = axes[0, 1]
    data = [
        factor_metrics.loc[factor_metrics["candidate_rank"] == rank, "top10_l2_share"]
        for rank in ranks
    ]
    ax.boxplot(data, labels=[str(rank) for rank in ranks], showfliers=False)
    ax.set_title("Top-10 item concentration")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("top-10 share of factor L2 mass")

    ax = axes[1, 0]
    if not matched.empty:
        data = [
            matched.loc[matched["candidate_rank"] == rank, "matched_cosine"]
            for rank in ranks
        ]
        ax.boxplot(data, labels=[str(rank) for rank in ranks], showfliers=False)
    ax.set_title("Fold-to-fold matched factor stability")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("matched cosine")

    ax = axes[1, 1]
    model_summary = (
        model_metrics.groupby("candidate_rank", as_index=False)
        .agg(diag_corr=("diag_corr", "mean"), offdiag_mse=("offdiag_mse", "mean"))
        .sort_values("candidate_rank")
    )
    ax.plot(
        model_summary["candidate_rank"],
        model_summary["diag_corr"],
        marker="o",
        color="#176D6A",
        linewidth=2,
        label="diag corr",
    )
    ax2 = ax.twinx()
    ax2.plot(
        model_summary["candidate_rank"],
        model_summary["offdiag_mse"],
        marker="o",
        color="#4B5EAA",
        linewidth=2,
        label="offdiag MSE",
    )
    ax.set_title("Reconstruction diagnostics")
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("diag corr")
    ax2.set_ylabel("offdiag MSE")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "item_specificity_diagnostics.png", dpi=160)
    plt.close(fig)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=("overview", "calibration", "rank_curve", "item_specificity", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=_parse_ints, default=DEFAULT_RANKS)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--rank-curve-jobs", type=int, default=5)
    parser.add_argument("--calibration-jobs", type=int, default=-1)
    parser.add_argument("--max-outer", type=int, default=100)
    parser.add_argument("--max-inner", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--primary-bootstrap", type=int, default=50)
    parser.add_argument("--estimate-bootstrap", type=int, default=10)
    parser.add_argument("--estimate-seeds", type=_parse_ints, default=(0, 1, 2, 3, 4))
    parser.add_argument("--max-eigenpairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.section in {"overview", "all"}:
        run_matrix_overview()
    if args.section in {"calibration", "all"}:
        run_sampling_calibration(
            seed=args.seed,
            primary_bootstrap=args.primary_bootstrap,
            estimate_bootstrap=args.estimate_bootstrap,
            estimate_seeds=args.estimate_seeds,
            n_jobs=args.calibration_jobs,
            max_eigenpairs=args.max_eigenpairs,
        )
    if args.section in {"rank_curve", "all"}:
        run_linear_rank_curve(
            ranks=args.ranks,
            seed=args.seed,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            n_jobs=args.rank_curve_jobs,
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            tol=args.tol,
        )
    if args.section in {"item_specificity", "all"}:
        run_item_specificity()


if __name__ == "__main__":
    main()
