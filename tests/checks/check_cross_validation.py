from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import make_cv_diagnostic_matrix
from pysrf import CVResult, SRF, cross_val_score
from pysrf.coherence import calibrate_cross_validation


def _fit_full_models(
    s: np.ndarray,
    candidate_ranks: tuple[int, ...],
    seed: int,
) -> dict[int, SRF]:
    return {
        candidate_rank: SRF(
            rank=candidate_rank,
            max_outer=20,
            max_inner=10,
            random_state=seed,
            check_input=False,
            tol=1e-5,
        ).fit(s)
        for candidate_rank in candidate_ranks
    }


def _save_plot(
    s: np.ndarray,
    result: CVResult,
    calibration,
    true_rank: int,
    signal_loss_tolerance: float,
    full_models: dict[int, SRF],
    path: Path,
) -> None:
    colors = {
        "signal": "#176D6A",
        "accent": "#A64B2A",
        "model": "#4B5EAA",
        "neutral": "#3F3F46",
        "muted": "#8A8F98",
        "gold": "#C59A2D",
    }
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), constrained_layout=True)
    fig.patch.set_facecolor("#FAFAF8")
    fig.suptitle("Calibrated SRF Cross-Validation Diagnostics", fontsize=16)
    for label, ax in zip("ABCDEFGH", axes.ravel()):
        _style_axis(ax)
        ax.text(
            -0.12,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    ax = axes[0, 0]
    image = ax.imshow(s, cmap="viridis", interpolation="nearest")
    ax.set_title("Input Similarity Matrix")
    ax.set_xlabel("Object index")
    ax.set_ylabel("Object index")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Similarity")

    ax = axes[0, 1]
    dims = np.arange(1, calibration.eigvals.size + 1)
    ax.plot(
        dims,
        calibration.eigvals,
        marker="o",
        color=colors["signal"],
        linewidth=2,
        markersize=4,
    )
    ax.axvline(true_rank, color=colors["neutral"], linestyle=":", label="true rank")
    ax.axvline(
        calibration.spectral_cutoff,
        color=colors["accent"],
        linestyle="--",
        label="spectral cutoff",
    )
    ax.set_title("Spectrum")
    ax.set_xlabel("Spectral dimension")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.set_xlim(0.5, calibration.eigvals.size + 0.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 2]
    ax.plot(
        dims,
        calibration.leakage,
        marker="o",
        color=colors["signal"],
        linewidth=2,
        markersize=4,
    )
    ax.axvline(true_rank, color=colors["neutral"], linestyle=":", label="true rank")
    ax.axvline(
        calibration.spectral_cutoff,
        color=colors["accent"],
        linestyle="--",
        label="spectral cutoff",
    )
    ax.set_title("Eigenspace Stability")
    ax.set_xlabel("Spectral dimension")
    ax.set_ylabel("Leakage score")
    ax.set_xlim(0.5, calibration.leakage.size + 0.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 3]
    ax.plot(
        calibration.sampling_grid,
        calibration.signal_loss_raw,
        marker="o",
        color=colors["muted"],
        linewidth=1.5,
        markersize=4,
        label="raw signal loss",
    )
    ax.plot(
        calibration.sampling_grid,
        calibration.signal_loss_monotone,
        color=colors["signal"],
        linewidth=2,
        label="monotone loss",
    )
    ax.axhline(
        signal_loss_tolerance,
        color=colors["gold"],
        linestyle=":",
        label=f"tolerance={signal_loss_tolerance:.2f}",
    )
    ax.axvline(
        calibration.detectability_floor,
        color=colors["muted"],
        linestyle="--",
        label="detectability floor",
    )
    ax.axvline(
        result.sampling_fraction,
        color=colors["accent"],
        linestyle="--",
        label=f"selected p={result.sampling_fraction:.2f}",
    )
    ax.set_title("Sampling Calibration")
    ax.set_xlabel("Sampling fraction")
    ax.set_ylabel("Signal loss")
    ax.set_xlim(0.0, 0.95)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    n_folds = result.fold_scores["fold"].max() + 1
    fold_id = result.fold_scores["repeat"] * n_folds + result.fold_scores["fold"]
    fold_id = fold_id.to_numpy(dtype=float)
    fold_id -= fold_id.mean()
    jitter = 0.045 * fold_id
    ax.scatter(
        result.fold_scores["candidate_rank"].to_numpy() + jitter,
        result.fold_scores["val_mse"],
        color=colors["muted"],
        alpha=0.6,
        s=22,
        linewidths=0,
        label="fold scores",
    )
    ax.plot(
        result.rank_scores["candidate_rank"],
        result.rank_scores["val_mse_mean"],
        color=colors["model"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="mean",
    )
    ax.axvline(true_rank, color=colors["neutral"], linestyle=":", label="true rank")
    ax.axvline(
        result.model_rank,
        color=colors["accent"],
        linestyle="--",
        label="selected rank",
    )
    ax.set_title("Fold-Level CV Scores")
    ax.set_xlabel("Candidate SRF rank")
    ax.set_ylabel("Validation MSE")
    ax.set_xlim(0.5, max(result.candidate_ranks) + 0.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    ax.errorbar(
        result.rank_scores["candidate_rank"],
        result.rank_scores["val_mse_mean"],
        yerr=result.rank_scores["val_mse_sem"],
        color=colors["model"],
        marker="o",
        linewidth=2.5,
        capsize=3,
        label="mean +/- SEM",
    )
    ax.axvline(true_rank, color=colors["neutral"], linestyle=":", label="true rank")
    ax.axvline(
        result.model_rank,
        color=colors["accent"],
        linestyle="--",
        label="selected rank",
    )
    ax.set_title("Model-Rank Selection")
    ax.set_xlabel("Candidate SRF rank")
    ax.set_ylabel("Mean validation MSE")
    ax.set_xlim(0.5, max(result.candidate_ranks) + 0.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    convergence_colors = {
        2: colors["muted"],
        true_rank: colors["model"],
        8: colors["gold"],
    }
    for candidate_rank, model in full_models.items():
        linewidth = 2.8 if candidate_rank == result.model_rank else 1.7
        alpha = 1.0 if candidate_rank == result.model_rank else 0.65
        label = f"rank {candidate_rank}"
        if candidate_rank == result.model_rank:
            label += " (selected)"
        ax.plot(
            np.arange(1, len(model.history_["rec_error"]) + 1),
            model.history_["rec_error"],
            color=convergence_colors.get(candidate_rank, colors["neutral"]),
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )
    ax.set_title("Full-Fit Convergence")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Reconstruction error")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 3]
    selected_model = full_models[result.model_rank]
    reconstruction = selected_model.reconstruct()
    i, j = np.triu_indices_from(s, k=1)
    ax.scatter(
        s[i, j],
        reconstruction[i, j],
        color=colors["signal"],
        alpha=0.35,
        s=12,
        linewidths=0,
    )
    limits = [
        min(float(s[i, j].min()), float(reconstruction[i, j].min())),
        max(float(s[i, j].max()), float(reconstruction[i, j].max())),
    ]
    ax.plot(limits, limits, color=colors["neutral"], linestyle=":", linewidth=1.5)
    ax.set_title("Selected-Rank Reconstruction")
    ax.set_xlabel("Observed similarity")
    ax.set_ylabel("Reconstructed similarity")

    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color="#D6D9DE", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")


def main() -> None:
    true_rank = 4
    signal_loss_tolerance = 0.10
    seed = 8
    s = make_cv_diagnostic_matrix(
        n=70,
        rank=true_rank,
        nuisance_rank=8,
        nuisance_scale=1.05,
        noise=0.05,
        seed=seed,
    )
    calibration = calibrate_cross_validation(
        s,
        max_eigenpairs=12,
        sampling_grid=np.linspace(0.15, 0.9, 8),
        signal_loss_tolerance=signal_loss_tolerance,
        n_bootstrap=5,
        random_state=seed,
        n_jobs=-1,
    )
    result = cross_val_score(
        s,
        ranks=range(1, 13),
        sampling_fraction=calibration.sampling_fraction,
        n_folds=5,
        n_repeats=2,
        random_state=seed,
        n_jobs=-1,
        srf_kwargs={
            "max_outer": 20,
            "max_inner": 10,
            "check_input": False,
            "tol": 1e-5,
        },
    )

    full_ranks = tuple(dict.fromkeys((2, true_rank, 8, result.model_rank)))
    full_models = _fit_full_models(s, candidate_ranks=full_ranks, seed=seed)
    plot_path = Path(__file__).parent / "plots" / "cross_validation_end_to_end.png"
    _save_plot(
        s,
        result,
        calibration,
        true_rank,
        signal_loss_tolerance,
        full_models,
        plot_path,
    )

    print(f"true rank:       {true_rank}")
    print(f"spectral cutoff: {calibration.spectral_cutoff}")
    print(f"sampling frac:   {result.sampling_fraction:.3f}")
    print(f"model rank:      {result.model_rank}")
    print(result.rank_scores.to_string(index=False))
    print(f"plot saved:      {plot_path}")


if __name__ == "__main__":
    main()
