from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysrf import cross_val_score
from pysrf.coherence import calibrate_cross_validation


def _make_low_rank_similarity(n: int, rank: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, rank)))
    s = w @ w.T
    e = rng.standard_normal((n, n))
    return s + noise * (e + e.T) / 2


def _save_plot(cv, calibration, true_rank: int) -> Path:
    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / "check_coherence.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Synthetic low-rank similarity matrix", fontsize=13)

    ax = axes[0]
    dims = np.arange(1, calibration.leakage.size + 1)
    ax.plot(dims, calibration.leakage, "o-", markersize=3)
    ax.axvline(true_rank, color="k", linestyle=":", label=f"true rank={true_rank}")
    ax.axvline(
        calibration.spectral_cutoff,
        color="r",
        linestyle="--",
        label=f"spectral cutoff={calibration.spectral_cutoff}",
    )
    ax.set_xlabel("spectral dimension")
    ax.set_ylabel("leakage")
    ax.set_title("Eigenspace stability")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        calibration.sampling_grid,
        calibration.signal_loss_raw,
        "o-",
        markersize=3,
        label="raw",
    )
    ax.plot(
        calibration.sampling_grid,
        calibration.signal_loss_monotone,
        "-",
        linewidth=2,
        label="monotone",
    )
    ax.axvline(
        cv.sampling_fraction,
        color="k",
        linestyle="--",
        label=f"p={cv.sampling_fraction:.2f}",
    )
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("signal loss")
    ax.set_title("Sampling calibration")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.errorbar(
        cv.rank_scores["candidate_rank"],
        cv.rank_scores["val_mse_mean"],
        yerr=cv.rank_scores["val_mse_sem"],
        fmt="o-",
        capsize=3,
    )
    ax.axvline(true_rank, color="k", linestyle=":", label=f"true rank={true_rank}")
    ax.axvline(
        cv.model_rank,
        color="b",
        linestyle="--",
        label=f"model rank={cv.model_rank}",
    )
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("validation MSE")
    ax.set_title("Model-rank CV")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main():
    seed = 42
    true_rank = 5
    s = _make_low_rank_similarity(n=70, rank=true_rank, noise=0.02, seed=seed)

    calibration = calibrate_cross_validation(
        s,
        max_eigenpairs=10,
        sampling_grid=np.linspace(0.15, 0.95, 8),
        n_bootstrap=5,
        random_state=seed,
        n_jobs=-1,
    )
    cv = cross_val_score(
        s,
        ranks=range(
            max(1, calibration.spectral_cutoff - 2),
            calibration.spectral_cutoff + 3,
        ),
        sampling_fraction=calibration.sampling_fraction,
        n_folds=3,
        random_state=seed,
        n_jobs=-1,
        srf_kwargs={"max_outer": 4, "max_inner": 4, "check_input": False},
    )
    if abs(calibration.spectral_cutoff - true_rank) > 2:
        raise AssertionError("spectral cutoff is far from the planted rank")
    if abs(cv.model_rank - true_rank) > 2:
        raise AssertionError("CV-selected model rank is far from the planted rank")
    plot_path = _save_plot(cv, calibration, true_rank)

    print(f"true rank:      {true_rank}")
    print(f"spectral cutoff: {cv.spectral_cutoff}")
    print(f"sampling frac:  {cv.sampling_fraction:.3f}")
    print(f"model rank:     {cv.model_rank}")
    print(cv.rank_scores.to_string(index=False))
    print(f"plot saved:     {plot_path}")


if __name__ == "__main__":
    main()
