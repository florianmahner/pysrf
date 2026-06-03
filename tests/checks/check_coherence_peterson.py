from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysrf import cross_val_score
from pysrf.coherence import calibrate_cross_validation

PETERSON_ROOT = Path("/SSD/datasets/similarity_datasets/peterson/various")


def main():
    rsm = np.ascontiguousarray(np.load(PETERSON_ROOT / "rsm.npy"), dtype=np.float64)
    n = rsm.shape[0]
    print(f"Peterson various: n={n}, range=[{rsm.min():.2f}, {rsm.max():.2f}]")

    print("\n--- Calibrated cross-validation ---")
    ranks_to_test = list(range(2, 36, 2))
    calibration = calibrate_cross_validation(
        rsm,
        max_eigenpairs=40,
        sampling_grid=np.linspace(0.1, 0.95, 20),
        n_bootstrap=30,
        random_state=42,
        n_jobs=-1,
    )
    cv = cross_val_score(
        rsm,
        ranks=ranks_to_test,
        sampling_fraction=calibration.sampling_fraction,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        srf_kwargs={"max_outer": 50},
    )

    print(f"spectral cutoff = {cv.spectral_cutoff}")
    print(f"sampling fraction = {cv.sampling_fraction:.3f}")
    print(f"model rank = {cv.model_rank}")
    print(f"CV scores:\n{cv.rank_scores.to_string(index=False)}")

    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Peterson 'various' (120 objects, fMRI)", fontsize=13)

    ax = axes[0]
    dims = np.arange(1, calibration.leakage.size + 1)
    ax.plot(dims, calibration.leakage, "o-", markersize=3)
    ax.axvline(
        calibration.spectral_cutoff,
        color="r",
        linestyle="--",
        label=f"spectral cutoff={calibration.spectral_cutoff}",
    )
    ax.set_xlabel("spectral dimension")
    ax.set_ylabel("leakage")
    ax.set_title("Eigenspace stability cutoff")
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
    ax.axvline(
        calibration.spectral_cutoff,
        color="r",
        linestyle="--",
        label=f"spectral cutoff={calibration.spectral_cutoff}",
    )
    ax.axvline(
        cv.model_rank,
        color="b",
        linestyle=":",
        label=f"model rank={cv.model_rank}",
    )
    ax.set_xlabel("SRF rank")
    ax.set_ylabel("validation MSE")
    ax.set_title("SRF model-rank CV")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(plot_dir / "check_coherence_peterson.png", dpi=150)
    print(f"\nPlot saved to {plot_dir / 'check_coherence_peterson.png'}")
    plt.close()


if __name__ == "__main__":
    main()
