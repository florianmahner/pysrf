# VGG16 Similarity Diagnostics

Diagnostics for SRF rank selection on a raw VGG16 linear-kernel similarity matrix (`tests/vgg16_similarity.npy`, 1854 x 1854, PSD, no cosine normalization). They probe why calibrated entrywise cross-validation selects far higher ranks than the spectral cutoff, and which parts of protocol, optimization, and data drive that gap. Findings and result tables live in [SUMMARY.md](SUMMARY.md); this README only maps the folder.

## Running

`run_diagnostics.py` runs the core pipeline. Sections build on each other: `rank_curve` reads the calibrated sampling fraction from `sampling_fraction_calibration/`, and `item_specificity` reads saved embeddings from `linear_kernel_rank_curve/`, so run `calibration` before `rank_curve` and `rank_curve` before `item_specificity`, or use `--section all`.

```bash
poetry run python tests/checks/vgg16_similarity/run_diagnostics.py --section all
poetry run python tests/checks/vgg16_similarity/run_diagnostics.py --section rank_curve --ranks 50,100,200 --n-folds 5 --max-inner 5
```

`--section` is one of `overview`, `calibration`, `rank_curve`, `item_specificity`, `all`. Remaining flags set seeds (`--seed`, `--estimate-seeds`), CV shape (`--ranks`, `--n-folds`, `--n-repeats`), parallelism (`--rank-curve-jobs`, `--calibration-jobs`), ADMM budgets (`--max-outer`, `--max-inner`, `--tol`), and calibration effort (`--primary-bootstrap`, `--estimate-bootstrap`, `--max-eigenpairs`). `plot_outer_convergence_overview.py` rebuilds `outer_convergence_overview/` from saved fit histories. Follow-up experiments have their own runners under `mechanism_probes/`.

## Intermediate progress

Long-running diagnostics save intermediate progress to disk so an active run stays inspectable. Runners write timestamped start/finish/failure JSON events under `progress_events/` and completed rank/fold outputs under `partial_results/` before building final CSV summaries and plots.

## Experiment folders

Core pipeline, written by `run_diagnostics.py`:

- [matrix_overview](matrix_overview/): spectrum, trace mass, and diagonal versus off-diagonal scale of the raw similarity matrix.
- [sampling_fraction_calibration](sampling_fraction_calibration/): spectral cutoff and CV sampling-fraction calibration, with repeat-seed variability.
- [linear_kernel_rank_curve](linear_kernel_rank_curve/): main calibrated entrywise CV rank curve, with saved embeddings and fit histories.
- [psd_raw_item_specificity](psd_raw_item_specificity/): factor concentration and fold-to-fold stability of saved rank-curve embeddings.

Controls and follow-ups:

- [linear_kernel_rank_curve_inner20_outer100](linear_kernel_rank_curve_inner20_outer100/): optimization control with a larger W-subproblem budget, testing whether inner-iteration count explains the high-rank preference.
- [linear_kernel_rank_curve_diagonal_hidden_inner5_outer100](linear_kernel_rank_curve_diagonal_hidden_inner5_outer100/): diagonal-hidden control where diagonal entries are hidden during each CV fit, probing diagonal/self-norm confounding of the off-diagonal objective.
- [psd_raw_item_specificity_diagonal_hidden_inner5_outer100](psd_raw_item_specificity_diagonal_hidden_inner5_outer100/): item-specificity diagnostics recomputed on diagonal-hidden fits.
- [degree_thinned_pairwise_cv_diagonal_hidden_inner5_outer100](degree_thinned_pairwise_cv_diagonal_hidden_inner5_outer100/): degree-thinned pairwise CV with capped per-item training degree, testing whether dense transductive item context is necessary for high ranks to help.
- [entrywise_split_item_leakage](entrywise_split_item_leakage/): split coverage per item and item/norm-only baselines, probing how much item-level leakage the entrywise split permits.
- [spectral_cutoff_vs_validation_objective](spectral_cutoff_vs_validation_objective/): oracle PSD rank-k approximations scored under the CV objective, probing the mismatch between spectral cutoff and entrywise validation MSE.
- [item_holdout_transductive_limitation](item_holdout_transductive_limitation/): note on why item-held-out CV is not a fair rank criterion for this transductive model; see its README.
- [strict_convergence](strict_convergence/): convergence-focused runs kept in a stable layout; see its README.
- [outer_convergence_overview](outer_convergence_overview/): outer-iteration objective and evar tails compared across the rank-curve experiments.
- [mechanism_probes](mechanism_probes/): targeted probes for explaining the rank pressure, including ADMM data-weight and bounds checks, complete-data factorization controls, error decomposition, stratum rank selection, initialization comparisons, and degree-thinning variants; see its README and SUMMARY.
- [cross_experiment_comparison](cross_experiment_comparison/): rank curves and factor-specificity metrics combined across experiments.
