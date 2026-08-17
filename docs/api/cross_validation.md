# Cross-Validation API

`cross_val_score` is the high-level dimensionality-selection API. By default, it first calibrates a spectral cutoff and a sampling fraction from the similarity matrix, then evaluates the candidate ranks you provide with restricted entrywise cross-validation. The returned `CVResult` exposes the selected model rank as `cv.model_rank`, the fold-level validation scores as `cv.fold_scores`, and the per-rank averages as `cv.rank_scores`. Calibration itself is documented in [Calibration](calibration.md).

Entries of a similarity matrix are not independent observations: once observed entries exceed the matrix-completion threshold, held-out entries are already determined by the training entries, which biases rank selection toward overly large ranks. `cross_val_score` therefore fixes a sparse observation pattern before cross-validation, keeping each fold's training set below that threshold. Folds are assigned at the level of unordered off-diagonal pairs, mirrored symmetrically, and treated as missing during fitting.

::: pysrf.cross_val_score
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: pysrf.CVResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
