# Cross-Validation API

`cross_val_score` is the high-level dimensionality-selection API. By default, it first calibrates a spectral cutoff and sampling fraction from eigenspace stability, then evaluates SRF model ranks around that cutoff with restricted entry-wise cross-validation. The returned `CVResult` exposes the selected model rank as `cv.model_rank`, the fold-level validation scores as `cv.fold_scores`, and the per-rank averages as `cv.rank_scores`.

Naive row- or fold-based CV leaks information on a similarity matrix, because an item kept in training also appears in many held-out pairs. PySRF therefore holds out symmetric off-diagonal entries and treats them as missing during fitting.

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
