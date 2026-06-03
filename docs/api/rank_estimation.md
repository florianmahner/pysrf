# Cross-Validation Calibration API

This lower-level API calibrates the cross-validation protocol. It returns a model-independent `spectral_cutoff` and a `sampling_fraction`; the final SRF model rank is still selected by `cross_val_score`.

Most users should call `cross_val_score` directly.

::: pysrf.coherence.calibrate_cross_validation
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: pysrf.coherence.CVCalibration
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
