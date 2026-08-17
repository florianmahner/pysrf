# Calibration API

This lower-level API lives in `pysrf.coherence`. It returns a model-independent spectral cutoff plus a sampling fraction. Most users should call `cross_val_score`, which runs this calibration by default.

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
