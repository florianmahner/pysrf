# Calibration API

This lower-level API lives in `pysrf.coherence` and calibrates the cross-validation protocol from a similarity matrix alone. It returns two quantities: a spectral cutoff, the number of leading eigendirections that remain stable under random off-diagonal subsampling, and a sampling fraction, the smallest sampling probability that preserves most of that spectral structure. Both are model independent and determined entirely by the similarity matrix. Most users should call `cross_val_score`, which runs this calibration by default.

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
