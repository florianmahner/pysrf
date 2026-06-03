# Rank Estimation API

`estimate_rank` is the first step in a typical PySRF workflow: it inspects a similarity matrix and estimates how many signal dimensions it contains, before you fit an `SRF`. It works by bootstrapping the eigenspace of `S` and measuring how stable the top dimensions stay under resampling, returning a `RankEstimate` whose `.rank` is the recommended number of dimensions and whose `.sampling_fraction` feeds straight into `cross_val_score`. See the "Choose the number of dimensions" section of [quickstart](../quickstart.md) for how to use it in practice.

::: pysrf.estimate_rank
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: pysrf.RankEstimate
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
