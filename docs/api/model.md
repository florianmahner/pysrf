# Model API

`SRF` is the main estimator in PySRF. Given a symmetric similarity matrix `S`, it learns a sparse, non-negative embedding `W` such that `S ≈ WW^T`, where each row of `W` holds an item's loadings on the recovered dimensions and small loadings mark dimensions that are irrelevant to that item. It works directly on matrices with missing entries, so no imputation is needed. Reach for `SRF` whenever you want interpretable, part-based dimensions out of a similarity matrix; see [quickstart](../quickstart.md) for an end-to-end walkthrough.

## SRF Class

::: pysrf.SRF
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

