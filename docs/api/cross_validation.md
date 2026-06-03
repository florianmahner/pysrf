# Cross-Validation API

`cross_val_score` confirms the rank suggested by `estimate_rank` by holding out individual entries of the similarity matrix and measuring how well the model predicts them. Naive row- or fold-based CV leaks information on a similarity matrix, because an item kept in training also appears (transposed) in the held-out rows, so PySRF uses a restricted entry-wise hold-out instead. It returns a tidy `pandas` DataFrame with columns `[rep, fold, rank, val_mse]`; read it with a `groupby("rank")["val_mse"].mean().idxmin()` to pick the best rank. See [quickstart](../quickstart.md) for a full example.

::: pysrf.cross_val_score
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
