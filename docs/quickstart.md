# Quick start

This guide walks through the core PySRF workflow: revealing
interpretable dimensions from a similarity matrix, interpreting the result,
learning from incomplete data, and choosing how many dimensions to keep.

## Reveal dimensions from a similarity matrix

A similarity matrix `S` captures pairwise relationships between stimuli.
These similarities can come from diverse forms of measured data across
systems, modalities, and species — for example behavioral similarity
judgments, neural recordings, or computational models. You can also think
of `S` as a weighted graph, where stronger edges connect more similar
items. PySRF expects this matrix to be symmetric and non-negative.

SRF factorizes the matrix into a single non-negative embedding `W` so that
`S ≈ WW^T`. Each row of `W` holds one item's loadings across the
dimensions, and each column is a dimension. Because loadings are
non-negative, the dimensions act as soft community memberships: an item
can belong to several dimensions at once, and a near-zero loading means
that dimension is irrelevant to that item.

```python
import numpy as np
from pysrf import SRF

# Build a small symmetric, non-negative similarity matrix
rng = np.random.default_rng(0)
s = rng.random((100, 100))
s = (s + s.T) / 2  # make it symmetric

# Reveal 10 dimensions
model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)

# Reconstruct the similarity matrix from the embedding
s_hat = model.reconstruct()

# Measure reconstruction error (lower is better)
error = model.score(s)
print(f"Reconstruction error: {error:.4f}")
```

`w` has shape `(100, 10)`: one row per item, one column per dimension.
Items that load strongly on the same dimension are similar for the same
reason, which is what makes the result directly interpretable.

## Handle missing data

Most behavioral similarity matrices are incomplete. The number of pairwise
comparisons grows quadratically with the number of items, so measuring
every pair quickly becomes infeasible and many entries are simply never
observed.

PySRF handles this without imputing anything. Mark unobserved entries as
`NaN`, and SRF learns only from the observed pairs. After fitting,
`reconstruct()` predicts a value for every pair, including the ones that
were missing.

```python
import numpy as np
from pysrf import SRF

rng = np.random.default_rng(0)
s = rng.random((100, 100))
s = (s + s.T) / 2

# Hide 30% of the entries
mask = rng.random((100, 100)) < 0.3
s[mask] = np.nan

# SRF fits using only the observed pairs
model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)

# The reconstruction fills in the missing pairs
s_completed = model.reconstruct()
```

`s_completed` contains the model's predicted similarities for both the
observed and the previously missing entries.

## Choose the number of dimensions

The number of dimensions (the rank) controls what SRF can capture. Too few
dimensions merge distinct factors or miss structure; too many split
coherent factors or overfit noise. Use `cross_val_score` to select the
model rank from held-out prediction error.

Ordinary entrywise cross-validation does not work here: entries in a
similarity matrix are not independent, because changing one item affects
all of its pairs. PySRF therefore uses a restricted hold-out scheme for
similarity matrices. It hides a sparse set of observed entries, treats them
as missing during fitting, and then predicts them from the learned
embedding. By default, `cross_val_score` calibrates the sampling fraction,
then evaluates the candidate ranks you provide.

```python
from pysrf import SRF, cross_val_score

# Select the model rank by calibrated cross-validation
cv = cross_val_score(s, range(1, 21), random_state=42)

print(f"Spectral cutoff: {cv.spectral_cutoff}")
print(f"Selected model rank: {cv.model_rank}")

# Refit at the chosen rank on the full matrix
model = SRF(rank=cv.model_rank, random_state=42)
w = model.fit_transform(s)
```

`cross_val_score` returns a `CVResult`. The final model rank is in
`cv.model_rank`, the fold-level scores are in `cv.fold_scores`, and the
per-rank averages are in `cv.rank_scores`.

For consensus embeddings and a full pipeline, see [Examples](examples.md).
