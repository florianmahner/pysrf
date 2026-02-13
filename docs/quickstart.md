# Quick start

This guide walks you through the core pysrf workflows: factorizing a
similarity matrix, handling missing data, and selecting the factorization rank
with cross-validation.

## Factorize a symmetric matrix

Create an `SRF` model, choose a rank, and call `fit_transform` to obtain the
non-negative embedding **W**. The reconstruction **WW**ᵀ approximates the
original matrix.

```python
import numpy as np
from pysrf import SRF

# Generate or load your similarity matrix
s = np.random.rand(100, 100)
s = (s + s.T) / 2  # ensure symmetry

# Fit the model
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)

# Reconstruct the matrix
s_reconstructed = model.reconstruct()
# or equivalently: s_reconstructed = w @ w.T

# Evaluate fit
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```

The `score` method returns the Frobenius-norm reconstruction error, which you
can use to compare models at different ranks.

## Handle missing data

Pass a matrix that contains `NaN` entries. Set `missing_values=np.nan` so the
model ignores those positions during fitting and fills them in during
reconstruction.

```python
import numpy as np
from pysrf import SRF

# Matrix with 30% missing values
s = np.random.rand(100, 100)
s = (s + s.T) / 2
s[np.random.rand(100, 100) < 0.3] = np.nan

# Fit — missing entries are excluded automatically
model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()
```

The completed matrix `s_completed` contains predicted values in place of the
original `NaN` entries.

## Select the rank with cross-validation

Use `cross_val_score` to evaluate a grid of candidate ranks. The function
holds out a fraction of observed entries, fits the model on the rest, and
scores reconstruction quality on the held-out set.

```python
from pysrf import cross_val_score

s = np.random.rand(100, 100)
s = (s + s.T) / 2

cv = cross_val_score(s, param_grid={"rank": [5, 10, 15, 20]})

print(f"Best parameters: {cv.best_params_}")
print(f"Best score: {cv.best_score_:.4f}")
```

For a deeper dive into cross-validation options, sampling-bound estimation,
and ensemble clustering, see the [Examples](examples.md) page.
