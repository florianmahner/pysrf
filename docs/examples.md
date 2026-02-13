# Examples

These examples demonstrate the full range of pysrf features, from basic
factorization to a complete analysis pipeline.

## Basic factorization

Generate a low-rank similarity matrix and recover its embedding:

```python
import numpy as np
from pysrf import SRF

# Ground-truth low-rank matrix
n, rank = 100, 10
w_true = np.random.rand(n, rank)
s = w_true @ w_true.T

# Fit the model
model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
s_hat = model.reconstruct()
```

Because the input is exactly rank-10, the reconstruction error is near zero.

## Missing data

Mark entries as `NaN` to simulate incomplete observations. The model fits on
the observed entries and fills in the gaps:

```python
import numpy as np
from pysrf import SRF

n, rank = 100, 10
w_true = np.random.rand(n, rank)
s = w_true @ w_true.T

# Remove 30% of entries
mask = np.random.rand(n, n) < 0.3
s[mask] = np.nan

model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()
```

## Cross-validation for rank selection

Sweep over candidate ranks and let `cross_val_score` pick the best one. Set
`estimate_sampling_fraction=True` to derive the hold-out fraction from the
data:

```python
from pysrf import cross_val_score, SRF

cv = cross_val_score(
    s,
    estimate_sampling_fraction=True,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    n_jobs=-1,
    random_state=42,
)

print(f"Best rank: {cv.best_params_['rank']}")
print(f"Best score: {cv.best_score_:.4f}")
```

## Ensemble and consensus clustering

Combine multiple factorization runs into a stable consensus embedding, then
cluster the result. The pipeline uses scikit-learn's `Pipeline` interface:

```python
from sklearn import pipeline
from pysrf.consensus import EnsembleEmbedding, ClusterEmbedding
from pysrf import SRF, cross_val_score

# 1. Select the rank
cv = cross_val_score(
    s,
    estimate_sampling_fraction=True,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    n_jobs=-1,
)

# 2. Build the ensemble-clustering pipeline
pipe = pipeline.Pipeline(
    [
        ("ensemble", EnsembleEmbedding(SRF(cv.best_params_), n_runs=50)),
        ("cluster", ClusterEmbedding(min_clusters=2, max_clusters=6, step=1)),
    ]
)

consensus_embedding = pipe.fit_transform(s)
```

`EnsembleEmbedding` runs the factorization `n_runs` times and averages the
aligned embeddings. `ClusterEmbedding` applies consensus clustering to find
stable groups.

## Value bounds

Constrain reconstructed values to a fixed range. This is useful when the
similarity measure has a known domain, such as [0, 1] for cosine similarity:

```python
from pysrf import SRF

model = SRF(rank=10, bounds=(0, 1), random_state=42)
w = model.fit_transform(s)
s_reconstructed = model.reconstruct()

assert s_reconstructed.min() >= 0
assert s_reconstructed.max() <= 1
```

## Sampling-bound estimation

Estimate the range of hold-out fractions that produce reliable
cross-validation scores:

```python
from pysrf import estimate_sampling_bounds_fast

pmin, pmax, s_denoised = estimate_sampling_bounds_fast(
    s,
    n_jobs=-1,
    random_state=42,
)

print(f"Minimum sampling rate: {pmin:.4f}")
print(f"Maximum sampling rate: {pmax:.4f}")

# Use the midpoint for cross-validation
sampling_rate = 0.5 * (pmin + pmax)
```

## Complete workflow

This example ties together every step: data generation, noise injection,
missing data, sampling-bound estimation, cross-validation, model fitting, and
evaluation.

```python
import numpy as np
from pysrf import SRF, cross_val_score, estimate_sampling_bounds_fast

# 1. Generate a low-rank matrix with noise and missing entries
np.random.seed(42)
n, true_rank = 100, 8
w_true = np.random.rand(n, true_rank)
s = w_true @ w_true.T
s += 0.1 * np.random.randn(n, n)
s = (s + s.T) / 2
mask = np.random.rand(n, n) < 0.2
s[mask] = np.nan

# 2. Estimate sampling bounds
pmin, pmax, _ = estimate_sampling_bounds_fast(s, n_jobs=-1)
print(f"Sampling bounds: [{pmin:.3f}, {pmax:.3f}]")

# 3. Cross-validate to find the best rank
result = cross_val_score(
    s,
    param_grid={"rank": range(5, 21)},
    estimate_sampling_fraction=True,
    n_repeats=3,
    n_jobs=-1,
    random_state=42,
)
best_rank = result.best_params_["rank"]
print(f"Best rank: {best_rank} (true rank: {true_rank})")

# 4. Fit the final model
model = SRF(rank=best_rank, max_outer=20, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()

# 5. Evaluate
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```
