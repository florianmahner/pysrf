# Examples

These examples show how to use PySRF for increasingly complex analyses,
from a basic factorization to constructing similarity matrices, selecting
the rank, building stable consensus embeddings, and running a complete
workflow. Every snippet runs against the real PySRF API and uses
`random_state=42` for reproducibility.

## Basic factorization

Start with a similarity matrix that has exact low-rank structure: build it
from a known embedding `w_true` and ask SRF to recover the latent
dimensions.

```python
import numpy as np
from pysrf import SRF

# Ground-truth low-rank structure
rng = np.random.default_rng(42)
n, rank = 100, 10
w_true = rng.random((n, rank))
s = w_true @ w_true.T

# Recover the dimensions
model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
s_hat = model.reconstruct()

print(f"Reconstruction error: {model.score(s):.6f}")
```

Because the input has exact rank 10, the reconstruction error is close to
zero. Real similarity matrices contain noise and their rank is unknown, so
the later examples show how to estimate it.

## Missing data

Similarity datasets are often incomplete: behavioral experiments sample
only a fraction of all item pairs, and some entries are simply missing.
SRF learns directly from the observed entries and treats the rest as
unobserved. Here we drop about 30% of the entries to `NaN`.

```python
import numpy as np
from pysrf import SRF

rng = np.random.default_rng(42)
n, rank = 100, 10
w_true = rng.random((n, rank))
s = w_true @ w_true.T

# Mark ~30% of entries as missing
mask = rng.random((n, n)) < 0.3
s[mask] = np.nan

model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()  # fills in the missing pairs
```

SRF fits using only the observed pairs and then predicts the held-out ones
in `reconstruct()`. This beats imputing the gaps first: the paper (§2.1)
shows that k-nearest-neighbor or median imputation distorts the pairwise
similarity structure and biases the recovered dimensions, whereas leaving
entries unobserved does not.

## Rank estimation and cross-validation

The number of dimensions is usually unknown. PySRF first proposes a rank
with `estimate_rank`, then `cross_val_score` checks whether nearby ranks
give a lower held-out error using a hold-out scheme designed for
similarity data. We search a small window around the estimate with 5
repeats.

```python
import numpy as np
from pysrf import estimate_rank, cross_val_score

rng = np.random.default_rng(42)
n, rank = 100, 10
w_true = rng.random((n, rank))
s = w_true @ w_true.T

estimate = estimate_rank(s, random_state=42)

ranks = range(max(1, estimate.rank - 3), estimate.rank + 4)
curve = cross_val_score(
    s,
    ranks=ranks,
    sampling_fraction=estimate.sampling_fraction,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

cv_rank = curve.groupby("rank")["val_mse"].mean().idxmin()
print(f"Estimated rank: {estimate.rank}")
print(f"Cross-validation minimum: {int(cv_rank)}")
```

`cross_val_score` returns a tidy DataFrame with one row per
`(rep, fold, rank, val_mse)`; averaging `val_mse` per rank and taking the
minimum gives the cross-validated rank.

## Constructing similarity matrices from features

SRF needs a symmetric, non-negative (ideally positive semi-definite)
similarity matrix. When you start from feature vectors rather than direct
pairwise judgments, you turn them into similarities with a kernel. Which
kernel to use depends on whether the features are non-negative or signed
(paper §4.3).

For **non-negative features** (for example, count-like or rectified
activations), a linear kernel `S = X @ X.T` already yields a non-negative
matrix. Optionally rescale it to `[0, 1]` by dividing by its maximum.

```python
import numpy as np
from pysrf import SRF

rng = np.random.default_rng(42)
x = rng.random((100, 25))  # non-negative features

s = x @ x.T
s = s / s.max()  # optional: rescale to [0, 1]

w = SRF(rank=10, random_state=42).fit_transform(s)
```

For **signed features** (for example, z-scored neural responses or
centered DNN activations), use an RBF kernel, which is always
non-negative. Set the bandwidth with the median heuristic: `sigma` is
`0.4` times the median pairwise Euclidean distance, a multiplier shown to
work well for symmetric NMF.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
from pysrf import SRF

rng = np.random.default_rng(42)
x = rng.standard_normal((100, 25))  # signed features

# Median-heuristic bandwidth: sigma = 0.4 * median pairwise distance
median_distance = np.median(pdist(x, metric="euclidean"))
sigma = 0.4 * median_distance

# rbf_kernel uses gamma = 1 / (2 * sigma**2)
gamma = 1.0 / (2.0 * sigma**2)
s = rbf_kernel(x, gamma=gamma)

w = SRF(rank=10, random_state=42).fit_transform(s)
```

`squareform(pdist(x))` gives the full distance matrix if you want to
inspect it; only the median of the pairwise distances is needed to set the
bandwidth.

## Consensus embeddings

The SRF objective is non-convex, so different random initializations can
converge to different solutions. Part of this is just permutation
ambiguity (the same dimensions in a different order), but runs can also
land in distinct local minima. To get a stable embedding, fit many runs
and align them. `EnsembleFit` runs the base estimator from several random
starts and stacks the embeddings; `AlignedConsensus` aligns the runs with
the Hungarian algorithm and returns the most central one.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from pysrf import SRF, EnsembleFit, AlignedConsensus, estimate_rank

rng = np.random.default_rng(42)
n, rank = 100, 10
w_true = rng.random((n, rank))
s = w_true @ w_true.T

estimate = estimate_rank(s, random_state=42)

pipe = Pipeline(
    [
        ("ensemble", EnsembleFit(SRF(rank=estimate.rank, random_state=42), n_runs=50)),
        ("consensus", AlignedConsensus(rank=estimate.rank)),
    ]
)

consensus_embedding = pipe.fit_transform(s)
```

PySRF reports this representative run rather than averaging the runs,
because an averaged embedding need not be a valid SRF solution and can
distort the reconstructed similarities.

## Value bounds

When the similarity measure has a known range, you can constrain the
reconstruction to stay inside it. For example, the triplet odd-one-out
similarities in the paper are bounded in `[0, 1]`; pass `bounds=(0, 1)` so
the reconstructed entries respect that range.

```python
import numpy as np
from pysrf import SRF

rng = np.random.default_rng(42)
n, rank = 100, 10
w_true = rng.random((n, rank))
s = w_true @ w_true.T
s = s / s.max()  # bring into [0, 1]

model = SRF(rank=10, bounds=(0, 1), random_state=42)
w = model.fit_transform(s)
s_reconstructed = model.reconstruct()

assert s_reconstructed.min() >= 0
assert s_reconstructed.max() <= 1
```

## Complete workflow

A full analysis ties the pieces together: build a noisy, incomplete
similarity matrix, estimate the rank, confirm it with cross-validation,
fit the final model, reconstruct the missing entries, and report the
reconstruction error.

```python
import numpy as np
from pysrf import SRF, estimate_rank, cross_val_score

# 1. Build a noisy, incomplete similarity matrix
rng = np.random.default_rng(42)
n, true_rank = 100, 8
w_true = rng.random((n, true_rank))
s = w_true @ w_true.T
s += 0.1 * rng.standard_normal((n, n))
s = (s + s.T) / 2          # keep it symmetric
mask = rng.random((n, n)) < 0.2
s[mask] = np.nan           # ~20% missing

# 2. Estimate the rank
estimate = estimate_rank(s, random_state=42, n_jobs=-1)

# 3. Confirm with cross-validation over nearby ranks
ranks = range(max(1, estimate.rank - 3), estimate.rank + 4)
curve = cross_val_score(
    s,
    ranks=ranks,
    sampling_fraction=estimate.sampling_fraction,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)
cv_rank = int(curve.groupby("rank")["val_mse"].mean().idxmin())
print(f"Estimated rank: {estimate.rank}; cross-validation minimum: {cv_rank}")

# 4. Fit the final model at the chosen rank
model = SRF(rank=cv_rank, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()

# 5. Evaluate on the observed entries
print(f"Reconstruction error: {model.score(s):.4f}")
```
