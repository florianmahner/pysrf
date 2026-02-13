---
hide:
  - navigation
  - toc
---

# pysrf

**Discover interpretable dimensions from representational similarities.**

Representational similarity is a widely used tool for comparing biological
and artificial systems. It captures *how much* two items align, but not
*why* they align. pysrf closes this gap: it decomposes a similarity matrix
into sparse, non-negative dimensions that reveal the latent structure
driving the similarities.

Given a symmetric similarity matrix $S$, pysrf finds a non-negative
embedding $W$ such that $S \approx WW^\top$. Each column of $W$ is a
dimension, and each row gives an item's coordinates along those dimensions.
Because both entries and dimensions are non-negative and sparse, the result
is an additive, compositional representation where dimensions contribute
positively without canceling each other out.

## When to use pysrf

pysrf operates on similarity matrices from any domain:

- **Behavioral data**: odd-one-out judgments, pairwise similarity ratings,
  or any task that yields a measure of similarity between items.
- **Neural data**: representational similarity matrices derived from fMRI,
  electrophysiology, or other neural recordings.
- **Language**: word-association networks, co-occurrence matrices, or
  semantic similarity graphs.
- **Machine learning**: similarity matrices from deep neural network
  activations, kernel matrices, or model comparison studies.

## Key capabilities

- **Missing data**: real-world similarity matrices are often incomplete.
  pysrf learns from observed entries and predicts the missing ones, without
  imputation.
- **Rank estimation**: the number of dimensions is chosen through
  entry-wise cross-validation, holding out individual similarities and
  testing how well the model predicts them.
- **Consensus embeddings**: the non-convex objective admits multiple local
  minima. pysrf fits an ensemble of embeddings with different
  initializations, aligns them, and selects the most stable solution.
- **Bounded reconstruction**: constrain reconstructed values to a fixed
  range such as [0, 1].
- **Fast solver**: a Cython-accelerated ADMM inner loop provides 10-50x
  speedup over pure Python.

## Quick example

```python
import numpy as np
from pysrf import SRF

# Your similarity matrix (e.g., from behavioral judgments or neural data)
s = np.random.rand(100, 100)
s = (s + s.T) / 2

# Decompose into 10 interpretable dimensions
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)

# Reconstruct the similarity matrix from the dimensions
s_reconstructed = model.reconstruct()
```

## Reference

Mahner, F. P.\*, Lam, K. C.\*, & Hebart, M. N. (2025). Interpretable
dimensions from sparse representational similarities. *In preparation*.

## Next steps

- [Installation](installation.md): set up pysrf and compile Cython extensions.
- [Quick start](quickstart.md): walk through the core workflow step by step.
- [Examples](examples.md): explore advanced features and full pipelines.
- [API reference](api/model.md): browse the complete API.
- [Development](development.md): contribute to pysrf.
