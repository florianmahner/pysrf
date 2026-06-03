---
hide:
  - navigation
  - toc
---

# PySRF

**Discover interpretable dimensions from (sparse) representational similarities.**

Across neuroscience, psychology, and machine learning, a common goal is comparing how different systems represent the same set of items. A standard way to do this is through a similarity matrix `S`, which records how similar every pair of items is to one another. But knowing that a dog is more similar to a cow than to a car does not tell you *why* — whether the similarity is driven by animacy, size, shape, or some other property.

PySRF answers that question. It implements SRF (Similarity-Based Representation Factorization), a method that decomposes a similarity matrix into a small number of sparse, non-negative dimensions: `S ≈ WW^T`. Each row of `W` gives the loadings for one item, one value per dimension. A similarity matrix can also be viewed as a weighted graph, and in that view SRF's dimensions are **soft community memberships** — each item gets a non-negative loading on every dimension, and a near-zero loading means that dimension simply doesn't apply to that item. For example, *lion* loads strongly on the animate dimension, while *ball* loads on both round and natural. Because loadings are non-negative, dimensions add up rather than cancel out, which makes the result easy to read.

## When to use PySRF

PySRF works on any symmetric, non-negative similarity matrix, however it was produced:

- **Behavioral data**: any task that yields a measure of similarity between items.
- **Neural data**: similarity matrices from fMRI, electrophysiology, or other neural recordings.
- **Machine learning**: kernels built from deep neural network activations.
- **Graph representations**: any data you can convert to an adjacency graph, such as word-association networks.

## Key capabilities

- **Missing data**: real similarity matrices are often incomplete. PySRF learns directly from the observed entries, with no imputation — and can then predict the entries you never measured.
- **Dimensionality estimation**: you usually don't know how many dimensions to use. `estimate_rank` estimates that number for you, and `cross_val_score` lets you confirm it by comparing nearby ranks.
- **Fast solver**: the core fit is Cython-accelerated, giving a 10–50× speedup over pure Python.

## Quick example

```python
import numpy as np
from pysrf import SRF, estimate_rank

# Your similarity matrix (e.g. from behavioral judgments or neural data)
s = np.random.rand(100, 100)
s = (s + s.T) / 2

# Estimate how many dimensions the data supports, then fit
estimate = estimate_rank(s, random_state=42)
model = SRF(rank=estimate.rank, random_state=42)
w = model.fit_transform(s)

# Reconstruct the similarity matrix from the learned dimensions
s_reconstructed = model.reconstruct()
```

## Reference

Mahner, F. P.\*, Lam, K. C.\*, Pereira, F., & Hebart, M. N. (2026).
*Revealing the core dimensions underlying representations in brains, behavior and AI.*
arXiv:2605.26921. (\* equal contribution).
[https://arxiv.org/abs/2605.26921](https://arxiv.org/abs/2605.26921)

## Next steps

- [Installation](installation.md): set up PySRF and compile the Cython extensions.
- [Quick start](quickstart.md): walk through the core workflow step by step.
- [Examples](examples.md): explore advanced features and full pipelines.
- [API reference](api/model.md): browse the complete API.
- [Development](development.md): contribute to PySRF.
