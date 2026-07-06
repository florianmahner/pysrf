---
hide:
  - navigation
  - toc
---

# PySRF

PySRF implements SRF (Similarity-Based Representation Factorization): given a symmetric, non-negative similarity matrix `S`, it reveals a small set of sparse, non-negative dimensions `W` with `S ≈ WWᵀ`. It learns from the observed entries only (no imputation) and selects the number of dimensions by cross-validation.

New here? See [What is SRF?](explanation.md) for how the method works and when to use it, or jump straight to [Installation](installation.md) and the [Quick start](quickstart.md).

## Quick example

```python
import numpy as np
from pysrf import SRF, cross_val_score

# Your similarity matrix (e.g. from behavioral judgments or neural data)
s = np.random.rand(100, 100)
s = (s + s.T) / 2

# Select the model rank, then fit
cv = cross_val_score(s, range(1, 21), random_state=42)
model = SRF(rank=cv.model_rank, random_state=42)
w = model.fit_transform(s)

# Reconstruct the similarity matrix from the learned dimensions
s_reconstructed = model.reconstruct()
```

## Next steps

- [What is SRF?](explanation.md): how the method works and when to use it.
- [Installation](installation.md): set up PySRF and compile the Cython extensions.
- [Quick start](quickstart.md): walk through the core workflow step by step.
- [Examples](examples.md): explore advanced features and full pipelines.
- [API reference](api/model.md): browse the complete API.
- [Development](development.md): contribute to PySRF.
