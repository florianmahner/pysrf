# pysrf

**Symmetric non-negative matrix factorization with ADMM optimization.**

pysrf decomposes a symmetric similarity matrix **S** into a low-rank
non-negative embedding **W** such that **S** ≈ **WW**ᵀ. It handles missing
data, supports bounded constraints on reconstructed values, and estimates the
factorization rank through cross-validation.

## Key features

- **Missing data**: factorize matrices with missing entries (NaN) and recover
  a completed matrix.
- **Bounded reconstruction**: constrain reconstructed values to a range such
  as [0, 1].
- **Rank estimation**: select the factorization rank with built-in
  cross-validation and sampling-bound estimation.
- **Ensemble clustering**: build stable consensus embeddings from multiple
  factorization runs.
- **Fast ADMM solver**: Cython-accelerated inner loop provides 10-50x speedup
  over pure Python.

## Quick example

```python
import numpy as np
from pysrf import SRF

# Your similarity matrix
s = np.random.rand(100, 100)
s = (s + s.T) / 2  # make symmetric

# Fit the model
model = SRF(rank=10, max_outer=20, random_state=42)
embedding = model.fit_transform(s)

# Reconstruct
s_reconstructed = model.reconstruct()
```

## Next steps

- [Installation](installation.md): set up pysrf and compile Cython extensions.
- [Quick start](quickstart.md): walk through core workflows step by step.
- [Examples](examples.md): explore advanced features and full pipelines.
- [API reference](api/model.md): browse the complete API.
- [Development](development.md): contribute to pysrf.
