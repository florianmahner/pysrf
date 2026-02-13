<p align="center">
  <img src="assets/logo.png" alt="pysrf" width="200">
</p>

<p align="center">
  <strong>
    Sparse non-negative factorization of similarity matrices
  </strong>
</p>

<p align="center">
  <a href="https://github.com/florianmahner/pysrf/actions/workflows/ci.yml"><img
    src="https://github.com/florianmahner/pysrf/actions/workflows/ci.yml/badge.svg"
    alt="CI"
  /></a>
  <a href="https://www.python.org/downloads/"><img
    src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white"
    alt="Python 3.10+"
  /></a>
  <a href="https://opensource.org/licenses/MIT"><img
    src="https://img.shields.io/badge/license-MIT-green"
    alt="License: MIT"
  /></a>
  <a href="https://github.com/astral-sh/ruff"><img
    src="https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=D7FF64"
    alt="ruff"
  /></a>
</p>

<p align="center">
  <a href="https://florianmahner.github.io/pysrf/"><strong>Docs</strong></a>
  &middot;
  <a href="https://florianmahner.github.io/pysrf/installation/"><strong>Install</strong></a>
  &middot;
  <a href="https://florianmahner.github.io/pysrf/quickstart/"><strong>Quick start</strong></a>
  &middot;
  <a href="https://florianmahner.github.io/pysrf/examples/"><strong>Examples</strong></a>
  &middot;
  <a href="https://florianmahner.github.io/pysrf/api/model/"><strong>API</strong></a>
</p>

<p align="center">
  Decompose a similarity matrix into sparse, interpretable dimensions:
  <em>S</em> ≈ <em>WW</em><sup>T</sup>.
  Handles missing data, estimates dimensionality via cross-validation, and
  produces stable consensus embeddings.
</p>

<p align="center">
  <img src="assets/factorization.svg" alt="S ≈ W × Wᵀ factorization diagram" width="800">
</p>

```python
from pysrf import SRF

model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
```

<p align="center">
  <em>
    Mahner, F. P.*, Lam, K. C.*, & Hebart, M. N. (2025).
    Interpretable dimensions from sparse representational similarities.
    <strong>In preparation</strong>.
  </em>
</p>
