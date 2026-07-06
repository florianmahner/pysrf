<img src="assets/graph.svg" alt="PySRF" align="right" width="180">

# PySRF

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-4B8BBE?style=flat-square&logo=python&logoColor=FFD43B)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-A78BFA?style=flat-square)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-06B6D4?style=flat-square&logo=readthedocs&logoColor=white)](https://florianmahner.github.io/pysrf/)
[![CI](https://img.shields.io/github/actions/workflow/status/florianmahner/pysrf/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/florianmahner/pysrf/actions/workflows/ci.yml)
[![ruff](https://img.shields.io/badge/code%20style-ruff-261230?style=flat-square&logo=ruff&logoColor=D7FF64)](https://github.com/astral-sh/ruff)

PySRF implements Similarity-Based Representation Factorization (SRF): given a symmetric, non-negative similarity matrix `S`, it reveals a small set of sparse, non-negative dimensions `W` with `S ≈ WWᵀ`. It learns from the observed entries only (no imputation) and selects the number of dimensions by cross-validation.

## Install

```bash
git clone https://github.com/florianmahner/pysrf.git && cd pysrf && ./setup.sh
```

## Quick start

```python
from pysrf import SRF

model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
```

See **[florianmahner.github.io/pysrf](https://florianmahner.github.io/pysrf/)** for installation details, walkthroughs, examples, and the API reference.
