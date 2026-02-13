<p align="center">
  <img src="assets/logo.png" alt="pysrf" width="200">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-A31F34?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://florianmahner.github.io/pysrf/"><img src="https://img.shields.io/badge/docs-online-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
</p>

---

Sparse non-negative factorization of similarity matrices: $S \approx WW^\top$.
Handles missing data, estimates dimensionality via cross-validation, and
produces stable consensus embeddings.

## Install

```bash
git clone https://github.com/florianmahner/pysrf.git && cd pysrf && ./setup.sh
```

## Usage

```python
from pysrf import SRF

model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
```

## Docs

Full guide at **[florianmahner.github.io/pysrf](https://florianmahner.github.io/pysrf/)**.

## Reference

Mahner, F. P.\*, Lam, K. C.\*, & Hebart, M. N. (2025). Interpretable
dimensions from sparse representational similarities. *In preparation*.
