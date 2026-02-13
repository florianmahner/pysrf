# pysrf

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img src="assets/logo.png" alt="pysrf logo" width="200" align="right">

Discover interpretable dimensions from representational similarities.
pysrf decomposes a symmetric similarity matrix $S$ into sparse,
non-negative dimensions $W$ such that $S \approx WW^\top$. It handles
incomplete data, estimates the number of dimensions through
cross-validation, and produces stable consensus embeddings.

## Installation

```bash
git clone https://github.com/fmahner/pysrf.git
cd pysrf
./setup.sh
```

See the [installation guide](https://fmahner.github.io/pysrf/installation/)
for manual setup, alternative methods, and troubleshooting.

## Quick example

```python
import numpy as np
from pysrf import SRF

s = np.random.rand(100, 100)
s = (s + s.T) / 2

model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)
s_reconstructed = model.reconstruct()
```

## Documentation

For the full guide, including examples, API reference, cross-validation, and
ensemble clustering, see the [documentation](https://fmahner.github.io/pysrf/).

## Reference

Mahner, F. P.\*, Lam, K. C.\*, & Hebart, M. N. (2025). Interpretable
dimensions from sparse representational similarities. *In preparation*.

## License

MIT
