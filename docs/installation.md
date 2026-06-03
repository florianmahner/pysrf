# Installation

PySRF is built with the [meson-python](https://meson-python.readthedocs.io/)
backend and ships a Cython extension (`_bsum`) that accelerates the inner
solver. The backend compiles this extension for you during install, so you
need a working C compiler (`gcc` or `clang`) on your system.

!!! note "Cython speedup"
    The compiled extension gives a large speedup over the pure Python
    fallback. If compilation fails, PySRF still imports and runs using the
    Python implementation, just more slowly.

## Requirements

- Python 3.10 or newer (development targets 3.12.4).
- A C compiler for building the Cython extension.

PySRF is not yet published on PyPI, so install it from source.

## Install from source

```bash
git clone https://github.com/florianmahner/pysrf.git
cd pysrf
pip install .
```

Or install directly from GitHub without cloning first:

```bash
pip install "git+https://github.com/florianmahner/pysrf.git"
```

In both cases the meson-python backend compiles the `_bsum` Cython extension
automatically. The runtime dependencies (numpy, scipy, scikit-learn, pandas,
joblib, tqdm) are installed for you.

## Editable install for development

For an editable install that compiles the extension into a separate build
directory and keeps the source tree clean:

```bash
git clone https://github.com/florianmahner/pysrf.git
cd pysrf
pip install -e . --no-build-isolation
```

See the [Development](development.md) page for the full developer workflow,
test commands, and contribution guidelines.

## Verify the installation

```python
import pysrf

print(pysrf.__version__)
```

To check whether the fast Cython backend is active:

```python
from pysrf.model import _w_solver_backend

print(_w_solver_backend)  # 'cython' if compiled, 'python' otherwise
```

## Performance: multi-threaded BLAS

The SRF solver relies on BLAS routines (via scipy) that benefit from
multi-threading. Set `OMP_NUM_THREADS` **before** importing pysrf to enable
parallel linear algebra:

```bash
export OMP_NUM_THREADS=4
python my_script.py
```

Or at the top of a script, before any imports:

```python
import os

os.environ["OMP_NUM_THREADS"] = "4"

from pysrf import SRF
```

With `verbose=1`, SRF logs the active thread count at fit time.

## Next steps

Head to the [Quick start](quickstart.md) to fit your first SRF model.
