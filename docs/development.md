# Development

## Set up the development environment

### Automated setup

Run the setup script to install all tools and dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

The script checks for pyenv and Poetry, pins Python 3.12.4, installs dependencies with `poetry install --no-root --all-extras`, compiles the Cython extension through an editable pip install, and runs the test suite.

### Manual setup

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install --no-root --all-extras
poetry run pip install -e . --no-build-isolation
poetry run pytest
```

Poetry cannot build the root package itself (meson-python backend), so dependencies are installed with `--no-root` and the package is compiled through an editable pip install.

## Makefile targets

The Makefile provides shortcuts for common tasks:

```bash
make install       # install the package with pip
make dev           # install dev dependencies and compile the Cython extension
make test          # run test suite
make test-cov      # run tests with coverage report
make lint          # run ruff linter
make format        # format code with ruff
make clean         # remove build artifacts
make docs          # build documentation with zensical
make docs-serve    # serve documentation locally with zensical
```

## Run tests

Run the full suite:

```bash
make test
```

Run tests with coverage:

```bash
make test-cov
```

Run a specific test file or test function:

```bash
poetry run pytest tests/test_model.py -v
poetry run pytest tests/test_model.py::test_srf_fit_complete_data -v
```

## Code quality

### Lint and format

Check code quality and auto-format:

```bash
make lint
make format
```

### Type hints

Use Python 3.10+ style type hints in all public functions:

```python
from __future__ import annotations

def my_function(x: np.ndarray, rank: int = 10) -> tuple[np.ndarray, float]:
    ...
```

## Cython extension

The performance-critical inner loop lives in `pysrf/_bsum.pyx`. meson-python compiles it during `make dev`:

```bash
make dev

# or
poetry run pip install -e . --no-build-isolation
```

`pysrf/_bsum.py` is a pure-Python fallback used when the compiled module is absent. Verify which implementation is active:

```python
from pysrf._bsum import BACKEND

print(BACKEND)  # 'cython' if compiled, 'python' otherwise
```

## Documentation

### Build and preview

PySRF uses [zensical](https://zensical.org) for documentation, configured in `mkdocs.yml`. Build and preview locally:

```bash
make docs          # build docs
make docs-serve    # serve locally at http://127.0.0.1:8000
```

### Docstring format

Use NumPy-style docstrings:

```python
def my_function(x: np.ndarray, param: int = 10) -> float:
    """
    Brief description.

    Longer description explaining the function's purpose and behavior.

    Parameters
    ----------
    x : ndarray
        Description of x.
    param : int, default=10
        Description of param.

    Returns
    -------
    result : float
        Description of return value.

    Examples
    --------
    >>> result = my_function(np.array([1, 2, 3]))
    >>> print(result)
    0.123
    """
    ...
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes.
4. Run tests: `make test`
5. Format code: `make format`
6. Commit: `git commit -m "Add feature"`
7. Push: `git push origin feature-name`
8. Open a pull request.

### Guidelines

- Write tests for new features.
- Maintain type hints.
- Update documentation.
- Keep changes focused.
- Follow existing code style.

## Publish to PyPI

Version lives in the `[project]` table of `pyproject.toml`. Update it, then build a wheel and sdist with `python -m build` (meson-python backend) and upload with twine:

```bash
python -m build
twine upload dist/*
```

For a test upload:

```bash
twine upload -r testpypi dist/*
```

## Project structure

```
pysrf/
├── pysrf/                    # main package
│   ├── __init__.py           # public API
│   ├── model.py              # SRF estimator
│   ├── cross_validation.py   # cross_val_score, CVResult
│   ├── consensus.py          # EnsembleFit, AlignedConsensus
│   ├── coherence/            # calibrate_cross_validation, CVCalibration
│   ├── _steps.py             # update steps
│   ├── _common.py            # shared helpers
│   ├── _bsum.py              # pure-Python fallback
│   └── _bsum.pyx             # Cython extension
├── benchmarks/               # bsum and fit benchmarks
├── tests/                    # test suite
├── docs/                     # documentation
├── mkdocs.yml                # docs configuration
├── meson.build               # Cython build (meson-python)
├── Makefile                  # development targets
├── setup.sh                  # automated setup
├── pyproject.toml            # project config
├── CITATION.cff              # citation metadata
└── README.md                 # project overview
```
