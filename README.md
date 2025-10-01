# pysrf

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Similarity-based Representation Factorization (SRF)** - A Python package for symmetric non-negative matrix factorization using ADMM optimization, with support for missing data and optional bounded constraints.

## Features

- 🚀 **Fast**: Cython-optimized inner loop with Python fallback
- 🎯 **Robust**: Handles missing data (NaN values) gracefully
- 📊 **Complete**: Includes cross-validation and hyperparameter optimization
- 🔬 **Theoretical**: P-bound estimation for optimal sampling rates
- 🧪 **Well-tested**: 29 comprehensive tests with 100% pass rate
- 🐍 **Modern**: Python 3.10+ with full type hints

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/pysrf.git
cd pysrf

# Run setup script (installs Python 3.12, dependencies, compiles Cython)
bash setup.sh
```

### Manual Installation

```bash
# Install dependencies
poetry install

# Compile Cython extensions (optional but recommended)
make compile

# Run tests
make test
```

### Using pip (when published)

```bash
pip install pysrf
```

## Quick Example

```python
import numpy as np
from pysrf import SRF

# Generate synthetic data
n, rank = 50, 10
w_true = np.random.rand(n, rank)
similarity = w_true @ w_true.T

# Fit model
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(similarity)

# Reconstruct
similarity_hat = model.reconstruct()
print(f"Reconstruction error: {np.linalg.norm(similarity - similarity_hat):.4f}")
```

## Usage

### Basic Factorization

```python
from pysrf import SRF

model = SRF(
    rank=10,              # Number of latent dimensions
    rho=3.0,              # ADMM penalty parameter
    max_outer=10,         # Outer iterations
    max_inner=30,         # Inner iterations per outer
    init='random_sqrt',   # Initialization method
    random_state=42
)

# Fit and transform
w = model.fit_transform(similarity_matrix)

# Reconstruct similarity matrix
similarity_reconstructed = model.reconstruct()

# Score on test data
mse = model.score(test_similarity_matrix)
```

### Handling Missing Data

```python
import numpy as np

# Create matrix with missing values
similarity[np.random.rand(*similarity.shape) < 0.2] = np.nan

# Fit with missing data
model = SRF(
    rank=10,
    missing_values=np.nan,  # Mark NaN as missing
    bounds=(0, 1)            # Optional: bound reconstructed values
)
w = model.fit_transform(similarity)
```

### Cross-Validation

```python
from pysrf import cross_val_score

# Define parameter grid
param_grid = {
    'rank': [5, 10, 15, 20],
    'rho': [2.0, 3.0, 4.0]
}

# Run cross-validation
result = cross_val_score(
    similarity_matrix,
    param_grid=param_grid,
    n_repeats=5,
    observed_fraction=0.8,
    n_jobs=-1,  # Parallel execution
    random_state=42
)

print(f"Best parameters: {result.best_params_}")
print(f"Best score: {result.best_score_:.4f}")

# Access full results
print(result.cv_results_)
```

### P-Bound Estimation

Estimate theoretical bounds on sampling rate for matrix completion:

```python
from pysrf.p_bounds import estimate_p_bound, estimate_p_bound_fast

# Standard estimation
pmin, pmax, s_noise = estimate_p_bound(
    similarity_matrix,
    method='dyson',
    verbose=True
)
print(f"Optimal sampling rate: [{pmin:.3f}, {pmax:.3f}]")

# Fast parallel version
pmin, pmax, s_noise = estimate_p_bound_fast(
    similarity_matrix,
    n_jobs=-1
)
```

## API Reference

### `SRF` Class

Main model class for symmetric non-negative matrix factorization.

**Parameters:**
- `rank` (int): Number of latent dimensions
- `rho` (float): ADMM penalty parameter (default: 3.0)
- `max_outer` (int): Maximum outer iterations (default: 10)
- `max_inner` (int): Maximum inner iterations (default: 30)
- `tol` (float): Convergence tolerance (default: 1e-4)
- `verbose` (bool): Print progress (default: False)
- `init` (str): Initialization method - 'random', 'random_sqrt', 'nndsvd', 'nndsvdar', 'eigenspectrum'
- `random_state` (int | None): Random seed
- `missing_values` (float | None): Value to treat as missing (default: np.nan)
- `bounds` (tuple[float, float] | None): (min, max) bounds for reconstructed values

**Methods:**
- `fit(X)`: Fit model to similarity matrix X
- `transform(X)`: Return learned factor matrix
- `fit_transform(X)`: Fit and return factors
- `reconstruct(w=None)`: Reconstruct similarity matrix from factors
- `score(X)`: Compute MSE on observed entries

**Attributes:**
- `w_`: Learned factor matrix (n_samples, rank)
- `components_`: Alias for w_
- `n_iter_`: Number of iterations performed
- `history_`: Dict of optimization metrics per iteration

### Cross-Validation Functions

#### `cross_val_score()`

High-level cross-validation with grid search.

**Parameters:**
- `similarity_matrix` (np.ndarray): Input matrix
- `param_grid` (dict): Parameter grid to search
- `n_repeats` (int): Number of CV folds (default: 5)
- `observed_fraction` (float): Training fraction (default: 0.8)
- `random_state` (int): Random seed
- `n_jobs` (int): Number of parallel jobs (default: -1)

**Returns:**
- `ADMMGridSearchCV` object with attributes:
  - `best_params_`: Best hyperparameters
  - `best_score_`: Best validation score
  - `cv_results_`: DataFrame with all results

#### `ADMMGridSearchCV`

Grid search cross-validator for matrix completion.

#### `EntryMaskSplit`

Cross-validator that creates entry-wise train/validation splits for symmetric matrices.

### P-Bound Functions

#### `estimate_p_bound()`

Estimate lower and upper bounds on sampling rate.

**Parameters:**
- `S` (np.ndarray): Similarity matrix
- `gamma`, `eta`, `rho`: Parameters for pmin bound
- `method` (str): 'dyson' or 'monte_carlo' for pmax
- `verbose` (bool): Print diagnostics

**Returns:**
- `pmin` (float): Lower bound
- `pmax` (float): Upper bound  
- `S_noise` (np.ndarray): Potentially noise-augmented matrix

#### `estimate_p_bound_fast()`

Parallel version using joblib for faster computation.

#### `pmin_bound()`

Estimate lower bound only using concentration inequalities.

#### `p_upper_only_k()`

Estimate upper bound for rank-k factorization using random matrix theory.

## Development

### Setup Development Environment

```bash
# Using Makefile
make dev

# Or manually
poetry install
poetry run pysrf-compile
```

### Common Commands

```bash
make help          # Show all commands
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Run linters
make format        # Format code
make compile       # Compile Cython
make clean         # Remove build artifacts
make build         # Build distribution
```

### Running Tests

```bash
# All tests
poetry run pytest tests/ -v

# Specific test file
poetry run pytest tests/test_model.py -v

# With coverage
poetry run pytest tests/ --cov=pysrf --cov-report=html
```

## Requirements

- Python >= 3.10
- NumPy >= 1.23
- scikit-learn >= 1.3
- SciPy >= 1.10
- pandas >= 2.0
- joblib >= 1.0

**Optional (for compilation):**
- Cython >= 3.0
- g++ compiler

## How It Works

### Algorithm

pysrf implements symmetric non-negative matrix factorization (SymNMF) using the Alternating Direction Method of Multipliers (ADMM). The algorithm solves:

```
min_{W≥0, V} ||M ⊙ (S - V)||²_F + ρ/2 ||V - WW^T||²_F
```

where:
- S is the input similarity matrix
- M is an observation mask (handles missing data)
- W is the factor matrix (n × rank)
- V is an auxiliary variable
- ρ is the penalty parameter

### Key Features

1. **Missing Data**: Handles NaN values by masking them in the optimization
2. **Bounds**: Optional constraints on reconstructed values
3. **Fast Inner Loop**: Cython-optimized quartic root solver (Shi et al., 2016)
4. **Robust Initialization**: Multiple methods including NNDSVD
5. **Parallel CV**: Joblib-based parallel cross-validation

## Citation

If you use this package, please cite:

```bibtex
@software{pysrf2024,
  title={pysrf: Similarity-based Representation Factorization},
  author={Mahner, Florian},
  year={2024},
  url={https://github.com/yourusername/pysrf}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run `make format` and `make lint`
5. Submit a pull request

## Troubleshooting

### Cython compilation fails

The package will automatically fall back to a pure Python implementation if Cython compilation fails. For better performance, ensure you have:
- g++ compiler installed
- Cython and NumPy installed

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Then recompile
make compile
```

### Import errors

If you get import errors, ensure you're in the poetry environment:

```bash
poetry shell
```

Or prefix commands with `poetry run`:

```bash
poetry run python your_script.py
```

## Acknowledgments

Based on algorithms from:
- Shi et al. (2016): "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization"
- ADMM framework by Boyd et al.

Adapted from the `similarity-factorization` research codebase.
