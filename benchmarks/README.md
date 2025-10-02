# Benchmarks

This directory contains benchmarking scripts for measuring the performance of different components in the pysrf library.

## Available Benchmarks

### `bench_cython.py`
Benchmarks Cython performance across different matrix sizes and ranks.

**Usage:**
```bash
# Basic benchmark with default parameters
poetry run python benchmarks/bench_cython.py

# Custom sizes and ranks
poetry run python benchmarks/bench_cython.py --sizes 50 100 200 500 --ranks 5 10 20

# More iterations for better accuracy
poetry run python benchmarks/bench_cython.py --number 50 --repeat 20
```

**Parameters:**
- `--sizes`: Matrix sizes to test (default: [50, 100, 200, 500])
- `--ranks`: Ranks to test (default: [5, 10, 20])
- `--number`: Number of executions per repeat (default: 20)
- `--repeat`: Number of repeats for timeit (default: 10)

**Output:**
- Performance comparison across different matrix sizes
- Speedup ratios showing Cython performance improvement
- Summary table with timing results

### `bench_pbound.py`
Benchmarks the `estimate_sampling_bounds_fast` function for different matrix sizes.

**Usage:**
```bash
# Basic benchmark
poetry run python benchmarks/bench_pbound.py

# Custom parameters
poetry run python benchmarks/bench_pbound.py --sizes 100 200 500 1000
```

### `bench_admm.py`
Benchmarks the full SRF (ADMM) algorithm across different matrix sizes and ranks.

**Usage:**
```bash
# Basic benchmark
poetry run python benchmarks/bench_admm.py

# Custom parameters
poetry run python benchmarks/bench_admm.py --sizes 100 200 500 --ranks 5 10 20
```

## Testing vs Benchmarking

- **Testing**: Correctness verification (Cython vs Python) is handled in `tests/test_cython_correctness.py`
- **Benchmarking**: Performance measurement across different parameters is handled in `benchmarks/`

## Running All Benchmarks

To run all benchmarks with default parameters:

```bash
make benchmark
```

## Notes

- All benchmarks use `timeit` for accurate timing measurements
- Results show performance across different matrix sizes and ranks
- Cython compilation is handled automatically
- Correctness testing is separate from performance benchmarking
