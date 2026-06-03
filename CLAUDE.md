# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.



## Git

- **Never add Claude as a co-author on commits** — no `Co-Authored-By` lines referencing Claude or Anthropic

## Coding Standards

- **Reuse existing code**: ALWAYS search `src/` first before writing new utilities
- **Paths**: Always use `pathlib.Path`, reference data via `cfg.data_dir`
- **Outputs**: Use `Path.cwd()` (Hydra changes to output dir)
- **Parallelism**: `joblib.Parallel` locally, `hydra/launcher=slurm` for cluster
- **Plotting**: `seaborn`/`matplotlib`, save as `.pdf` for experiments, `.png` for sandbox
- **Variables**: Always lowercase (`w`, `x`, `s`), never uppercase (`W`, `X`, `S`)
- **Atomic functions**: Break complex operations into small, reusable helpers prefixed with `_`
- **No complex one-liners**: Use explicit loops instead of dense list comprehensions

```python
# Good: explicit loop
r_obs = np.zeros(k)
for d in range(k):
    r_obs[d] = _correlation(w[:, d], x[:, d])

# Bad: dense one-liner
r_obs = np.array([_correlation(w[:, d], x[:, d]) for d in range(k)])
```

**Atomic functions example:**
```python
def _correlation(a, b, two_sided=True):
    r = pearsonr(a, b).statistic
    return np.abs(r) if two_sided else r

def _pvalue(obs, null):
    return (np.sum(null >= obs) + 1) / (len(null) + 1)

def permutation_test(a, b, permutations=1000, two_sided=True):
    r_obs = _correlation(a, b, two_sided)
    null = np.array([_correlation(a, rng.permutation(b), two_sided) for _ in range(permutations)])
    return _pvalue(r_obs, null), null, r_obs
```
