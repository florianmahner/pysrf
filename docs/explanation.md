# What is SRF?

A similarity matrix `S` records how similar every pair of items is to one another. Knowing that a dog is more similar to a cow than to a car does not tell you *why* — whether the similarity is driven by animacy, size, shape, or some other property.

SRF (Similarity-Based Representation Factorization) reveals those properties. It factorizes a symmetric, non-negative similarity matrix into a small set of sparse, non-negative dimensions: `S ≈ WWᵀ`. Each row of `W` gives the loadings for one item, one value per dimension.

A similarity matrix can also be viewed as a weighted graph, and in that view SRF's dimensions are **soft community memberships** — each item gets a non-negative loading on every dimension, and a near-zero loading means that dimension simply doesn't apply to that item. For example, *lion* loads strongly on the animate dimension, while *ball* loads on both round and natural. Because loadings are non-negative, dimensions add up rather than cancel out, which makes the result easy to read.

## Why it works on real data

- **Missing data**: real similarity matrices are often incomplete. SRF learns directly from the observed entries, with no imputation — and can then predict the entries you never measured.
- **Dimensionality selection**: you usually don't know how many dimensions to use. `cross_val_score` calibrates the hold-out protocol and selects the SRF model rank by validation error.
- **Fast solver**: the core fit is Cython-accelerated, giving a 10–50× speedup over pure Python.

## When to use SRF

SRF works on any symmetric, non-negative similarity matrix, however it was produced:

- **Behavioral data**: any task that yields a measure of similarity between items.
- **Neural data**: similarity matrices from fMRI, electrophysiology, or other neural recordings.
- **Machine learning**: kernels built from deep neural network activations.
- **Graph representations**: any data you can convert to an adjacency graph, such as word-association networks.

## Citation

To cite PySRF, use the **Cite this repository** button on [GitHub](https://github.com/florianmahner/pysrf) (generated from [`CITATION.cff`](https://github.com/florianmahner/pysrf/blob/master/CITATION.cff)), or read the preprint at [arXiv:2605.26921](https://arxiv.org/abs/2605.26921).
