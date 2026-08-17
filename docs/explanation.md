# What is SRF?

A similarity matrix records how similar every pair of items is. Knowing that a dog has a more similar representation to a cow than to a car does not reveal whether this similarity is driven by animacy, naturalness, real-world size, shape, or other, potentially unknown properties.

SRF (Similarity-Based Representation Factorization) reveals such properties. It represents a symmetric, non-negative similarity matrix \(S\) with a low-rank symmetric factorization \(S \approx WW^\top\), where \(W\) is non-negative and each row of \(W\) holds one item's loadings across dimensions. As dimensions can only contribute positively to similarity, this yields an additive, parts-based representation that can be interpreted directly.

A similarity matrix can also be viewed as a weighted graph where edges connect similar items. In that view, dimensions revealed by SRF act as **soft community memberships**: each item receives a non-negative loading on every dimension, and a near-zero loading means a dimension does not apply to that item. For example, *lion* loads strongly on an animate dimension, while *ball* loads on both round and natural.

## Why it works on real data

- **Missing data**: many similarity datasets are incomplete because exhaustive sampling of all stimulus pairs is infeasible. SRF fits the model while treating unobserved entries as missing, so it learns from incomplete similarity matrices without imputing missing values. After fitting, it predicts every pair, including pairs that were never measured.
- **Dimensionality selection**: the number of dimensions is usually unknown. `cross_val_score` selects it with a restricted hold-out scheme designed specifically for similarity data, choosing the rank that best predicts held-out similarities.
- **Fast solver**: the core fit runs in compiled Cython, with a pure-Python fallback when the extension is not compiled.

## When to use SRF

SRF works on any symmetric, non-negative similarity matrix, however it was produced:

- **Behavioral data**: similarity judgments, including pairwise ratings and triplet odd-one-out tasks.
- **Neural data**: similarity matrices from fMRI, electrophysiology, or other neural recordings.
- **Machine learning**: kernels built from deep neural network activations.
- **Graph representations**: weighted graphs such as word association networks, where SRF learns from sparse, partially observed connections.

## Citation

To cite PySRF, use the **Cite this repository** button on [GitHub](https://github.com/florianmahner/pysrf) (generated from [`CITATION.cff`](https://github.com/florianmahner/pysrf/blob/master/CITATION.cff)), or read the preprint at [arXiv:2605.26921](https://arxiv.org/abs/2605.26921).
