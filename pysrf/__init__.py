"""pysrf: Similarity-based Representation Factorization."""

from __future__ import annotations

from .model import SRF
from .cross_validation import (
    cross_val_score,
    ADMMGridSearchCV,
    EntryMaskSplit,
    mask_missing_entries,
)
from .bounds import (
    estimate_p_bound,
    estimate_p_bound_fast,
    pmin_bound,
    p_upper_only_k,
)

# Version is managed in pyproject.toml
try:
    from importlib.metadata import version

    __version__ = version("pysrf")
except Exception:
    __version__ = "0.1.0"  # fallback for development
__all__: list[str] = [
    "SRF",
    "cross_val_score",
    "ADMMGridSearchCV",
    "EntryMaskSplit",
    "mask_missing_entries",
    "estimate_p_bound",
    "estimate_p_bound_fast",
    "pmin_bound",
    "p_upper_only_k",
]
