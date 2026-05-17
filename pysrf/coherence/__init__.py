"""Eigenspace coherence for dimensionality estimation."""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

from ._estimate import CoherenceProfile, estimate_rank

__all__ = ["CoherenceProfile", "estimate_rank"]
