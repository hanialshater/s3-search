"""The operators (docs/INTERFACES.md §6). Each takes a ``Backend`` first and calls only its methods and the arrays'
own operators (§0). Ported from the prototype (baseline/esci_gpu_vs_opensearch.ipynb)."""
from __future__ import annotations

from .lexical import bm25f, bm25f_groups, bm25f_multi
from .dense import dense_exact, dense_scores
from .filters import range_mask, term_mask
from .facets import facet_terms, top_buckets
from .page import page
from .fusion import rrf
from .features import FEATURE_BASE, feature_names, gather_features

__all__ = ["bm25f", "bm25f_groups", "bm25f_multi", "dense_exact", "dense_scores", "term_mask", "range_mask", "facet_terms", "top_buckets", "page",
           "rrf", "FEATURE_BASE", "feature_names", "gather_features"]
