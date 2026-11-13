"""Filters as bitmaps (bool masks on the device; packed uint32 on disk)."""
from __future__ import annotations


def term_mask(b, col, values, missing=None):
    """``bool[N]``: ``col`` (keyword codes or numeric values) is one of ``values``; missing values never match."""
    m = b.isin(col, values)
    return m & ~missing if missing is not None else m


def range_mask(b, col, lo=None, hi=None, lo_inc=True, hi_inc=True, missing=None):
    """``bool[N]``: ``lo <(=) col <(=) hi``; open ends when ``None``."""
    m = b.full(col.shape[0], True, "bool")
    if lo is not None:
        m = m & ((col >= lo) if lo_inc else (col > lo))
    if hi is not None:
        m = m & ((col <= hi) if hi_inc else (col < hi))
    return m & ~missing if missing is not None else m
