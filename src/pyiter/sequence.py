"""
Deprecated module. The stream implementation now lives in `pyiter.stream`.

This module is kept only for backwards compatibility so that existing code
doing `from pyiter.sequence import ...` continues to work. It re-exports every
public name from `pyiter.stream`. Please import from `pyiter.stream` (or the
top-level `pyiter` package) instead.
"""

from .stream import (  # re-export for backwards compatibility
    AutoIncrementIndex,
    IndexedValue,
    Sequence,
    SequenceProducer,
    Stream,
    it,
    is_debugging,
    none_or,
    none_or_else,
    seq,
    sequence,
    throw,
)

__all__ = [
    "AutoIncrementIndex",
    "IndexedValue",
    "Sequence",
    "SequenceProducer",
    "Stream",
    "is_debugging",
    "it",
    "none_or",
    "none_or_else",
    "seq",
    "sequence",
    "throw",
]
