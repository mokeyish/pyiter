"""
Deprecated module. The stream implementation now lives in `pyiter.stream`.

This stub is kept (hand-maintained) so that `from pyiter.sequence import ...`
keeps type-checking. It re-exports every public name from `pyiter.stream`.
Please import from `pyiter.stream` (or the top-level `pyiter` package) instead.

NOTE: pyright --createstub drops pure re-export modules, so this file is
preserved verbatim by the `just stubgen` recipe after regeneration.
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
