"""
Hand-maintained stub for the `pyiter.sequence` backwards-compatibility module.

`pyiter.stream` only carries the current design (`Stream`, `StreamProducer`,
`it`); the deprecated entrypoints live in `pyiter.sequence`. This stub
declares the real definitions (sequence / seq / Sequence / SequenceProducer)
*and* re-exports the rest from `pyiter.stream`, so `from pyiter.sequence
import ...` keeps type-checking for every public name.

NOTE: `pyright --createstub` only emits the real definitions here (it does
not trace the `__all__` re-imports), so this file is preserved verbatim by
the `just stubgen` recipe after regeneration.
"""

from .stream import (  # re-export for backwards compatibility
    AutoIncrementIndex,
    IndexedValue,
    Stream,
    StreamProducer,
    it,
    is_debugging,
    none_or,
    none_or_else,
    throw,
)

sequence: StreamProducer = ...
seq: StreamProducer = ...
Sequence = Stream
SequenceProducer = StreamProducer

__all__ = [
    "AutoIncrementIndex",
    "IndexedValue",
    "Sequence",
    "SequenceProducer",
    "Stream",
    "StreamProducer",
    "is_debugging",
    "it",
    "none_or",
    "none_or_else",
    "seq",
    "sequence",
    "throw",
]
