"""
Deprecated compatibility module.

The stream implementation now lives in `pyiter.stream`, and the current
entrypoint is `pyiter.it`. This module is kept only so that existing code
doing `from pyiter.sequence import ...` (or importing the deprecated
`sequence` / `seq` names) keeps working.

The deprecated entrypoints are *defined here* (not in `pyiter.stream`), so
`pyiter.stream` only carries the current design. Every other public name is
re-exported from `pyiter.stream`. Please import from `pyiter.stream` (or the
top-level `pyiter` package) instead.
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

# region deprecated entrypoints (defined here so pyiter.stream stays clean)
sequence: StreamProducer = StreamProducer()
"""
Deprecated. Use `pyiter.it` instead.

    Example 1:
>>> sequence('hello', 'world').map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 2:
>>> sequence(['hello', 'world']).map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 3:
>>> sequence(range(10)).map(lambda x: x*x).to_list()
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""

seq: StreamProducer = sequence
"""
Deprecated. Use `pyiter.it` instead.

    Example 1:
>>> seq('hello', 'world').map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 2:
>>> seq(['hello', 'world']).map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 3:
>>> seq(range(10)).map(lambda x: x*x).to_list()
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""

Sequence = Stream
"""
Deprecated. Use [Stream] instead.
"""

SequenceProducer = StreamProducer
"""
Deprecated. Use [StreamProducer] instead.
"""
# endregion

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
