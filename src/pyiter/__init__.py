"""
PyIter is a Python package for iterative operations inspired by the Kotlin、CSharp(linq)、
TypeSrcipt and Rust . Enables strong typing and type inference for iterative operations.
"""

from .sequence import sequence, seq, iterate, it, Sequence

__ALL__ = [sequence, seq, iterate, it, Sequence] # type: ignore
