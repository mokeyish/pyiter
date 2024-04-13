"""
PyIter is a Python package for iterative operations inspired by the Kotlin、CSharp(linq)、
TypeSrcipt and Rust . Enables strong typing and type inference for iterative operations.

Example:
>>> from pyiter import iterate as it
>>> from tqdm import tqdm

>>> text = ["hello", "world"]
>>> it(text).map(str.upper).to_list()
['HELLO', 'WORLD']

>>> # use tqdm
>>> it(range(10)).map(lambda x: str(x)).progress(lambda x: tqdm(x, total=x.len)).parallel_map(lambda x: x, max_workers=5).to_list()
"""

from .sequence import sequence, seq, iterate, it, Sequence

__ALL__ = [sequence, seq, iterate, it, Sequence] # type: ignore
