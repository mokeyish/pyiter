from typing import Iterable, Iterator
from .transform import Transform, T


class ConcatTransform(Transform[Iterable[T], T]):
    """Concatenate multiple iterables into a single iterable."""

    def __init__(self, iter: Iterable[Iterable[T]]):
        super().__init__(iter)

    def __do_iter__(self) -> Iterator[T]:
        for i in self.iter:
            yield from i
