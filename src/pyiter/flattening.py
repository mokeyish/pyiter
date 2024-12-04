from typing import Iterable, Iterator
from .transform import Transform, T


class FlatteningTransform(Transform[Iterable[T], T]):
    """A transform that flattens an iterable of iterables."""

    def __init__(self, iter: Iterable[Iterable[T]]):
        super().__init__(iter)

    def __do_iter__(self) -> Iterator[T]:
        for i in self.iter:
            yield from i
