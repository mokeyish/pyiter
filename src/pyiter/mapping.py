from typing import Callable, Iterable, Iterator
from .transform import Transform, T, U


class MappingTransform(Transform[T, U]):
    """A transform that applies a function to each element of an iterable."""

    def __init__(self, iter: Iterable[T], transform: Callable[[T], U]):
        super().__init__(iter)
        self.transform = transform

    def __do_iter__(self) -> Iterator[U]:
        for i in self.iter:
            yield self.transform(i)
