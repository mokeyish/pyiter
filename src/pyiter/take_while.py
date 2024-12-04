from typing import Callable, Iterable, Iterator
from .transform import Transform, T


class TakeWhileTransform(Transform[T, T]):
    """A transform that takes elements from an iterable while a predicate is true."""

    def __init__(self, iter: Iterable[T], predicate: Callable[[T], bool]):
        super().__init__(iter)
        self.predicate = predicate

    def __do_iter__(self) -> Iterator[T]:
        for e in self.iter:
            if self.predicate(e):
                yield e
            else:
                break
