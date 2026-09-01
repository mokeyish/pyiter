from typing import Callable, Iterable, Iterator
from .transform import Transform, T


class TakeUntilTransform(Transform[T, T]):
    """A transform that takes elements from an iterable until a predicate is true."""

    def __init__(self, iter: Iterable[T], predicate: Callable[[T], bool]):
        super().__init__(iter)
        self.predicate = predicate

    def __do_iter__(self) -> Iterator[T]:
        for e in self.iter:
            if self.predicate(e):
                break
            yield e
