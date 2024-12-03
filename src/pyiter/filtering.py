from typing import Callable, Iterable, Iterator
from .transform import Transform, T


class FilteringTransform(Transform[T, T]):
    """Transform that filters elements based on a predicate"""
    def __init__(self, iter: Iterable[T], predicate: Callable[[T], bool]):
        super().__init__(iter)
        self.predicate = predicate

    def __do_iter__(self) -> Iterator[T]:
        for i in self.iter:
            if self.predicate(i):
                yield i
