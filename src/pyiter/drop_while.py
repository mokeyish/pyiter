from typing import Callable, Iterable, Iterator
from .transform import Transform, T


class DropWhileTransform(Transform[T, T]):
    """A transform that drops elements from an iterable while a predicate is true."""

    def __init__(self, iter: Iterable[T], predicate: Callable[[T], bool]):
        super().__init__(iter)
        self.predicate = predicate

    def __do_iter__(self) -> Iterator[T]:
        is_dropping = True
        for e in self.iter:
            if is_dropping and self.predicate(e):
                continue
            else:
                is_dropping = False
            yield e
