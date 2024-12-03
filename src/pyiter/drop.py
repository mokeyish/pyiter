from typing import Iterable, Iterator
from .transform import Transform, T


class DropTransform(Transform[T, T]):
    """A transform that drops the first n elements of an iterable."""

    def __init__(self, iter: Iterable[T], n: int):
        super().__init__(iter)
        self.n = n

    def __do_iter__(self) -> Iterator[T]:
        for i, e in enumerate(self.iter):
            if i < self.n:
                continue
            yield e