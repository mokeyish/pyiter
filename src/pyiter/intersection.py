from typing import Iterable, Iterator, Set
from .transform import Transform, T


class IntersectionTransform(Transform[Iterable[T], T]):
    """A transform that yields the intersection of iterables."""

    def __init__(self, iter: Iterable[Iterable[T]]):
        super().__init__(iter)

    def __do_iter__(self) -> Iterator[T]:
        from .sequence import it
        iters = it(self.iter)
        seen: Set[T] = set()
        for v in iters.first():
            if v not in seen and iters.all(lambda iter: v in iter):
                yield v
                seen.add(v)