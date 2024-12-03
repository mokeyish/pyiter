from typing import Iterable, Iterator, Tuple
from .transform import Transform, T


class CombinationTransform(Transform[T, Tuple[T, ...]]):
    """
    A transform that yields all possible combinations of n elements from the input iterable.
    """

    def __init__(self, iter: Iterable[T], n: int):
        super().__init__(iter)
        self.n = n

    def __do_iter__(self) -> Iterator[Tuple[T, ...]]:
        from itertools import combinations
        yield from combinations(self.iter, self.n)
