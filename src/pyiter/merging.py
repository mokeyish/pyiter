from typing import Callable, Iterable, Iterator
from .transform import Transform, T, O, U

# Zipping ?
class MergingTransform(Transform[T, U]):
    """
    Merges two iterables into a single iterable by applying a transformation function to each pair of elements.
    """

    def __init__(
            self,
            iter: Iterable[T], 
            other_iter: Iterable[O], 
            transform: Callable[[T, O], U] = lambda a,b:(a,b)
        ):
        super().__init__(iter)
        self.other_iter = other_iter
        self.transform = transform

    def __do_iter__(self) -> Iterator[U]:
        iter1 = iter(self.iter)
        iter2 = iter(self.other_iter)
        while True:
            try:
                yield self.transform(next(iter1), next(iter2))
            except StopIteration:
                break
