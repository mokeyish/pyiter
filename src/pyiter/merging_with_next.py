from typing import Callable, Iterable, Iterator
from .transform import Transform, T, U


class MergingWithNextTransform(Transform[T, U]):
    """
    Transforms an iterable by applying a function to each element and the next element.
    """

    def __init__(self, iter: Iterable[T], transform: Callable[[T, T], U] = lambda a,b:(a,b)):
        super().__init__(iter)
        self.transform = transform

    def __do_iter__(self) -> Iterator[U]:
        it = iter(self.iter)
        try:
            c = next(it)
            while True:
                n = next(it)
                yield self.transform(c, n)
                c = n
        except StopIteration:
            pass