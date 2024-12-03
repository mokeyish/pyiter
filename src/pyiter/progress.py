from typing import Callable, Iterable, Iterator
from .transform import Transform, T


class ProgressTransform(Transform[T, T]):
    """
    A transform that applies a progress function to an iterable.
    """

    def __init__(self, iter: Iterable[T], progress_func: Callable[[Iterable[T]], Iterable[T]]):
        super().__init__(iter)
        self.progress_func = progress_func

    def __do_iter__(self) -> Iterator[T]:
        yield from self.progress_func(self.iter)
