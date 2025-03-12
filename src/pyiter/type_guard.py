from typing import Callable, Iterable, Iterator
import sys
from .transform import Transform, T, U

if sys.version_info < (3, 10):
    from typing_extensions import TypeGuard
else:
    from typing import TypeGuard


class TypeGuardTransform(Transform[T, U]):
    """A transform that applies type guard to each item of the iterable."""

    def __init__(self, iter: Iterable[T], type_guard: Callable[[T], TypeGuard[U]]):
        super().__init__(iter)
        self.type_guard = type_guard

    def __do_iter__(self) -> Iterator[U]:
        for i in self.iter:
            if self.type_guard(i):
                yield i
