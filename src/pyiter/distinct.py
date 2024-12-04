from typing import Any, Callable, Iterable, Iterator, Optional, Set
from .transform import Transform, T


class DistinctTransform(Transform[T, T]):
    """
    A transform that removes duplicates from an iterable.
    """

    def __init__(self, iter: Iterable[T], key_selector: Optional[Callable[[T], Any]] = None):
        super().__init__(iter)
        self.key_selector = key_selector

    def __do_iter__(self) -> Iterator[T]:
        seen: Set[Any] = set()
        for e in self.iter:
            k = self.key_selector(e) if self.key_selector else e
            if k not in seen:
                seen.add(k)
                yield e
