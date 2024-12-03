from typing import Any, Callable, Iterable, Iterator, List, Optional
from .transform import Transform, T


class DedupTransform(Transform[T, List[T]]):
    """
    A transform that groups consecutive duplicates from an iterable.
    """

    def __init__(self, iter: Iterable[T], key_selector: Optional[Callable[[T], Any]]=None):
        super().__init__(iter)
        self.key_selector  = key_selector

    def __do_iter__(self) -> Iterator[List[T]]:
        duplicates: List[T] = []
        seen: Optional[Any] = None

        for e in self.iter:
            k = self.key_selector(e) if self.key_selector else e
            if k != seen:
                if len(duplicates) > 0:
                    yield duplicates
                duplicates = [e]
                seen = k
                continue
            duplicates.append(e)

        if len(duplicates) > 0:
            yield duplicates
