from typing import Deque, Iterable, Iterator, List
from collections import deque
from .transform import Transform, T


class WindowedTransform(Transform[T, List[T]]):
    """
    A transform that yields windows of a given size from an iterable.
    If partial_windows is True, then windows that are smaller than the given size are yielded.
    If partial_windows is False, then only windows that are exactly the given size are yielded.
    """

    def __init__(self, iter: Iterable[T], size: int, step: int, partial_windows: bool):
        super().__init__(iter)
        self.size = size
        self.step = step
        self.partial_windows = partial_windows

    def __do_iter__(self) -> Iterator[List[T]]:
        window: Deque[T] = deque(maxlen=self.size)
        for e in self.iter:
            window.append(e)
            if len(window) == self.size:
                yield list(window)
            if len(window) == self.size:
                for _ in range(self.step):
                    window.popleft()

        if self.partial_windows and len(window) > 0:
            yield list(window)
