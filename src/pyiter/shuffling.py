from typing import Iterable, Iterator, Union
from random import Random
from .transform import Transform, T


class ShufflingTransform(Transform[T, T]):
    """
    A transform that shuffles the elements of an iterable.
    """

    def __init__(
        self,
        iter: Iterable[T],
        random: Union["Random", int, float, str, bytes, bytearray, None] = None,
    ):
        super().__init__(iter)
        if isinstance(random, Random):
            self.random = random
        elif random is None:
            self.random = Random()
        else:
            self.random = Random(random)

    def __do_iter__(self) -> Iterator[T]:
        lst = list(self.iter)
        self.random.shuffle(lst)
        yield from lst
