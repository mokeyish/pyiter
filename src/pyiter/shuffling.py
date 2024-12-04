from typing import Iterable, Iterator, Optional, Union
from random import Random
from .transform import Transform, T


class ShufflingTransform(Transform[T, T]):
    """
    A transform that shuffles the elements of an iterable.
    """

    def __init__(self, iter: Iterable[T], random: Optional[Union[Random, str, int]] = None):
        super().__init__(iter)
        if random is None or isinstance(random, (str, int)):
            self.random = Random(random)
        else:
            self.random = random

    def __do_iter__(self) -> Iterator[T]:
        lst = list(self.iter)
        self.random.shuffle(lst)
        yield from lst
