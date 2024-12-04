from typing import Callable, DefaultDict, Generic, Iterable, Iterator, List, NamedTuple
from .transform import Transform, T, K
from .list_like import ListLike


class Grouping(NamedTuple, Generic[K, T]):
    key: K
    values: ListLike[T]


class GroupingTransform(Transform[T, Grouping[K, T]]):
    """
    A transform that groups elements of an iterable by a key function.
    """

    def __init__(self, iter: Iterable[T], key_func: Callable[[T], K]):
        super().__init__(iter)
        self.key_func = key_func

    def __do_iter__(self) -> Iterator[Grouping[K, T]]:
        from collections import defaultdict
        from .sequence import it

        d: DefaultDict[K, List[T]] = defaultdict(list)
        for e in self.iter:
            d[self.key_func(e)].append(e)
        yield from it(d.items()).map(lambda x: Grouping(x[0], ListLike(x[1])))
