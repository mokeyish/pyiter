from typing import Any, Callable, Iterable, List, Union, overload
from .sequence import Sequence
from .transform import T, new_transform


class ListLike(List[T], Sequence[T]):
    def __init__(self, iterable: Iterable[T] = []):
        super().__init__(iterable)
        self.__transform__ = new_transform(iterable)

    @overload
    def count(self) -> int: ...
    @overload
    def count(self, predicate: Callable[[T], bool]) -> int: ...
    @overload
    def count(self, predicate: Callable[[T, int], bool]) -> int: ...
    @overload
    def count(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> int: ...
    def count(self, predicate: Union[Callable[..., bool], Any, None] = None) -> int:
        """
        Returns the number of elements in the Sequence that satisfy the specified [predicate] function.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).count()
        3
        >>> it(lst).count(lambda x: x > 0)
        3
        >>> it(lst).count(lambda x: x > 2)
        1
        """
        if predicate is None:
            return len(self)
        predicate = self.__callback_overload_warpper__(predicate)
        return sum(1 for i in self if predicate(i))
