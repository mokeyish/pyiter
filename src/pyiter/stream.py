from __future__ import annotations

if __name__ == "__main__":
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from typing import (
    overload,
    Any,
    List,
    Set,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Union,
    Optional,
    Tuple,
    Type,
    Callable,
    Literal,
    NamedTuple,
    Awaitable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT
    from random import Random
    from .parallel_mapping import ParallelMappingTransform
    from .grouping import Grouping
    from .list_like import ListLike

# from typing_extensions import deprecated
from functools import cached_property
import sys


if sys.version_info < (3, 11):
    # Generic NamedTuple
    origin__namedtuple_mro_entries = NamedTuple.__mro_entries__  # type: ignore
    NamedTuple.__mro_entries__ = lambda bases: origin__namedtuple_mro_entries(bases[:1])  # type: ignore

from .transform import Transform, NonTransform, new_transform, T, U, K, O as V
from .error import LazyEvaluationException


class Stream(Generic[T], Iterable[T]):
    """
    Given an [iterator] function constructs a [Stream] that returns values through the [Iterator]
    provided by that function.

    The values are evaluated lazily, and the sequence is potentially infinite.
    """

    __transform__: Transform[Any, T]

    def __init__(self, iterable: Union[Iterable[T], Transform[Any, T]]) -> None:
        super().__init__()

        self.__transform__ = new_transform(iterable)

    @cached_property
    def transforms(self) -> Iterable[Transform[Any, Any]]:
        return [*self.__transform__.transforms()]

    @property
    def data(self) -> List[T]:
        if self.__transform__.cache is not None:
            return self.__transform__.cache.copy()
        if is_debugging():
            raise LazyEvaluationException("The sequence has not been evaluated yet.")
        return self.to_list()

    def dedup(self) -> Stream[T]:
        """
        Removes consecutive repeated elements in the sequence.

        If the sequence is sorted, this removes all duplicates.

        Example 1:
        >>> lst = [ 'a1', 'a1', 'b2', 'a2', 'a1']
        >>> it(lst).dedup().to_list()
        ['a1', 'b2', 'a2', 'a1']

        Example 1:
        >>> lst = [ 'a1', 'a1', 'b2', 'a2', 'a1']
        >>> it(lst).sorted().dedup().to_list()
        ['a1', 'a2', 'b2']
        """
        return self.dedup_by(lambda x: x)

    @overload
    def dedup_by(self, key_selector: Callable[[T], Any]) -> Stream[T]: ...
    @overload
    def dedup_by(self, key_selector: Callable[[T, int], Any]) -> Stream[T]: ...
    @overload
    def dedup_by(self, key_selector: Callable[[T, int, Stream[T]], Any]) -> Stream[T]: ...
    def dedup_by(self, key_selector: Callable[..., Any]) -> Stream[T]:
        """
        Removes all but the first of consecutive elements in the sequence that resolve to the same key.
        """
        return self.dedup_into_group_by(key_selector).map(lambda x: x[0])

    @overload
    def dedup_with_count_by(self, key_selector: Callable[[T], Any]) -> Stream[Tuple[T, int]]: ...
    @overload
    def dedup_with_count_by(
        self, key_selector: Callable[[T, int], Any]
    ) -> Stream[Tuple[T, int]]: ...
    @overload
    def dedup_with_count_by(
        self, key_selector: Callable[[T, int, Stream[T]], Any]
    ) -> Stream[Tuple[T, int]]: ...
    def dedup_with_count_by(self, key_selector: Callable[..., Any]) -> Stream[Tuple[T, int]]:
        """
        Removes all but the first of consecutive elements and its count that resolve to the same key.

        Example 1:
        >>> lst = [ 'a1', 'a1', 'b2', 'a2', 'a1']
        >>> it(lst).dedup_with_count_by(lambda x: x).to_list()
        [('a1', 2), ('b2', 1), ('a2', 1), ('a1', 1)]

        Example 1:
        >>> lst = [ 'a1', 'a1', 'b2', 'a2', 'a1']
        >>> it(lst).sorted().dedup_with_count_by(lambda x: x).to_list()
        [('a1', 3), ('a2', 1), ('b2', 1)]
        """
        return self.dedup_into_group_by(key_selector).map(lambda x: (x[0], len(x)))

    @overload
    def dedup_into_group_by(self, key_selector: Callable[[T], Any]) -> Stream[List[T]]: ...
    @overload
    def dedup_into_group_by(self, key_selector: Callable[[T, int], Any]) -> Stream[List[T]]: ...
    @overload
    def dedup_into_group_by(
        self, key_selector: Callable[[T, int, Stream[T]], Any]
    ) -> Stream[List[T]]: ...
    def dedup_into_group_by(self, key_selector: Callable[..., Any]) -> Stream[List[T]]:
        from .dedup import DedupTransform

        return it(DedupTransform(self, key_selector))

    @overload
    def filter(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def filter(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def filter(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def filter(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns a Stream containing only elements matching the given [predicate].

        Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter(lambda x: x.startswith('a')).to_list()
        ['a1', 'a2']

        Example 2:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter(lambda x, i: x.startswith('a') or i % 2 == 0 ).to_list()
        ['a1', 'b2', 'a2']
        """
        from .filtering import FilteringTransform

        return it(FilteringTransform(self, self.__callback_overload_warpper__(predicate)))

    def filter_is_instance(self, typ: Type[U]) -> Stream[U]:
        """
        Returns a Stream containing all elements that are instances of specified type parameter typ.

        Example 1:
        >>> lst = [ 'a1', 1, 'b2', 3]
        >>> it(lst).filter_is_instance(int).to_list()
        [1, 3]

        """
        from .type_guard import TypeGuardTransform, TypeGuard

        def guard(x: T) -> TypeGuard[U]:
            return isinstance(x, typ)

        return it(TypeGuardTransform(self, guard))

    @overload
    def filter_not(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def filter_not(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def filter_not(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def filter_not(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns a Stream containing all elements not matching the given [predicate].

        Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter_not(lambda x: x.startswith('a')).to_list()
        ['b1', 'b2']

        Example 2:
        >>> lst = [ 'a1', 'a2', 'b1', 'b2']
        >>> it(lst).filter_not(lambda x, i: x.startswith('a') and i % 2 == 0 ).to_list()
        ['a2', 'b1', 'b2']
        """
        predicate = self.__callback_overload_warpper__(predicate)
        return self.filter(lambda x: not predicate(x))

    @overload
    def filter_not_none(self: Stream[Optional[U]]) -> Stream[U]: ...
    @overload
    def filter_not_none(self: Stream[T]) -> Stream[T]: ...
    def filter_not_none(self: Stream[Optional[U]]) -> Stream[U]:
        """
        Returns a Stream containing all elements that are not `None`.

        Example 1:
        >>> lst = [ 'a', None, 'b']
        >>> it(lst).filter_not_none().to_list()
        ['a', 'b']
        """
        from .type_guard import TypeGuardTransform, TypeGuard

        def guard(x: Optional[U]) -> TypeGuard[U]:
            return x is not None

        return it(TypeGuardTransform(self, guard))

    @overload
    def map(self, transform: Callable[[T], U]) -> Stream[U]: ...
    @overload
    def map(self, transform: Callable[[T], U], return_exceptions: Literal[False]) -> Stream[U]: ...
    @overload
    def map(
        self, transform: Callable[[T], U], return_exceptions: Literal[True]
    ) -> Stream[Union[U, BaseException]]: ...
    @overload
    def map(self, transform: Callable[[T, int], U]) -> Stream[U]: ...
    @overload
    def map(
        self, transform: Callable[[T, int], U], return_exceptions: Literal[False]
    ) -> Stream[U]: ...
    @overload
    def map(
        self, transform: Callable[[T, int], U], return_exceptions: Literal[True]
    ) -> Stream[Union[U, BaseException]]: ...
    @overload
    def map(self, transform: Callable[[T, int, Stream[T]], U]) -> Stream[U]: ...
    @overload
    def map(
        self, transform: Callable[[T, int, Stream[T]], U], return_exceptions: Literal[False]
    ) -> Stream[U]: ...
    @overload
    def map(
        self, transform: Callable[[T, int, Stream[T]], U], return_exceptions: Literal[True]
    ) -> Stream[Union[U, BaseException]]: ...
    def map(
        self, transform: Callable[..., U], return_exceptions: bool = False
    ) -> Union[Stream[U], Stream[Union[U, BaseException]]]:
        """
        Returns a Stream containing the results of applying the given [transform] function
        to each element in the original Stream.

        Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).map(lambda x: x['age']).to_list()
        [12, 13]

        Example 2:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).map(lambda x, i: x['name'] + str(i)).to_list()
        ['A0', 'B1']

        Example 3:
        >>> lst = ['hi', 'abc']
        >>> it(lst).map(len).to_list()
        [2, 3]
        """
        from .mapping import MappingTransform

        transform = self.__callback_overload_warpper__(transform)
        if return_exceptions:

            def transform_wrapper(x: T) -> Union[U, BaseException]:
                try:
                    return transform(x)
                except BaseException as e:
                    return e

            return it(MappingTransform(self, transform_wrapper))

        return it(MappingTransform(self, transform))

    @overload
    async def map_async(self, transform: Callable[[T], Awaitable[U]]) -> Stream[U]: ...
    @overload
    async def map_async(
        self,
        transform: Callable[[T, int], Awaitable[U]],
        return_exceptions: Literal[True],
    ) -> Stream[Union[U, BaseException]]: ...
    @overload
    async def map_async(
        self,
        transform: Callable[[T, int, Stream[T]], Awaitable[U]],
        return_exceptions: Literal[False] = False,
    ) -> Stream[U]: ...
    async def map_async(
        self, transform: Callable[..., Awaitable[U]], return_exceptions: bool = False
    ) -> Union[Stream[U], Stream[Union[U, BaseException]]]:
        """
        Similar to `.map()` but you can input a async transform then await it.
        """
        from asyncio import gather

        if return_exceptions:
            return it(await gather(*self.map(transform), return_exceptions=True))
        return it(await gather(*self.map(transform)))

    @overload
    def map_not_none(self, transform: Callable[[T], Optional[U]]) -> Stream[U]: ...
    @overload
    def map_not_none(self, transform: Callable[[T, int], Optional[U]]) -> Stream[U]: ...
    @overload
    def map_not_none(self, transform: Callable[[T, int, Stream[T]], Optional[U]]) -> Stream[U]: ...
    def map_not_none(self, transform: Callable[..., Optional[U]]) -> Stream[U]:
        """
        Returns a Stream containing only the non-none results of applying the given [transform] function
        to each element in the original collection.

        Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': None}]
        >>> it(lst).map_not_none(lambda x: x['age']).to_list()
        [12]
        """
        return self.map(transform).filter_not_none()  # type: ignore

    @overload
    def parallel_map(
        self,
        transform: Callable[[T], U],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: ParallelMappingTransform.Executor = "Thread",
    ) -> Stream[U]: ...
    @overload
    def parallel_map(
        self,
        transform: Callable[[T, int], U],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: ParallelMappingTransform.Executor = "Thread",
    ) -> Stream[U]: ...
    @overload
    def parallel_map(
        self,
        transform: Callable[[T, int, Stream[T]], U],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: ParallelMappingTransform.Executor = "Thread",
    ) -> Stream[U]: ...
    def parallel_map(
        self,
        transform: Callable[..., U],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: ParallelMappingTransform.Executor = "Thread",
    ) -> Stream[U]:
        """
        Returns a Stream containing the results of applying the given [transform] function
        to each element in the original Stream.

        Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).parallel_map(lambda x: x['age']).to_list()
        [12, 13]

        Example 2:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).parallel_map(lambda x: x['age'], max_workers=2).to_list()
        [12, 13]

        Example 3:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).parallel_map(lambda x, i: x['age'] + i, max_workers=2).to_list()
        [12, 14]
        """
        from .parallel_mapping import ParallelMappingTransform

        return it(
            ParallelMappingTransform(
                self,
                self.__callback_overload_warpper__(transform),
                max_workers,
                chunksize,
                executor,
            )
        )

    @overload
    def find(self, predicate: Callable[[T], bool]) -> Optional[T]: ...
    @overload
    def find(self, predicate: Callable[[T, int], bool]) -> Optional[T]: ...
    @overload
    def find(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Optional[T]: ...
    def find(self, predicate: Callable[..., bool]) -> Optional[T]:
        """
        Returns the first element matching the given [predicate], or `None` if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).find(lambda x: x == 'b')
        'b'
        """
        return self.first_or_none(predicate)

    def find_last(self, predicate: Callable[[T], bool]) -> Optional[T]:
        """
        Returns the last element matching the given [predicate], or `None` if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).find_last(lambda x: x == 'b')
        'b'
        """
        return self.last_or_none(predicate)

    @overload
    def first(self) -> T: ...
    @overload
    def first(self, predicate: Callable[[T], bool]) -> T: ...
    @overload
    def first(self, predicate: Callable[[T, int], bool]) -> T: ...
    @overload
    def first(self, predicate: Callable[[T, int, Stream[T]], bool]) -> T: ...
    def first(self, predicate: Optional[Callable[..., bool]] = None) -> T:
        """
        Returns first element.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first()
        'a'

        Example 2:
        >>> lst = []
        >>> it(lst).first()
        Traceback (most recent call last):
        ...
        ValueError: Stream is empty.

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first(lambda x: x == 'b')
        'b'

        Example 4:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first(lambda x: x == 'd')
        Traceback (most recent call last):
        ...
        ValueError: Stream is empty.

        Example 5:
        >>> lst = [None]
        >>> it(lst).first() is None
        True
        """
        for e in self:
            if predicate is None or predicate(e):
                return e
        raise ValueError("Stream is empty.")

    @overload
    def first_not_none_of(self: Stream[Optional[U]]) -> U: ...
    @overload
    def first_not_none_of(
        self: Stream[Optional[U]], transform: Callable[[Optional[U]], Optional[U]]
    ) -> U: ...
    @overload
    def first_not_none_of(
        self: Stream[Optional[U]],
        transform: Callable[[Optional[U], int], Optional[U]],
    ) -> U: ...
    @overload
    def first_not_none_of(
        self: Stream[Optional[U]],
        transform: Callable[[Optional[U], int, Stream[Optional[U]]], Optional[U]],
    ) -> U: ...
    def first_not_none_of(
        self: Stream[Optional[U]],
        transform: Optional[Callable[..., Optional[U]]] = None,
    ) -> U:
        """
        Returns the first non-`None` result of applying the given [transform] function to each element in the original collection.

        Example 1:
        >>> lst = [{ 'name': 'A', 'age': None}, { 'name': 'B', 'age': 12}]
        >>> it(lst).first_not_none_of(lambda x: x['age'])
        12

        Example 2:
        >>> lst = [{ 'name': 'A', 'age': None}, { 'name': 'B', 'age': None}]
        >>> it(lst).first_not_none_of(lambda x: x['age'])
        Traceback (most recent call last):
        ...
        ValueError: No element of the Stream was transformed to a non-none value.
        """

        v = (
            self.first_not_none_of_or_none()
            if transform is None
            else self.first_not_none_of_or_none(transform)
        )
        if v is None:
            raise ValueError("No element of the Stream was transformed to a non-none value.")
        return v

    @overload
    def first_not_none_of_or_none(self) -> Optional[T]: ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T], T]) -> Optional[T]: ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T, int], T]) -> Optional[T]: ...
    @overload
    def first_not_none_of_or_none(
        self, transform: Callable[[T, int, Stream[T]], T]
    ) -> Optional[T]: ...
    def first_not_none_of_or_none(
        self, transform: Optional[Callable[..., T]] = None
    ) -> Optional[T]:
        """
        Returns the first non-`None` result of applying the given [transform] function to each element in the original collection.

        Example 1:
        >>> lst = [{ 'name': 'A', 'age': None}, { 'name': 'B', 'age': 12}]
        >>> it(lst).first_not_none_of_or_none(lambda x: x['age'])
        12

        Example 2:
        >>> lst = [{ 'name': 'A', 'age': None}, { 'name': 'B', 'age': None}]
        >>> it(lst).first_not_none_of_or_none(lambda x: x['age']) is None
        True
        """
        if transform is None:
            return self.first_or_none()
        return self.map_not_none(transform).first_or_none()

    @overload
    def first_or_none(self) -> Optional[T]: ...
    @overload
    def first_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]: ...
    @overload
    def first_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[T]: ...
    @overload
    def first_or_none(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Optional[T]: ...
    def first_or_none(self, predicate: Optional[Callable[..., bool]] = None) -> Optional[T]:
        """
        Returns the first element, or `None` if the Stream is empty.

        Example 1:
        >>> lst = []
        >>> it(lst).first_or_none() is None
        True

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first_or_none()
        'a'

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first_or_none(lambda x: x == 'b')
        'b'
        """
        if predicate is not None:
            return self.first_or_default(predicate, None)
        else:
            return self.first_or_default(None)

    @overload
    def first_or_default(self, default: U) -> Union[T, U]: ...
    @overload
    def first_or_default(self, predicate: Callable[[T], bool], default: U) -> Union[T, U]: ...
    @overload
    def first_or_default(self, predicate: Callable[[T, int], bool], default: U) -> Union[T, U]: ...
    @overload
    def first_or_default(
        self, predicate: Callable[[T, int, Stream[T]], bool], default: U
    ) -> Union[T, U]: ...
    def first_or_default(  # type: ignore
        self, predicate: Union[Callable[..., bool], U], default: Optional[U] = None
    ) -> Union[T, U, None]:
        """
        Returns the first element, or the given [default] if the Stream is empty.

        Example 1:
        >>> lst = []
        >>> it(lst).first_or_default('a')
        'a'

        Example 2:
        >>> lst = ['b']
        >>> it(lst).first_or_default('a')
        'b'

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first_or_default(lambda x: x == 'b', 'd')
        'b'

        Example 4:
        >>> lst = []
        >>> it(lst).first_or_default(lambda x: x == 'b', 'd')
        'd'
        """
        seq = self
        if isinstance(predicate, Callable):
            seq = self.filter(predicate)  # type: ignore
        else:
            default = predicate
        return next(iter(seq), default)

    @overload
    def last(self) -> T: ...
    @overload
    def last(self, predicate: Callable[[T], bool]) -> T: ...
    @overload
    def last(self, predicate: Callable[[T, int], bool]) -> T: ...
    @overload
    def last(self, predicate: Callable[[T, int, Stream[T]], bool]) -> T: ...
    def last(self, predicate: Optional[Callable[..., bool]] = None) -> T:
        """
        Returns last element.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last()
        'c'

        Example 2:
        >>> lst = []
        >>> it(lst).last()
        Traceback (most recent call last):
        ...
        ValueError: Stream is empty.
        """
        v = self.last_or_none(predicate) if predicate is not None else self.last_or_none()
        if v is None:
            raise ValueError("Stream is empty.")
        return v

    @overload
    def last_or_none(self) -> Optional[T]: ...
    @overload
    def last_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]: ...
    @overload
    def last_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[T]: ...
    @overload
    def last_or_none(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Optional[T]: ...
    def last_or_none(self, predicate: Optional[Callable[..., bool]] = None) -> Optional[T]:
        """
        Returns the last element matching the given [predicate], or `None` if no such element was found.

        Exmaple 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last_or_none()
        'c'

        Exmaple 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last_or_none(lambda x: x != 'c')
        'b'

        Exmaple 3:
        >>> lst = []
        >>> it(lst).last_or_none(lambda x: x != 'c') is None
        True
        """
        last: Optional[T] = None
        for i in self if predicate is None else self.filter(predicate):
            last = i
        return last

    def index_of_or_none(self, element: T) -> Optional[int]:
        """
        Returns first index of [element], or None if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_or_none('b')
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_or_none('d')
        """
        for i, x in enumerate(self):
            if x == element:
                return i
        return None

    def index_of(self, element: T) -> int:
        """
        Returns first index of [element], or -1 if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of('b')
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of('d')
        -1
        """
        return none_or(self.index_of_or_none(element), -1)

    def index_of_or(self, element: T, default: int) -> int:
        """
        Returns first index of [element], or default value if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_or('b', 1)
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_or('d', 0)
        0
        """
        return none_or(self.index_of_or_none(element), default)

    def index_of_or_else(self, element: T, f: Callable[[], int]) -> int:
        """
        Returns first index of [element], or computes the value from a callback if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_or_else('b', lambda: 2)
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_or_else('d', lambda: 0)
        0
        """
        return none_or_else(self.index_of_or_none(element), f)

    def last_index_of_or_none(self, element: T) -> Optional[int]:
        """
         Returns last index of [element], or None if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).last_index_of_or_none('b')
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last_index_of_or_none('d')
        """
        seq = self.reversed()
        last_idx = len(seq) - 1
        for i, x in enumerate(seq):
            if x == element:
                return last_idx - i
        return None

    def last_index_of(self, element: T) -> int:
        """
         Returns last index of [element], or -1 if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).last_index_of('b')
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last_index_of('d')
        -1
        """
        return none_or(self.last_index_of_or_none(element), -1)

    def last_index_of_or(self, element: T, default: int) -> int:
        """
         Returns last index of [element], or default value if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).last_index_of_or('b', 0)
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last_index_of_or('d', len(lst))
        3
        """
        return none_or(self.last_index_of_or_none(element), default)

    def last_index_of_or_else(self, element: T, f: Callable[[], int]) -> int:
        """
         Returns last index of [element], or computes the value from a callback if the collection does not contain element.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).last_index_of_or_else('b', lambda: 0)
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).last_index_of_or_else('d', lambda: len(lst))
        3
        """
        return none_or_else(self.last_index_of_or_none(element), f)

    @overload
    def index_of_first_or_none(self, predicate: Callable[[T], bool]) -> Optional[int]: ...
    @overload
    def index_of_first_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[int]: ...
    @overload
    def index_of_first_or_none(
        self, predicate: Callable[[T, int, Stream[T]], bool]
    ) -> Optional[int]: ...
    def index_of_first_or_none(self, predicate: Callable[..., bool]) -> Optional[int]:
        """
        Returns first index of element matching the given [predicate], or None if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or_none(lambda x: x == 'b')
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or_none(lambda x: x == 'd')
        """
        predicate = self.__callback_overload_warpper__(predicate)
        for i, x in enumerate(self):
            if predicate(x):
                return i
        return None

    @overload
    def index_of_first(self, predicate: Callable[[T], bool]) -> int: ...
    @overload
    def index_of_first(self, predicate: Callable[[T, int], bool]) -> int: ...
    @overload
    def index_of_first(self, predicate: Callable[[T, int, Stream[T]], bool]) -> int: ...
    def index_of_first(self, predicate: Callable[..., bool]) -> int:
        """
        Returns first index of element matching the given [predicate], or -1 if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first(lambda x: x == 'b')
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first(lambda x: x == 'd')
        -1

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first(lambda x: x == 'a')
        0
        """
        return none_or(self.index_of_first_or_none(predicate), -1)

    @overload
    def index_of_first_or(self, predicate: Callable[[T], bool], default: int) -> int: ...
    @overload
    def index_of_first_or(self, predicate: Callable[[T, int], bool], default: int) -> int: ...
    @overload
    def index_of_first_or(
        self, predicate: Callable[[T, int, Stream[T]], bool], default: int
    ) -> int: ...
    def index_of_first_or(self, predicate: Callable[..., bool], default: int) -> int:
        """
        Returns first index of element matching the given [predicate], or default value if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or(lambda x: x == 'b', 0)
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or(lambda x: x == 'd', 0)
        0

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or(lambda x: x == 'a', 0)
        0
        """
        return none_or(self.index_of_first_or_none(predicate), default)

    @overload
    def index_of_first_or_else(
        self, predicate: Callable[[T], bool], f: Callable[[], int]
    ) -> int: ...
    @overload
    def index_of_first_or_else(
        self, predicate: Callable[[T, int], bool], f: Callable[[], int]
    ) -> int: ...
    @overload
    def index_of_first_or_else(
        self, predicate: Callable[[T, int, Stream[T]], bool], f: Callable[[], int]
    ) -> int: ...
    def index_of_first_or_else(self, predicate: Callable[..., bool], f: Callable[[], int]) -> int:
        """
        Returns first index of element matching the given [predicate], or computes the value from a callback if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or_else(lambda x: x == 'b', lambda: len(lst))
        1

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or_else(lambda x: x == 'd', lambda: len(lst))
        3

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_first_or_else(lambda x: x == 'a', lambda: len(lst))
        0
        """
        return none_or_else(self.index_of_first_or_none(predicate), f)

    @overload
    def index_of_last_or_none(self, predicate: Callable[[T], bool]) -> Optional[int]: ...
    @overload
    def index_of_last_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[int]: ...
    @overload
    def index_of_last_or_none(
        self, predicate: Callable[[T, int, Stream[T]], bool]
    ) -> Optional[int]: ...
    def index_of_last_or_none(self, predicate: Callable[..., bool]) -> Optional[int]:
        """
        Returns last index of element matching the given [predicate], or -1 if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).index_of_last_or_none(lambda x: x == 'b')
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_or_none(lambda x: x == 'd')
        """
        seq = self.reversed()
        last_idx = len(seq) - 1
        predicate = self.__callback_overload_warpper__(predicate)
        for i, x in enumerate(seq):
            if predicate(x):
                return last_idx - i
        return None

    @overload
    def index_of_last(self, predicate: Callable[[T], bool]) -> int: ...
    @overload
    def index_of_last(self, predicate: Callable[[T, int], bool]) -> int: ...
    @overload
    def index_of_last(self, predicate: Callable[[T, int, Stream[T]], bool]) -> int: ...
    def index_of_last(self, predicate: Callable[..., bool]) -> int:
        """
        Returns last index of element matching the given [predicate], or -1 if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).index_of_last(lambda x: x == 'b')
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last(lambda x: x == 'd')
        -1

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last(lambda x: x == 'a')
        0
        """
        return none_or(self.index_of_last_or_none(predicate), -1)

    @overload
    def index_of_last_or(self, predicate: Callable[[T], bool], default: int) -> int: ...
    @overload
    def index_of_last_or(self, predicate: Callable[[T, int], bool], default: int) -> int: ...
    @overload
    def index_of_last_or(
        self, predicate: Callable[[T, int, Stream[T]], bool], default: int
    ) -> int: ...
    def index_of_last_or(self, predicate: Callable[..., bool], default: int) -> int:
        """
        Returns last index of element matching the given [predicate], or default value if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).index_of_last_or(lambda x: x == 'b', 0)
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_or(lambda x: x == 'd', -99)
        -99

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_or(lambda x: x == 'a', 0)
        0
        """
        return none_or(self.index_of_last_or_none(predicate), default)

    @overload
    def index_of_last_or_else(
        self, predicate: Callable[[T], bool], f: Callable[[], int]
    ) -> int: ...
    @overload
    def index_of_last_or_else(
        self, predicate: Callable[[T, int], bool], f: Callable[[], int]
    ) -> int: ...
    @overload
    def index_of_last_or_else(
        self, predicate: Callable[[T, int, Stream[T]], bool], f: Callable[[], int]
    ) -> int: ...
    def index_of_last_or_else(self, predicate: Callable[..., bool], f: Callable[[], int]) -> int:
        """
        Returns last index of element matching the given [predicate], or default value if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).index_of_last_or_else(lambda x: x == 'b', lambda: -len(lst))
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_or_else(lambda x: x == 'd', lambda: -len(lst))
        -3

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_or_else(lambda x: x == 'a', lambda: -len(lst))
        0
        """
        return none_or_else(self.index_of_last_or_none(predicate), f)

    @overload
    def single(self) -> T: ...
    @overload
    def single(self, predicate: Callable[[T], bool]) -> T: ...
    @overload
    def single(self, predicate: Callable[[T, int], bool]) -> T: ...
    @overload
    def single(self, predicate: Callable[[T, int, Stream[T]], bool]) -> T: ...
    def single(self, predicate: Optional[Callable[..., bool]] = None) -> T:
        """
        Returns the single element matching the given [predicate], or throws exception if there is no
        or more than one matching element.

        Exmaple 1:
        >>> lst = ['a']
        >>> it(lst).single()
        'a'

        Exmaple 2:
        >>> lst = []
        >>> it(lst).single() is None
        Traceback (most recent call last):
        ...
        ValueError: Stream contains no element matching the predicate.

        Exmaple 2:
        >>> lst = ['a', 'b']
        >>> it(lst).single() is None
        Traceback (most recent call last):
        ...
        ValueError: Stream contains more than one matching element.
        """
        single: Optional[T] = None
        found = False
        for i in self if predicate is None else self.filter(predicate):
            if found:
                raise ValueError("Stream contains more than one matching element.")
            single = i
            found = True
        if single is None:
            raise ValueError("Stream contains no element matching the predicate.")
        return single

    @overload
    def single_or_none(self) -> Optional[T]: ...
    @overload
    def single_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]: ...
    @overload
    def single_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[T]: ...
    @overload
    def single_or_none(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Optional[T]: ...
    def single_or_none(self, predicate: Optional[Callable[..., bool]] = None) -> Optional[T]:
        """
        Returns the single element matching the given [predicate], or `None` if element was not found
        or more than one element was found.

        Exmaple 1:
        >>> lst = ['a']
        >>> it(lst).single_or_none()
        'a'

        Exmaple 2:
        >>> lst = []
        >>> it(lst).single_or_none()

        Exmaple 2:
        >>> lst = ['a', 'b']
        >>> it(lst).single_or_none()

        """
        single: Optional[T] = None
        found = False
        for i in self if predicate is None else self.filter(predicate):
            if found:
                return None
            single = i
            found = True
        if not found:
            return None
        return single

    # noinspection PyShadowingNames
    def drop(self, n: int) -> Stream[T]:
        """
        Returns a Stream containing all elements except first [n] elements.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop(0).to_list()
        ['a', 'b', 'c']

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop(1).to_list()
        ['b', 'c']

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop(4).to_list()
        []
        """
        if n < 0:
            raise ValueError(f"Requested element count {n} is less than zero.")
        if n == 0:
            return self

        from .drop import DropTransform

        return it(DropTransform(self, n))

    # noinspection PyShadowingNames
    @overload
    def drop_while(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def drop_while(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def drop_while(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def drop_while(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns a Stream containing all elements except first elements that satisfy the given [predicate].

        Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).drop_while(lambda x: x < 3 ).to_list()
        [3, 4, 1]
        """
        from .drop_while import DropWhileTransform

        return it(DropWhileTransform(self, self.__callback_overload_warpper__(predicate)))

    # noinspection PyShadowingNames
    @overload
    def drop_until(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def drop_until(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def drop_until(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def drop_until(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns a Stream containing all elements except the first elements dropped until the first element that satisfies the given [predicate].

        Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).drop_until(lambda x: x >= 3).to_list()
        [3, 4, 1]

        Example 2:
        >>> lst = [1, 2, 1, 4]
        >>> it(lst).drop_until(lambda x: x == 4).to_list()
        [4]
        """
        from .drop_until import DropUntilTransform

        return it(DropUntilTransform(self, self.__callback_overload_warpper__(predicate)))

    def drop_last(self, n: int) -> Stream[T]:
        """
        Returns a Stream containing all elements except last [n] elements.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop_last(0).to_list()
        ['a', 'b', 'c']

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop_last(1).to_list()
        ['a', 'b']

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop_last(4).to_list()
        []
        """
        if n < 0:
            raise ValueError(f"Requested element count {n} is less than zero.")
        if n == 0:
            return self

        size = len(self)
        if size <= n:
            return Stream([])
        return self.take(size - n)

    def skip(self, n: int) -> Stream[T]:
        """
        Returns a Stream containing all elements except first [n] elements.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).skip(0).to_list()
        ['a', 'b', 'c']

         Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).skip(1).to_list()
        ['b', 'c']

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).skip(4).to_list()
        []
        """
        return self.drop(n)

    @overload
    def skip_while(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def skip_while(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def skip_while(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def skip_while(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns a Stream containing all elements except first elements that satisfy the given [predicate].

        Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).skip_while(lambda x: x < 3 ).to_list()
        [3, 4, 1]
        """
        return self.drop_while(predicate)

    def take(self, n: int) -> Stream[T]:
        """
        Returns an Stream containing first [n] elements.

        Example 1:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take(0).to_list()
        []

        Example 2:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take(2).to_list()
        ['a', 'b']
        """
        if n < 0:
            raise ValueError(f"Requested element count {n} is less than zero.")
        if n == 0:
            return Stream([])
        from .take import TakeTransform

        return it(TakeTransform(self, n))

    # noinspection PyShadowingNames
    @overload
    def take_while(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def take_while(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def take_while(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def take_while(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns an Stream containing first elements satisfying the given [predicate].

        Example 1:
        >>> lst = ['a', 'b', 'c', 'd']
        >>> it(lst).take_while(lambda x: x in ['a', 'b']).to_list()
        ['a', 'b']
        """
        from .take_while import TakeWhileTransform

        return it(TakeWhileTransform(self, self.__callback_overload_warpper__(predicate)))

    # noinspection PyShadowingNames
    @overload
    def take_until(self, predicate: Callable[[T], bool]) -> Stream[T]: ...
    @overload
    def take_until(self, predicate: Callable[[T, int], bool]) -> Stream[T]: ...
    @overload
    def take_until(self, predicate: Callable[[T, int, Stream[T]], bool]) -> Stream[T]: ...
    def take_until(self, predicate: Callable[..., bool]) -> Stream[T]:
        """
        Returns a Stream containing the first elements taken until the first element that satisfies the given [predicate].

        Example 1:
        >>> lst = [1, 2, 3, 4]
        >>> it(lst).take_until(lambda x: x > 2).to_list()
        [1, 2]

        Example 2:
        >>> lst = [1, 2, 3, 4]
        >>> it(lst).take_until(lambda x: x > 10).to_list()
        [1, 2, 3, 4]
        """
        from .take_until import TakeUntilTransform

        return it(TakeUntilTransform(self, self.__callback_overload_warpper__(predicate)))

    def take_last(self, n: int) -> Stream[T]:
        """
        Returns an Stream containing last [n] elements.

        Example 1:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(0).to_list()
        []

        Example 2:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(2).to_list()
        ['b', 'c']

        Example 3:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(10).to_list()
        ['a', 'b', 'c']
        """
        if n < 0:
            raise ValueError(f"Requested element count {n} is less than zero.")
        if n == 0:
            return Stream([])

        return self.drop(max(len(self) - n, 0))

    # noinspection PyShadowingNames
    def sorted(self) -> Stream[T]:
        """
        Returns an Stream that yields elements of this Stream sorted according to their natural sort order.

        Example 1:
        >>> lst = ['b', 'a', 'e', 'c']
        >>> it(lst).sorted().to_list()
        ['a', 'b', 'c', 'e']

         Example 2:
        >>> lst = [2, 1, 4, 3]
        >>> it(lst).sorted().to_list()
        [1, 2, 3, 4]
        """
        lst = list(self)
        lst.sort()  # type: ignore
        return it(lst)

    # noinspection PyShadowingNames
    @overload
    def sorted_by(self, key_selector: Callable[[T], SupportsRichComparisonT]) -> Stream[T]: ...
    @overload
    def sorted_by(self, key_selector: Callable[[T, int], SupportsRichComparisonT]) -> Stream[T]: ...
    @overload
    def sorted_by(
        self, key_selector: Callable[[T, int, Stream[T]], SupportsRichComparisonT]
    ) -> Stream[T]: ...
    def sorted_by(self, key_selector: Callable[..., SupportsRichComparisonT]) -> Stream[T]:
        """
        Returns a sequence that yields elements of this sequence sorted according to natural sort
        order of the value returned by specified [key_selector] function.

        Example 1:
        >>> lst = [ {'name': 'A', 'age': 12 }, {'name': 'C', 'age': 10 }, {'name': 'B', 'age': 11 } ]
        >>> it(lst).sorted_by(lambda x: x['name']).to_list()
        [{'name': 'A', 'age': 12}, {'name': 'B', 'age': 11}, {'name': 'C', 'age': 10}]
        >>> it(lst).sorted_by(lambda x: x['age']).to_list()
        [{'name': 'C', 'age': 10}, {'name': 'B', 'age': 11}, {'name': 'A', 'age': 12}]
        """
        lst = list(self)
        lst.sort(key=self.__callback_overload_warpper__(key_selector))
        return it(lst)

    def sorted_descending(self) -> Stream[T]:
        """
        Returns a Stream of all elements sorted descending according to their natural sort order.

        Example 1:
        >>> lst = ['b', 'c', 'a']
        >>> it(lst).sorted_descending().to_list()
        ['c', 'b', 'a']
        """
        return self.sorted().reversed()

    @overload
    def sorted_by_descending(
        self, key_selector: Callable[[T], SupportsRichComparisonT]
    ) -> Stream[T]: ...
    @overload
    def sorted_by_descending(
        self, key_selector: Callable[[T, int], SupportsRichComparisonT]
    ) -> Stream[T]: ...
    @overload
    def sorted_by_descending(
        self, key_selector: Callable[[T, int, Stream[T]], SupportsRichComparisonT]
    ) -> Stream[T]: ...
    def sorted_by_descending(
        self, key_selector: Callable[..., SupportsRichComparisonT]
    ) -> Stream[T]:
        """
        Returns a sequence that yields elements of this sequence sorted descending according
        to natural sort order of the value returned by specified [key_selector] function.

        Example 1:
        >>> lst = [ {'name': 'A', 'age': 12 }, {'name': 'C', 'age': 10 }, {'name': 'B', 'age': 11 } ]
        >>> it(lst).sorted_by_descending(lambda x: x['name']).to_list()
        [{'name': 'C', 'age': 10}, {'name': 'B', 'age': 11}, {'name': 'A', 'age': 12}]
        >>> it(lst).sorted_by_descending(lambda x: x['age']).to_list()
        [{'name': 'A', 'age': 12}, {'name': 'B', 'age': 11}, {'name': 'C', 'age': 10}]
        """
        return self.sorted_by(key_selector).reversed()

    # noinspection PyShadowingNames
    def sorted_with(self, comparator: Callable[[T, T], int]) -> Stream[T]:
        """
        Returns a sequence that yields elements of this sequence sorted according to the specified [comparator].

        Example 1:
        >>> lst = ['aa', 'bbb', 'c']
        >>> it(lst).sorted_with(lambda a, b: len(a)-len(b)).to_list()
        ['c', 'aa', 'bbb']
        """
        from functools import cmp_to_key

        lst = list(self)
        lst.sort(key=cmp_to_key(comparator))
        return it(lst)

    @overload
    def associate(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]: ...
    @overload
    def associate(self, transform: Callable[[T, int], Tuple[K, V]]) -> Dict[K, V]: ...
    @overload
    def associate(self, transform: Callable[[T, int, Stream[T]], Tuple[K, V]]) -> Dict[K, V]: ...
    def associate(self, transform: Callable[..., Tuple[K, V]]) -> Dict[K, V]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Stream.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}
        """
        transform = self.__callback_overload_warpper__(transform)
        dic: Dict[K, V] = dict()
        for i in self:
            k, v = transform(i)
            dic[k] = v
        return dic

    @overload
    def associate_by(self, key_selector: Callable[[T], K]) -> Dict[K, T]: ...
    @overload
    def associate_by(self, key_selector: Callable[[T, int], K]) -> Dict[K, T]: ...
    @overload
    def associate_by(self, key_selector: Callable[[T, int, Stream[T]], K]) -> Dict[K, T]: ...
    @overload
    def associate_by(
        self, key_selector: Callable[[T], K], value_transform: Callable[[T], V]
    ) -> Dict[K, V]: ...
    def associate_by(
        self,
        key_selector: Callable[..., K],
        value_transform: Optional[Callable[[T], V]] = None,
    ) -> Union[Dict[K, T], Dict[K, V]]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Stream.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate_by(lambda x: int(x))
        {1: '1', 2: '2', 3: '3'}

        Example 2:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate_by(lambda x: int(x), lambda x: x+x)
        {1: '11', 2: '22', 3: '33'}

        """
        key_selector = self.__callback_overload_warpper__(key_selector)

        dic: Dict[K, Any] = dict()
        for i in self:
            k = key_selector(i)
            dic[k] = i if value_transform is None else value_transform(i)
        return dic

    @overload
    def associate_by_to(
        self, destination: Dict[K, T], key_selector: Callable[[T], K]
    ) -> Dict[K, T]: ...
    @overload
    def associate_by_to(
        self,
        destination: Dict[K, V],
        key_selector: Callable[[T], K],
        value_transform: Callable[[T], V],
    ) -> Dict[K, V]: ...
    def associate_by_to(
        self,
        destination: Dict[K, Any],
        key_selector: Callable[[T], K],
        value_transform: Optional[Callable[[T], Any]] = None,
    ) -> Dict[K, Any]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Stream.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate_by_to({}, lambda x: int(x))
        {1: '1', 2: '2', 3: '3'}

        Example 2:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate_by_to({}, lambda x: int(x), lambda x: x+'!' )
        {1: '1!', 2: '2!', 3: '3!'}

        """
        for i in self:
            k = key_selector(i)
            destination[k] = i if value_transform is None else value_transform(i)
        return destination

    @overload
    def all(self, predicate: Callable[[T], bool]) -> bool: ...
    @overload
    def all(self, predicate: Callable[[T, int], bool]) -> bool: ...
    @overload
    def all(self, predicate: Callable[[T, int, Stream[T]], bool]) -> bool: ...
    def all(self, predicate: Callable[..., bool]) -> bool:
        """
        Returns True if all elements of the Stream satisfy the specified [predicate] function.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).all(lambda x: x > 0)
        True
        >>> it(lst).all(lambda x: x > 1)
        False
        """
        predicate = self.__callback_overload_warpper__(predicate)
        for i in self:
            if not predicate(i):
                return False
        return True

    @overload
    def any(self, predicate: Callable[[T], bool]) -> bool: ...
    @overload
    def any(self, predicate: Callable[[T, int], bool]) -> bool: ...
    @overload
    def any(self, predicate: Callable[[T, int, Stream[T]], bool]) -> bool: ...
    def any(self, predicate: Callable[..., bool]) -> bool:
        """
        Returns True if any elements of the Stream satisfy the specified [predicate] function.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).any(lambda x: x > 0)
        True
        >>> it(lst).any(lambda x: x > 3)
        False
        """
        predicate = self.__callback_overload_warpper__(predicate)
        for i in self:
            if predicate(i):
                return True
        return False

    @overload
    def count(self) -> int: ...
    @overload
    def count(self, predicate: Callable[[T], bool]) -> int: ...
    @overload
    def count(self, predicate: Callable[[T, int], bool]) -> int: ...
    @overload
    def count(self, predicate: Callable[[T, int, Stream[T]], bool]) -> int: ...
    def count(self, predicate: Optional[Callable[..., bool]] = None) -> int:
        """
        Returns the number of elements in the Stream that satisfy the specified [predicate] function.

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

    def contains(self, value: T) -> bool:
        """
        Returns True if the Stream contains the specified [value].

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).contains(1)
        True
        >>> it(lst).contains(4)
        False
        """
        return value in self

    def element_at(self, index: int) -> T:
        """
        Returns the element at the specified [index] in the Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at(1)
        2

        Example 2:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at(3)
        Traceback (most recent call last):
        ...
        IndexError: Index 3 out of range
        """
        return self.element_at_or_else(
            index, lambda index: throw(IndexError(f"Index {index} out of range"))
        )

    @overload
    def element_at_or_else(self, index: int) -> Optional[T]:
        """
        Returns the element at the specified [index] in the Stream or the [default] value if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_else(1, 'default')
        2
        >>> it(lst).element_at_or_else(4, lambda x: 'default')
        'default'
        """
        ...

    @overload
    def element_at_or_else(self, index: int, default: T) -> T: ...
    @overload
    def element_at_or_else(self, index: int, default: Callable[[int], T]) -> T: ...
    def element_at_or_else(
        self, index: int, default: Union[Callable[[int], T], T, None] = None
    ) -> Optional[T]:
        """
        Returns the element at the specified [index] in the Stream or the [default] value if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_else(1, lambda x: 'default')
        2
        >>> it(lst).element_at_or_else(4, lambda x: 'default')
        'default'

        """
        if index >= 0:
            if (
                isinstance(self.__transform__, NonTransform)
                and isinstance(self.__transform__.iter, list)
                and index < len(self.__transform__.iter)
            ):
                return self.__transform__.iter[index]
            for i, e in enumerate(self):
                if i == index:
                    return e
        return default(index) if callable(default) else default  # type: ignore

    def element_at_or_default(self, index: int, default: T) -> T:
        """
        Returns the element at the specified [index] in the Stream or the [default] value if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_default(1, 'default')
        2
        >>> it(lst).element_at_or_default(4, 'default')
        'default'

        """
        return self.element_at_or_else(index, default)

    def element_at_or_none(self, index: int) -> Optional[T]:
        """
        Returns the element at the specified [index] in the Stream or None if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_none(1)
        2
        >>> it(lst).element_at_or_none(4) is None
        True
        """
        return self.element_at_or_else(index)

    def distinct(self) -> Stream[T]:
        """
        Returns a new Stream containing the distinct elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3, 1, 2, 3]
        >>> it(lst).distinct().to_list()
        [1, 2, 3]

        Example 2:
        >>> lst = [(1, 'A'), (1, 'A'), (1, 'A'), (2, 'A'), (3, 'C'), (3, 'D')]
        >>> it(lst).distinct().sorted().to_list()
        [(1, 'A'), (2, 'A'), (3, 'C'), (3, 'D')]

        """
        from .distinct import DistinctTransform

        return it(DistinctTransform(self))

    @overload
    def distinct_by(self, key_selector: Callable[[T], Any]) -> Stream[T]: ...
    @overload
    def distinct_by(self, key_selector: Callable[[T, int], Any]) -> Stream[T]: ...
    @overload
    def distinct_by(self, key_selector: Callable[[T, int, Stream[T]], Any]) -> Stream[T]: ...
    def distinct_by(self, key_selector: Callable[..., Any]) -> Stream[T]:
        """
        Returns a new Stream containing the distinct elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3, 1, 2, 3]
        >>> it(lst).distinct_by(lambda x: x%2).to_list()
        [1, 2]
        """
        from .distinct import DistinctTransform

        return it(DistinctTransform(self, self.__callback_overload_warpper__(key_selector)))

    @overload
    def reduce(self, accumulator: Callable[[T, T], T]) -> T: ...
    @overload
    def reduce(self, accumulator: Callable[[U, T], U], initial: U) -> U: ...
    def reduce(self, accumulator: Callable[..., U], initial: Optional[U] = None) -> Optional[U]:
        """
        Returns the result of applying the specified [accumulator] function to the given Stream's elements.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).reduce(lambda x, y: x+y)
        6
        """
        result: Optional[U] = initial
        for i, e in enumerate(self):
            if i == 0 and initial is None:
                result = e  # type: ignore
                continue

            result = accumulator(result, e)
        return result

    def fold(self, initial: U, accumulator: Callable[[U, T], U]) -> U:
        """
        Returns the result of applying the specified [accumulator] function to the given Stream's elements.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).fold(0, lambda x, y: x+y)
        6
        """
        return self.reduce(accumulator, initial)

    @overload
    def sum_of(self, selector: Callable[[T], int]) -> int: ...
    @overload
    def sum_of(self, selector: Callable[[T], float]) -> float: ...
    def sum_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        """
        Returns the sum of the elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum_of(lambda x: x)
        6
        """
        return sum(selector(i) for i in self)

    @overload
    def max_of(self, selector: Callable[[T], int]) -> int: ...
    @overload
    def max_of(self, selector: Callable[[T], float]) -> float: ...
    def max_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        """
        Returns the maximum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max_of(lambda x: x)
        3
        """
        return max(selector(i) for i in self)

    @overload
    def max_by_or_none(self, selector: Callable[[T], int]) -> Optional[T]: ...
    @overload
    def max_by_or_none(self, selector: Callable[[T], float]) -> Optional[T]: ...
    def max_by_or_none(self, selector: Callable[[T], Union[float, int]]) -> Optional[T]:
        """
        Returns the first element yielding the largest value of the given function
        or `none` if there are no elements.

        Example 1:
        >>> lst = [ { "name": "A", "num": 100 }, { "name": "B", "num": 200 }]
        >>> it(lst).max_by_or_none(lambda x: x["num"])
        {'name': 'B', 'num': 200}

        Example 2:
        >>> lst = []
        >>> it(lst).max_by_or_none(lambda x: x["num"])
        """

        max_item = None
        max_val = None

        for item in self:
            val = selector(item)
            if max_val is None or val > max_val:
                max_item = item
                max_val = val

        return max_item

    @overload
    def max_by(self, selector: Callable[[T], int]) -> T: ...
    @overload
    def max_by(self, selector: Callable[[T], float]) -> T: ...
    def max_by(self, selector: Callable[[T], Union[float, int]]) -> T:
        """
        Returns the first element yielding the largest value of the given function.

        Example 1:
        >>> lst = [ { "name": "A", "num": 100 }, { "name": "B", "num": 200 }]
        >>> it(lst).max_by(lambda x: x["num"])
        {'name': 'B', 'num': 200}

        Exmaple 2:
        >>> lst = []
        >>> it(lst).max_by(lambda x: x["num"])
        Traceback (most recent call last):
        ...
        ValueError: Stream is empty.
        """
        max_item = self.max_by_or_none(selector)
        if max_item is None:
            raise ValueError("Stream is empty.")
        return max_item

    @overload
    def min_of(self, selector: Callable[[T], int]) -> int:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min_of(lambda x: x)
        1
        """
        ...

    @overload
    def min_of(self, selector: Callable[[T], float]) -> float: ...
    def min_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        return min(selector(i) for i in self)

    @overload
    def min_by_or_none(self, selector: Callable[[T], int]) -> Optional[T]: ...
    @overload
    def min_by_or_none(self, selector: Callable[[T], float]) -> Optional[T]: ...
    def min_by_or_none(self, selector: Callable[[T], float]) -> Optional[T]:
        """
        Returns the first element yielding the smallest value of the given function
        or `none` if there are no elements.

        Example 1:
        >>> lst = [ { "name": "A", "num": 100 }, { "name": "B", "num": 200 }]
        >>> it(lst).min_by_or_none(lambda x: x["num"])
        {'name': 'A', 'num': 100}

        Exmaple 2:
        >>> lst = []
        >>> it(lst).min_by_or_none(lambda x: x["num"])
        """
        min_item = None
        min_val = None

        for item in self:
            val = selector(item)
            if min_val is None or val < min_val:
                min_item = item
                min_val = val

        return min_item

    @overload
    def min_by(self, selector: Callable[[T], int]) -> T: ...
    @overload
    def min_by(self, selector: Callable[[T], float]) -> T: ...
    def min_by(self, selector: Callable[[T], float]) -> T:
        """
        Returns the first element yielding the smallest value of the given function.

        Example 1:
        >>> lst = [ { "name": "A", "num": 100 }, { "name": "B", "num": 200 }]
        >>> it(lst).min_by(lambda x: x["num"])
        {'name': 'A', 'num': 100}

        Exmaple 2:
        >>> lst = []
        >>> it(lst).min_by(lambda x: x["num"])
        Traceback (most recent call last):
        ...
        ValueError: Stream is empty.
        """
        min_item = self.min_by_or_none(selector)
        if min_item is None:
            raise ValueError("Stream is empty.")

        return min_item

    def mean_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        """
        Returns the mean of the elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).mean_of(lambda x: x)
        2.0
        """
        return self.sum_of(selector) / len(self)

    @overload
    def sum(self: Stream[int]) -> int:
        """
        Returns the sum of the elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum()
        6
        """
        ...

    @overload
    def sum(self: Stream[float]) -> float: ...
    def sum(self: Union[Stream[int], Stream[float]]) -> Union[float, int]:
        """
        Returns the sum of the elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum()
        6
        """
        return sum(self)

    @overload
    def max(self: Stream[int]) -> int: ...
    @overload
    def max(self: Stream[float]) -> float: ...
    def max(self: Union[Stream[int], Stream[float]]) -> Union[float, int]:
        """
        Returns the maximum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max()
        3
        """
        return max(self)

    @overload
    def max_or_default(self: Stream[int]) -> int: ...
    @overload
    def max_or_default(self: Stream[int], default: V) -> Union[int, V]: ...
    @overload
    def max_or_default(self: Stream[float]) -> float: ...
    @overload
    def max_or_default(self: Stream[float], default: V) -> Union[float, V]: ...
    def max_or_default(
        self: Union[Stream[int], Stream[float]], default: Optional[V] = None
    ) -> Union[float, int, V, None]:
        """
        Returns the maximum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max_or_default()
        3

        Example 2:
        >>> lst = []
        >>> it(lst).max_or_default() is None
        True

        Example 3:
        >>> lst = []
        >>> it(lst).max_or_default(9)
        9
        """
        if self.is_empty():
            return default
        return max(self)

    @overload
    def max_or_none(self: Stream[int]) -> int: ...
    @overload
    def max_or_none(self: Stream[float]) -> float: ...
    def max_or_none(
        self: Union[Stream[int], Stream[float]],
    ) -> Union[float, int, None]:
        """
        Returns the maximum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max_or_none()
        3

        Example 2:
        >>> lst = []
        >>> it(lst).max_or_none() is None
        True
        """
        return self.max_or_default(None)

    @overload
    def min(self: Stream[int]) -> int:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min()
        1
        """
        ...

    @overload
    def min(self: Stream[float]) -> float: ...
    def min(self: Union[Stream[int], Stream[float]]) -> Union[float, int]:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min()
        1
        """
        return min(self)

    @overload
    def min_or_none(self: Stream[int]) -> Optional[int]:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min_or_none()
        1
        """
        ...

    @overload
    def min_or_none(self: Stream[float]) -> Optional[float]: ...
    def min_or_none(
        self: Union[Stream[int], Stream[float]],
    ) -> Union[float, int, None]:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min_or_none()
        1
        """
        return self.min_or_default(None)

    @overload
    def min_or_default(self: Stream[int]) -> int:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min_or_default()
        1
        """
        ...

    @overload
    def min_or_default(self: Stream[int], default: V) -> Union[int, V]: ...
    @overload
    def min_or_default(self: Stream[float]) -> float: ...
    @overload
    def min_or_default(self: Stream[float], default: V) -> Union[float, V]: ...
    def min_or_default(
        self: Union[Stream[int], Stream[float]], default: Optional[V] = None
    ) -> Union[float, int, V, None]:
        """
        Returns the minimum element of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min_or_default()
        1

        Example 2:
        >>> lst = []
        >>> it(lst).min_or_default(9)
        9
        """
        if self.is_empty():
            return default
        return min(self)

    @overload
    def mean(self: Stream[int]) -> float:
        """
        Returns the mean of the elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).mean()
        2.0
        """
        ...

    @overload
    def mean(self: Stream[float]) -> float: ...
    def mean(self: Union[Stream[int], Stream[float]]) -> float:
        """
        Returns the mean of the elements of the given Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).mean()
        2.0
        """
        return self.sum() / len(self)

    # noinspection PyShadowingNames
    def reversed(self) -> Stream[T]:
        """
        Returns a list with elements in reversed order.

        Example 1:
        >>> lst = ['b', 'c', 'a']
        >>> it(lst).reversed().to_list()
        ['a', 'c', 'b']
        """
        lst = list(self)
        lst.reverse()
        return it(lst)

    @overload
    def flat_map(self, transform: Callable[[T], Iterable[U]]) -> Stream[U]: ...
    @overload
    def flat_map(self, transform: Callable[[T, int], Iterable[U]]) -> Stream[U]: ...
    @overload
    def flat_map(self, transform: Callable[[T, int, Stream[T]], Iterable[U]]) -> Stream[U]: ...
    def flat_map(self, transform: Callable[..., Iterable[U]]) -> Stream[U]:
        """
        Returns a single list of all elements yielded from results of [transform]
        function being invoked on each element of original collection.

        Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flat_map(lambda x: x).to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        return self.map(transform).flatten()

    def flatten(self: Iterable[Iterable[U]]) -> Stream[U]:
        """
        Returns a sequence of all elements from all sequences in this sequence.

        Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flatten().to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        from .flattening import FlatteningTransform

        return it(FlatteningTransform(self))

    @overload
    def group_by(self, key_selector: Callable[[T], K]) -> Stream[Grouping[K, T]]: ...
    @overload
    def group_by(self, key_selector: Callable[[T, int], K]) -> Stream[Grouping[K, T]]: ...
    @overload
    def group_by(
        self, key_selector: Callable[[T, int, Stream[T]], K]
    ) -> Stream[Grouping[K, T]]: ...
    def group_by(self, key_selector: Callable[..., K]) -> Stream[Grouping[K, T]]:
        """
        Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

        Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by(lambda x: x%2).map(lambda x: (x.key, x.values.to_list())).to_list()
        [(1, [1, 3, 5]), (0, [2, 4])]
        """
        from .grouping import GroupingTransform

        return it(GroupingTransform(self, self.__callback_overload_warpper__(key_selector)))

    @overload
    def group_by_to(
        self, destination: Dict[K, List[T]], key_selector: Callable[[T], K]
    ) -> Dict[K, List[T]]: ...
    @overload
    def group_by_to(
        self, destination: Dict[K, List[T]], key_selector: Callable[[T, int], K]
    ) -> Dict[K, List[T]]: ...
    @overload
    def group_by_to(
        self,
        destination: Dict[K, List[T]],
        key_selector: Callable[[T, int, Stream[T]], K],
    ) -> Dict[K, List[T]]: ...
    def group_by_to(
        self, destination: Dict[K, List[T]], key_selector: Callable[..., K]
    ) -> Dict[K, List[T]]:
        """
        Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

        Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by_to({}, lambda x: x%2)
        {1: [1, 3, 5], 0: [2, 4]}
        """
        key_selector = self.__callback_overload_warpper__(key_selector)
        for e in self:
            k = key_selector(e)
            if k not in destination:
                destination[k] = []
            destination[k].append(e)
        return destination

    @overload
    def for_each(self, action: Callable[[T], None]) -> None: ...
    @overload
    def for_each(self, action: Callable[[T, int], None]) -> None: ...
    @overload
    def for_each(self, action: Callable[[T, int, Stream[T]], None]) -> None: ...
    def for_each(self, action: Callable[..., None]) -> None:
        """
        Invokes [action] function on each element of the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).for_each(lambda x: print(x))
        a
        b
        c

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).for_each(lambda x, i: print(x, i))
        a 0
        b 1
        c 2
        """
        self.on_each(action)

    @overload
    def parallel_for_each(
        self, action: Callable[[T], None], max_workers: Optional[int] = None
    ) -> None: ...
    @overload
    def parallel_for_each(
        self, action: Callable[[T, int], None], max_workers: Optional[int] = None
    ) -> None: ...
    @overload
    def parallel_for_each(
        self,
        action: Callable[[T, int, Stream[T]], None],
        max_workers: Optional[int] = None,
    ) -> None: ...
    def parallel_for_each(
        self, action: Callable[..., None], max_workers: Optional[int] = None
    ) -> None:
        """
        Invokes [action] function on each element of the given Stream in parallel.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).parallel_for_each(lambda x: print(x))
        a
        b
        c

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).parallel_for_each(lambda x: print(x), max_workers=2)
        a
        b
        c
        """
        self.parallel_on_each(action, max_workers)

    @overload
    def on_each(self, action: Callable[[T], None]) -> Stream[T]: ...
    @overload
    def on_each(self, action: Callable[[T, int], None]) -> Stream[T]: ...
    @overload
    def on_each(self, action: Callable[[T, int, Stream[T]], None]) -> Stream[T]: ...
    def on_each(self, action: Callable[..., None]) -> Stream[T]:
        """
        Invokes [action] function on each element of the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).on_each(lambda x: print(x)) and None
        a
        b
        c

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).on_each(lambda x, i: print(x, i)) and None
        a 0
        b 1
        c 2
        """
        action = self.__callback_overload_warpper__(action)
        for i in self:
            action(i)
        return self

    @overload
    def parallel_on_each(
        self,
        action: Callable[[T], None],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: "ParallelMappingTransform.Executor" = "Thread",
    ) -> Stream[T]: ...
    @overload
    def parallel_on_each(
        self,
        action: Callable[[T, int], None],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: "ParallelMappingTransform.Executor" = "Thread",
    ) -> Stream[T]: ...
    @overload
    def parallel_on_each(
        self,
        action: Callable[[T, int, Stream[T]], None],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: "ParallelMappingTransform.Executor" = "Thread",
    ) -> Stream[T]: ...
    def parallel_on_each(
        self,
        action: Callable[..., None],
        max_workers: Optional[int] = None,
        chunksize: int = 1,
        executor: "ParallelMappingTransform.Executor" = "Thread",
    ) -> Stream[T]:
        """
        Invokes [action] function on each element of the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).parallel_on_each(lambda x: print(x)) and None
        a
        b
        c

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).parallel_on_each(lambda x: print(x), max_workers=2) and None
        a
        b
        c
        """
        from .parallel_mapping import ParallelMappingTransform

        action = self.__callback_overload_warpper__(action)
        for _ in ParallelMappingTransform(self, action, max_workers, chunksize, executor):
            pass
        return self

    @overload
    def zip(self, other: Iterable[U]) -> Stream[Tuple[T, U]]: ...
    @overload
    def zip(self, other: Iterable[U], transform: Callable[[T, U], V]) -> Stream[V]: ...
    def zip(
        self,
        other: Iterable[Any],
        transform: Optional[Callable[..., V]] = None,  # type: ignore
    ) -> Stream[Any]:
        """
        Returns a new Stream of tuples, where each tuple contains two elements.

        Example 1:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> it(lst1).zip(lst2).to_list()
        [('a', 1), ('b', 2), ('c', 3)]

        Example 2:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> it(lst1).zip(lst2, lambda x, y: x + '__' +str( y)).to_list()
        ['a__1', 'b__2', 'c__3']
        """
        if transform is None:

            def transform(*x: Any) -> Tuple[Any, ...]:
                return (*x,)

        from .merging import MergingTransform

        return it(MergingTransform(self, other, transform))

    @overload
    def zip_with_next(self) -> Stream[Tuple[T, T]]: ...
    @overload
    def zip_with_next(self, transform: Callable[[T, T], V]) -> Stream[V]: ...
    def zip_with_next(self, transform: Optional[Callable[[T, T], Any]] = None) -> Stream[Any]:
        """
        Returns a sequence containing the results of applying the given [transform] function
        to an each pair of two adjacent elements in this sequence.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).zip_with_next(lambda x, y: x + '__' + y).to_list()
        ['a__b', 'b__c']

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).zip_with_next().to_list()
        [('a', 'b'), ('b', 'c')]
        """
        from .merging_with_next import MergingWithNextTransform

        return it(MergingWithNextTransform(self, transform or (lambda a, b: (a, b))))

    @overload
    def unzip(self: Stream[Tuple[U, V]]) -> "Tuple[ListLike[U], ListLike[V]]": ...
    @overload
    def unzip(self, transform: Callable[[T], Tuple[U, V]]) -> "Tuple[ListLike[U], ListLike[V]]": ...
    @overload
    def unzip(
        self, transform: Callable[[T, int], Tuple[U, V]]
    ) -> "Tuple[ListLike[U], ListLike[V]]": ...
    @overload
    def unzip(
        self, transform: Callable[[T, int, Stream[T]], Tuple[U, V]]
    ) -> "Tuple[ListLike[U], ListLike[V]]": ...
    def unzip(  # type: ignore
        self: Stream[Tuple[U, V]],
        transform: Union[Optional[Callable[..., Tuple[Any, Any]]], bool] = None,
    ) -> "Tuple[ListLike[U], ListLike[V]]":
        """
        Returns a pair of lists, where first list is built from the first values of each pair from this array, second list is built from the second values of each pair from this array.

        Example 1:
        >>> lst = [{'name': 'a', 'age': 11}, {'name': 'b', 'age': 12}, {'name': 'c', 'age': 13}]
        >>> a, b = it(lst).unzip(lambda x: (x['name'], x['age']))
        >>> a
        ['a', 'b', 'c']
        >>> b
        [11, 12, 13]

        Example 1:
        >>> lst = [('a', 11), ('b', 12), ('c', 13)]
        >>> a, b = it(lst).unzip()
        >>> a
        ['a', 'b', 'c']
        >>> b
        [11, 12, 13]
        """
        from .list_like import ListLike

        it = self
        if isinstance(transform, bool):
            transform = None

        if transform is not None:
            transform = self.__callback_overload_warpper__(transform)
            it = it.map(transform)

        a = it.map(lambda x: x[0])  # type: ignore
        b = it.map(lambda x: x[1])  # type: ignore

        return ListLike(a), ListLike(b)

    def with_index(self) -> Stream[IndexedValue[T]]:
        """
        Returns a sequence containing the elements of this sequence and their indexes.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).with_index().to_list()
        [IndexedValue(0, a), IndexedValue(1, b), IndexedValue(2, c)]
        """
        return self.indexed()

    @overload
    def shuffled(self) -> Stream[T]: ...
    @overload
    def shuffled(self, seed: Union[int, float, str, bytes, bytearray, None]) -> Stream[T]: ...
    @overload
    def shuffled(self, random: "Random") -> Stream[T]: ...
    def shuffled(  # type: ignore
        self, seed: Union["Random", int, float, str, bytes, bytearray, None] = None
    ) -> Stream[T]:
        """
        Returns a sequence that yields elements of this sequence randomly shuffled
        using the specified [random] instance as the source of randomness.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).shuffled('123').to_list()
        ['b', 'a', 'c']

        Example 2:
        >>> from random import Random
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).shuffled(Random('123')).to_list()
        ['b', 'a', 'c']

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).shuffled(123).to_list()
        ['c', 'b', 'a']
        """
        from .shuffling import ShufflingTransform

        return it(ShufflingTransform(self, seed))

    @overload
    def partition(self, predicate: Callable[[T], bool]) -> "Tuple[ListLike[T], ListLike[T]]": ...
    @overload
    def partition(
        self, predicate: Callable[[T, int], bool]
    ) -> "Tuple[ListLike[T], ListLike[T]]": ...
    @overload
    def partition(
        self, predicate: Callable[[T, int, Stream[T]], bool]
    ) -> "Tuple[ListLike[T], ListLike[T]]": ...
    def partition(self, predicate: Callable[..., bool]) -> "Tuple[ListLike[T], ListLike[T]]":
        """
        Partitions the elements of the given Stream into two groups,
        the first group containing the elements for which the predicate returns true,
        and the second containing the rest.

        Example 1:
        >>> lst = ['a', 'b', 'c', '2']
        >>> it(lst).partition(lambda x: x.isalpha())
        (['a', 'b', 'c'], ['2'])

        Example 2:
        >>> lst = ['a', 'b', 'c', '2']
        >>> it(lst).partition(lambda _, i: i % 2 == 0)
        (['a', 'c'], ['b', '2'])
        """
        from .list_like import ListLike

        predicate_a = self.__callback_overload_warpper__(predicate)
        predicate_b = self.__callback_overload_warpper__(predicate)
        part_a = self.filter(predicate_a)
        part_b = self.filter(lambda x: not predicate_b(x))
        return ListLike(part_a), ListLike(part_b)

    def indexed(self) -> Stream[IndexedValue[T]]:
        return self.map(lambda x, i: IndexedValue(x, i))

    @overload
    def combinations(self, n: Literal[2]) -> Stream[Tuple[T, T]]: ...
    @overload
    def combinations(self, n: Literal[3]) -> Stream[Tuple[T, T, T]]: ...
    @overload
    def combinations(self, n: Literal[4]) -> Stream[Tuple[T, T, T, T]]: ...
    @overload
    def combinations(self, n: Literal[5]) -> Stream[Tuple[T, T, T, T, T]]: ...
    def combinations(self, n: int) -> Stream[Tuple[T, ...]]:
        """
        Returns a Stream of all possible combinations of size [n] from the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).combinations(2).to_list()
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        """
        from .combination import CombinationTransform

        return it(CombinationTransform(self, n))

    def nth(self, n: int) -> T:
        """
        Returns the nth element of the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).nth(2)
        'c'
        """
        return self.skip(n).first()

    def windowed(self, size: int, step: int = 1, partialWindows: bool = False) -> Stream[List[T]]:
        """
         Returns a Stream of all possible sliding windows of size [size] from the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).windowed(3).to_list()
        [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]

        Example 2:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).windowed(3, 2).to_list()
        [['a', 'b', 'c'], ['c', 'd', 'e']]

        Example 3:
        >>> lst = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> it(lst).windowed(3, 2, True).to_list()
        [['a', 'b', 'c'], ['c', 'd', 'e'], ['e', 'f']]
        """
        from .windowed import WindowedTransform

        return it(WindowedTransform(self, size, step, partialWindows))

    def chunked(self, size: int) -> Stream[List[T]]:
        """
        Returns a Stream of all possible chunks of size [size] from the given Stream.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).chunked(3).to_list()
        [['a', 'b', 'c'], ['d', 'e']]


        Example 2:
        >>> lst = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> it(lst).chunked(3).to_list()
        [['a', 'b', 'c'], ['d', 'e', 'f']]
        """
        return self.windowed(size, size, True)

    def repeat(self, n: int) -> Stream[T]:
        """
        Returns a Stream containing this sequence repeated n times.

        Example 1:
        >>> lst = ['a', 'b']
        >>> it(lst).repeat(3).to_list()
        ['a', 'b', 'a', 'b', 'a', 'b']
        """
        from .concat import ConcatTransform

        return it(ConcatTransform([self] * n))

    def concat(self, *other: Iterable[T]) -> Stream[T]:
        """
        Returns a Stream of all elements of the given Stream, followed by all elements of the given Stream.

        Example 1:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> it(lst1).concat(lst2).to_list()
        ['a', 'b', 'c', 1, 2, 3]

        Example 2:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> lst3 = [4, 5, 6]
        >>> it(lst1).concat(lst2, lst3).to_list()
        ['a', 'b', 'c', 1, 2, 3, 4, 5, 6]
        """
        from .concat import ConcatTransform

        return it(ConcatTransform([self, *other]))

    def intersect(self, *other: Iterable[T]) -> Stream[T]:
        """
        Returns a set containing all elements that are contained by both this collection and the specified collection.

        The returned set preserves the element iteration order of the original collection.

        To get a set containing all elements that are contained at least in one of these collections use union.

        Example 1:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = ['a2', 'b2', 'c']
        >>> it(lst1).intersect(lst2).to_list()
        ['c']

        Example 2:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = ['a2', 'b', 'c']
        >>> lst3 = ['a3', 'b', 'c3']
        >>> it(lst1).intersect(lst2, lst3).to_list()
        ['b']


        Example 1:
        >>> lst1 = ['a', 'a', 'c']
        >>> lst2 = ['a2', 'b2', 'a']
        >>> it(lst1).intersect(lst2).to_list()
        ['a']
        """
        from .intersection import IntersectionTransform

        return it(IntersectionTransform([self, *other]))

    def union(self, *other: Stream[T]) -> Stream[T]:
        """
        Returns a set containing all distinct elements from both collections.

        The returned set preserves the element iteration order of the original collection. Those elements of the other collection that are unique are iterated in the end in the order of the other collection.

        To get a set containing all elements that are contained in both collections use intersect.

        Example 1:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = ['a2', 'b2', 'c']
        >>> it(lst1).union(lst2).to_list()
        ['a', 'b', 'c', 'a2', 'b2']

        Example 2:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = ['a2', 'b', 'c']
        >>> lst3 = ['a3', 'b', 'c3']
        >>> it(lst1).union(lst2, lst3).to_list()
        ['a', 'b', 'c', 'a2', 'a3', 'c3']


        Example 1:
        >>> lst1 = ['a', 'a', 'c']
        >>> lst2 = ['a2', 'b2', 'a']
        >>> it(lst1).union(lst2).to_list()
        ['a', 'c', 'a2', 'b2']
        """
        return self.concat(*other).distinct()

    def join(self: Stream[str], separator: str = " ") -> str:
        """
        Joins the elements of the given Stream into a string.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).join(', ')
        'a, b, c'
        """
        return separator.join(self)

    @overload
    def progress(self) -> Stream[T]: ...
    @overload
    def progress(
        self, progress_func: Union[Literal["tqdm"], Literal["tqdm_rich"]]
    ) -> Stream[T]: ...
    @overload
    def progress(self, progress_func: Callable[[Stream[T]], Iterable[T]]) -> Stream[T]: ...
    def progress(
        self,
        progress_func: Union[
            Callable[[Stream[T]], Iterable[T]],
            Literal["tqdm"],
            Literal["tqdm_rich"],
            None,
        ] = None,
    ) -> Stream[T]:
        """
        Returns a Stream that enable a progress bar for the given Stream.

        Example 1:
        >>> from tqdm import tqdm
        >>> from time import sleep
        >>> it(range(10)).progress(lambda x: tqdm(x, total=len(x))).parallel_map(lambda x: sleep(0.), max_workers=5).to_list() and None
        >>> for _ in it(list(range(10))).progress(lambda x: tqdm(x, total=len(x))).to_list(): pass
        """
        if progress_func is not None and callable(progress_func):
            return it(progress_func(self))

        def import_tqdm():
            if progress_func == "tqdm_rich":
                import warnings
                from tqdm.rich import tqdm
                from tqdm import TqdmExperimentalWarning

                warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
            else:
                from tqdm import tqdm
            return tqdm

        try:
            tqdm = import_tqdm()
        except ImportError:
            from pip import main as pip  # type: ignore

            pip(["install", "tqdm"])
            tqdm = import_tqdm()

        return it(tqdm(self, total=len(self)))

    def typing_as(self, typ: Type[U]) -> Stream[U]:
        """
        Cast the element as specific Type to gain code completion base on type annotations.
        """
        el = self.first_not_none_of_or_none()
        if el is None or isinstance(el, typ) or not isinstance(el, dict):
            return self  # type: ignore

        class AttrDict(Dict[str, Any]):
            def __init__(self, value: Dict[str, Any]) -> None:
                super().__init__(**value)
                setattr(self, "__dict__", value)
                self.__getattr__ = value.__getitem__
                self.__setattr__ = value.__setattr__  # type: ignore

        return self.map(AttrDict)  # type: ignore  # use https://github.com/cdgriffith/Box ?

    def to_set(self) -> Set[T]:
        """
        Returns a set containing all elements of this Stream.

        Example 1:
        >>> it(['a', 'b', 'c', 'c']).to_set() == {'a', 'b', 'c'}
        True
        """
        return set(self)

    @overload
    def to_dict(self: Stream[Tuple[K, V]]) -> Dict[K, V]: ...
    @overload
    def to_dict(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]: ...
    @overload
    def to_dict(self, transform: Callable[[T, int], Tuple[K, V]]) -> Dict[K, V]: ...
    @overload
    def to_dict(self, transform: Callable[[T, int, Stream[T]], Tuple[K, V]]) -> Dict[K, V]: ...
    def to_dict(self, transform: Optional[Callable[..., Tuple[K, V]]] = None) -> Dict[K, V]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Stream.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).to_dict(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}

        Example 2:
        >>> lst = [(1, '1'), (2, '2'), (3, '3')]
        >>> it(lst).to_dict()
        {1: '1', 2: '2', 3: '3'}
        """
        return self.associate(transform or (lambda x: x))  # type: ignore

    def to_list(self) -> List[T]:
        """
        Returns a list with elements of the given Stream.

        Example 1:
        >>> it(['b', 'c', 'a']).to_list()
        ['b', 'c', 'a']
        """
        if self.__transform__.cache is not None:
            return self.__transform__.cache.copy()
        return [s for s in self]

    async def to_list_async(self: Iterable[Awaitable[T]]) -> List[T]:
        """
        Returns a list with elements of the given Stream.

        Example 1:
        >>> it(['b', 'c', 'a']).to_list()
        ['b', 'c', 'a']
        """
        from asyncio import gather

        return await gather(*self)  # type: ignore

    def let(self, block: Callable[[Stream[T]], U]) -> U:
        """
        Calls the specified function [block] with `self` value as its argument and returns its result.

        Example 1:
        >>> it(['a', 'b', 'c']).let(lambda x: x.map(lambda y: y + '!')).to_list()
        ['a!', 'b!', 'c!']
        """
        return block(self)

    def also(self, block: Callable[[Stream[T]], Any]) -> Stream[T]:
        """
        Calls the specified function [block] with `self` value as its argument and returns `self` value.

        Example 1:
        >>> it(['a', 'b', 'c']).also(lambda x: x.map(lambda y: y + '!')).to_list()
        ['a', 'b', 'c']
        """
        block(self)
        return self

    @property
    def size(self) -> int:
        """
        Returns the size of the given Stream.
        """
        return len(self.data)

    def is_empty(self) -> bool:
        """
        Returns True if the Stream is empty, False otherwise.

        Example 1:
        >>> it(['a', 'b', 'c']).is_empty()
        False

        Example 2:
        >>> it([None]).is_empty()
        False

        Example 3:
        >>> it([]).is_empty()
        True
        """
        return id(self.first_or_default(self)) == id(self)

    def __iter__(self) -> Iterator[T]:
        return self.__do_iter__()

    def iter(self) -> Iterator[T]:
        return self.__do_iter__()

    def __do_iter__(self) -> Iterator[T]:
        yield from self.__transform__

    def __len__(self) -> int:
        return len(self.__transform__)

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        if self.__transform__.cache is None:
            return "[...]"
        return repr(self.to_list())

    def __getitem__(self, key: int) -> T:
        """
        Returns the element at the specified [index] in the Stream.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst)[1]
        2

        Example 2:
        >>> lst = [1, 2, 3]
        >>> it(lst)[3]
        Traceback (most recent call last):
        ...
        IndexError: Index 3 out of range
        """
        return self.element_at(key)

    @overload
    def __callback_overload_warpper__(self, callback: Callable[[T], U]) -> Callable[[T], U]: ...
    @overload
    def __callback_overload_warpper__(
        self, callback: Callable[[T, int], U]
    ) -> Callable[[T], U]: ...
    @overload
    def __callback_overload_warpper__(
        self, callback: Callable[[T, int, Stream[T]], U]
    ) -> Callable[[T], U]: ...
    def __callback_overload_warpper__(self, callback: Callable[..., U]) -> Callable[[T], U]:
        if hasattr(callback, "__code__"):
            if callback.__code__.co_argcount == 2:
                index = AutoIncrementIndex()
                return lambda x: callback(x, index())
            if callback.__code__.co_argcount == 3:
                index = AutoIncrementIndex()
                return lambda x: callback(x, index(), self)
        return callback


class AutoIncrementIndex:
    idx = 0

    def __call__(self) -> int:
        val = self.idx
        self.idx += 1
        return val


class IndexedValue(NamedTuple, Generic[T]):
    val: T
    idx: int

    def __repr__(self) -> str:
        return f"IndexedValue({self.idx}, {self.val})"


def throw(exception: Exception) -> Any:
    raise exception


def none_or(value: Optional[T], default: T) -> T:
    return value if value is not None else default


def none_or_else(value: Optional[T], f: Callable[[], T]) -> T:
    return value if value is not None else f()


def is_debugging() -> bool:
    from inspect import currentframe
    from traceback import walk_stack

    return (
        it(walk_stack(currentframe()))
        .take(20)
        .map(lambda s: s[0])
        .any(
            lambda s: (
                s.f_code.co_name == "get_contents_debug_adapter_protocol"
                and "pydevd_resolver.py" in s.f_code.co_filename
            )
        )
    )


class StreamProducer:
    @overload
    def __call__(self, elements: List[T]) -> Stream[T]: ...
    @overload
    def __call__(self, elements: Iterable[T]) -> Stream[T]: ...
    @overload
    def __call__(self, *elements: T) -> Stream[T]: ...
    def __call__(self, *iterable: Union[Iterable[T], List[T], T]) -> Stream[T]:  # type: ignore
        if len(iterable) == 1:
            iter = iterable[0]
            if isinstance(iter, Stream):
                return iter  # type: ignore
            if isinstance(iter, Iterable) and not isinstance(iter, str):
                return Stream(iter)  # type: ignore
        return Stream(iterable)  # type: ignore

    def json(self, filepath: str, **kwargs: Dict[str, Any]) -> Stream[Any]:
        """
        Reads and parses the input of a json file.
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f, **kwargs)  # type: ignore
            return self(data)

    def csv(self, filepath: str) -> Stream[List[str]] | Stream[Dict[str, str]]:
        """
        Reads and parses the input of a csv file.
        """
        return self.read_csv(filepath)

    def read_csv(
        self, filepath: str, header: Optional[int] = 0
    ) -> Stream[List[str]] | Stream[Dict[str, str]]:
        """
        Reads and parses the input of a csv file.

        Example 1:
        >>> it.read_csv('tests/data/a.csv').to_list()
        [{'a': 'a1', 'b': '1'}, {'a': 'a2', 'b': '2'}]
        """
        import csv

        it = self
        with open(filepath) as f:
            reader = csv.reader(f)
            iter = it(*reader)
            if header is None or header < 0:
                return iter

            headers = iter.element_at_or_none(header)
            if headers is not None:
                if header == 0:
                    iter = iter.skip(1)
                else:
                    iter = iter.filter(lambda _, i: i != header)

                return iter.map(
                    lambda row: it(row).associate_by(
                        lambda _, ordinal: (
                            headers[ordinal] if ordinal < len(headers) else f"undefined_{ordinal}"
                        )
                    )
                )
            return iter

    def __repr__(self) -> str:
        return __package__ or self.__class__.__name__


it: StreamProducer = StreamProducer()
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> it('hello', 'world').map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 2:
>>> it(['hello', 'world']).map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 3:
>>> it(range(10)).map(lambda x: x*x).to_list()
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""


if __name__ == "__main__":
    import doctest

    doctest.testmod()
