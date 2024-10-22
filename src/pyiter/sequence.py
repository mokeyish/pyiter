from __future__ import annotations
from typing import (
    overload, Any, List, Set, Dict, Deque, DefaultDict, Generic, Iterable, Iterator, Union, 
    Optional, Tuple, Type, TypeVar, Callable, Literal, NamedTuple, Awaitable,
    TYPE_CHECKING
)

from typing_extensions import deprecated

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT
    from random import Random

import sys
if sys.version_info < (3, 11):
    # Generic NamedTuple
    origin__namedtuple_mro_entries =  NamedTuple.__mro_entries__ # type: ignore
    NamedTuple.__mro_entries__ = lambda bases: origin__namedtuple_mro_entries(bases[:1]) # type: ignore


T = TypeVar("T")
R = TypeVar("R")

K = TypeVar("K")
V = TypeVar("V")

IterableS = TypeVar("IterableS", bound=Iterable[Any])


class Sequence(Generic[T], Iterable[T]):
    """
    Given an [iterator] function constructs a [Sequence] that returns values through the [Iterator]
    provided by that function.
    
    The values are evaluated lazily, and the sequence is potentially infinite.
    """
    _iter: SequenceTransform[Iterable[T], T]

    def __init__(self, iterable: Iterable[T]) -> None:
        super().__init__()
        self._iter = SequenceTransform(iterable)

    def dedup(self) -> Sequence[T]:
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
    def dedup_by(self, key_selector: Callable[[T], Any]) -> Sequence[T]:
        ...
    @overload
    def dedup_by(self, key_selector: Callable[[T, int], Any]) -> Sequence[T]:
        ...
    @overload
    def dedup_by(self, key_selector: Callable[[T, int, Sequence[T]], Any]) -> Sequence[T]:
        ...
    def dedup_by(self, key_selector: Callable[..., Any]) -> Sequence[T]:
        """
        Removes all but the first of consecutive elements in the sequence that resolve to the same key.
        """
        return self.dedup_into_group_by(key_selector).map(lambda x: x[0])

    @overload
    def dedup_with_count_by(self, key_selector: Callable[[T], Any]) -> Sequence[Tuple[T, int]]:
        ...
    @overload
    def dedup_with_count_by(self, key_selector: Callable[[T, int], Any]) -> Sequence[Tuple[T, int]]:
        ...
    @overload
    def dedup_with_count_by(self, key_selector: Callable[[T, int, Sequence[T]], Any]) -> Sequence[Tuple[T, int]]:
        ...
    def dedup_with_count_by(self, key_selector: Callable[..., Any]) -> Sequence[Tuple[T, int]]:
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
    def dedup_into_group_by(self, key_selector: Callable[[T], Any]) -> Sequence[List[T]]:
        ...
    @overload
    def dedup_into_group_by(self, key_selector: Callable[[T, int], Any]) -> Sequence[List[T]]:
        ...
    @overload
    def dedup_into_group_by(self, key_selector: Callable[[T, int, Sequence[T]], Any]) -> Sequence[List[T]]:
        ...
    def dedup_into_group_by(self, key_selector: Callable[..., Any]) -> Sequence[List[T]]:
        return DedupTransform(self, key_selector).as_sequence()

    @overload
    def filter(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def filter(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def filter(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def filter(self, predicate: Callable[..., bool]) -> Sequence[T]:
        """
        Returns a Sequence containing only elements matching the given [predicate].

        Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter(lambda x: x.startswith('a')).to_list()
        ['a1', 'a2']

        Example 2:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter(lambda x, i: x.startswith('a') or i % 2 == 0 ).to_list()
        ['a1', 'b2', 'a2']
        """
        return FilteringTransform(self, self._callback_overload_warpper(predicate)).as_sequence()

    @deprecated("use `.filter(lambda x, idx: ... )` instead.", category=None)
    def filter_indexed(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        return self.filter(predicate)

    def filter_is_instance(self, typ: Type[R]) -> Sequence[R]:
        """
        Returns a Sequence containing all elements that are instances of specified type parameter typ.

        Example 1:
        >>> lst = [ 'a1', 1, 'b2', 3]
        >>> it(lst).filter_is_instance(int).to_list()
        [1, 3]

        """
        return self.filter(lambda x: isinstance(x, typ)) # type: ignore

    @overload
    def filter_not(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def filter_not(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def filter_not(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def filter_not(self, predicate: Callable[..., bool]) -> Sequence[T]:
        """
        Returns a Sequence containing all elements not matching the given [predicate].

        Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter_not(lambda x: x.startswith('a')).to_list()
        ['b1', 'b2']

        Example 2:
        >>> lst = [ 'a1', 'a2', 'b1', 'b2']
        >>> it(lst).filter_not(lambda x, i: x.startswith('a') and i % 2 == 0 ).to_list()
        ['a2', 'b1', 'b2']
        """
        predicate = self._callback_overload_warpper(predicate)
        return self.filter(lambda x: not predicate(x))

    @overload
    def filter_not_none(self: Sequence[Optional[R]]) -> Sequence[R]:
        ...
    @overload
    def filter_not_none(self: Sequence[T]) -> Sequence[T]:
        ...
    def filter_not_none(self: Sequence[Optional[R]]) -> Sequence[R]:
        """
        Returns a Sequence containing all elements that are not `None`.

        Example 1:
        >>> lst = [ 'a', None, 'b']
        >>> it(lst).filter_not_none().to_list()
        ['a', 'b']
        """
        return self.filter(lambda x: x is not None) # type: ignore
    
    @overload
    def map(self, transform: Callable[[T], R]) -> Sequence[R]:
        ...
    @overload
    def map(self, transform: Callable[[T, int], R]) -> Sequence[R]:
        ...
    @overload
    def map(self, transform: Callable[[T, int, Sequence[T]], R]) -> Sequence[R]:
        ...
    def map(self, transform: Callable[..., R]) -> Sequence[R]:
        """
        Returns a Sequence containing the results of applying the given [transform] function
        to each element in the original Sequence.

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
        return MappingTransform(self, self._callback_overload_warpper(transform)).as_sequence()
    
    @deprecated("use `.map(lambda x, idx: ... )` instead.", category=None)
    def map_indexed(self, transform: Callable[[T, int], R]) -> Sequence[R]:
        return self.map(transform)
    

    @overload
    async def map_async(self, transform: Callable[[T], Awaitable[R]]) -> Sequence[R]:
        ...
    @overload
    async def map_async(self, transform: Callable[[T, int], Awaitable[R]], return_exceptions: Literal[True]) -> Sequence[Union[R, BaseException]]:
        ...
    @overload
    async def map_async(self, transform: Callable[[T, int, Sequence[T]], Awaitable[R]], return_exceptions: Literal[False] = False) -> Sequence[R]:
        ...
    async def map_async(self, transform: Callable[..., Awaitable[R]], return_exceptions: bool = False):
        """
        Similar to `.map()` but you can input a async transform then await it.
        """
        from asyncio import gather
        if return_exceptions:
            return it(await gather(*self.map(transform), return_exceptions=True))
        return it(await gather(*self.map(transform)))

    @overload
    def map_not_none(self, transform: Callable[[T], Optional[R]]) -> Sequence[R]:
        ...
    @overload
    def map_not_none(self, transform: Callable[[T, int], Optional[R]]) -> Sequence[R]:
        ...
    @overload
    def map_not_none(self, transform: Callable[[T, int, Sequence[T]], Optional[R]]) -> Sequence[R]:
        ...
    def map_not_none(self, transform: Callable[..., Optional[R]]) -> Sequence[R]:
        """
        Returns a Sequence containing only the non-none results of applying the given [transform] function
        to each element in the original collection.

        Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': None}]
        >>> it(lst).map_not_none(lambda x: x['age']).to_list()
        [12]
        """
        return self.map(transform).filter_not_none() # type: ignore
    
    @overload
    def parallel_map(self, transform: Callable[[T], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread') -> Sequence[R]:
        ...
    @overload
    def parallel_map(self, transform: Callable[[T, int], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread') -> Sequence[R]:
        ...
    @overload
    def parallel_map(self, transform: Callable[[T, int, Sequence[T]], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread') -> Sequence[R]:
        ...
    def parallel_map(self, transform: Callable[..., R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread') -> Sequence[R]:
        """
        Returns a Sequence containing the results of applying the given [transform] function
        to each element in the original Sequence.

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
        return ParallelMappingTransform(self, self._callback_overload_warpper(transform), max_workers, chunksize, executor).as_sequence()

    @overload
    def find(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...
    @overload
    def find(self, predicate: Callable[[T, int], bool]) -> Optional[T]:
        ...
    @overload
    def find(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[T]:
        ...
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
    def first(self) -> T:
        ...
    @overload
    def first(self, predicate: Callable[[T], bool]) -> T:
        ...
    @overload
    def first(self, predicate: Callable[[T, int], bool]) -> T:
        ...
    @overload
    def first(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> T:
        ...
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
        ValueError: Sequence is empty.

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first(lambda x: x == 'b')
        'b'

        Example 4:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).first(lambda x: x == 'd')
        Traceback (most recent call last):
        ...
        ValueError: Sequence is empty.

        Example 5:
        >>> lst = [None]
        >>> it(lst).first() is None
        True
        """
        for e in self:
            if predicate is None or predicate(e):
                return e
        raise ValueError("Sequence is empty.")

    @overload
    def first_not_none_of(self: Sequence[Optional[R]]) -> R:
        ...
    @overload
    def first_not_none_of(self: Sequence[Optional[R]], transform: Callable[[Optional[R]], Optional[R]]) -> R:
        ...
    @overload
    def first_not_none_of(self: Sequence[Optional[R]], transform: Callable[[Optional[R], int], Optional[R]]) -> R:
        ...
    @overload
    def first_not_none_of(self: Sequence[Optional[R]], transform: Callable[[Optional[R], int, Sequence[Optional[R]]], Optional[R]]) -> R:
        ...
    def first_not_none_of(self: Sequence[Optional[R]], transform: Optional[Callable[..., Optional[R]]] = None) -> R:
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
        ValueError: No element of the Sequence was transformed to a non-none value.
        """

        v = self.first_not_none_of_or_none() if transform is None else self.first_not_none_of_or_none(transform)
        if v is None:
            raise ValueError('No element of the Sequence was transformed to a non-none value.')
        return v
    
    @overload
    def first_not_none_of_or_none(self) -> T:
        ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T], T]) -> T:
        ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T, int], T]) -> T:
        ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T, int, Sequence[T]], T]) -> T:
        ...
    def first_not_none_of_or_none(self, transform: Optional[Callable[..., T]] = None) -> T:
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
    def first_or_none(self) -> T:
        ...
    @overload
    def first_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...
    @overload
    def first_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[T]:
        ...
    @overload
    def first_or_none(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[T]:
        ...
    def first_or_none(self, predicate: Optional[Callable[..., bool]] = None) -> Optional[T]:
        """
        Returns the first element, or `None` if the Sequence is empty.

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
        return next(iter(self if predicate is None else self.filter(predicate)), None)


    @overload
    def first_or_default(self, default: T) -> T :
        ...
    @overload
    def first_or_default(self, predicate: Callable[[T], bool], default: T) -> T :
        ...
    @overload
    def first_or_default(self, predicate: Callable[[T, int], bool], default: T) -> T :
        ...
    @overload
    def first_or_default(self, predicate: Callable[[T, int, Sequence[T]], bool], default: T) -> T :
        ...
    def first_or_default(self, predicate: Union[Callable[..., bool], T], default: Optional[T] = None) -> T: # type: ignore
        """
        Returns the first element, or the given [default] if the Sequence is empty.

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
        if isinstance(predicate, Callable):
            return none_or(self.first_or_none(predicate), default) # type: ignore
        else:
            default = predicate
        return none_or(self.first_or_none(), default)

    @overload
    def last(self) -> T:
        ...
    @overload
    def last(self, predicate: Callable[[T], bool]) -> T:
        ...
    @overload
    def last(self, predicate: Callable[[T, int], bool]) -> T:
        ...
    @overload
    def last(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> T:
        ...
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
        ValueError: Sequence is empty.
        """
        v = self.last_or_none(predicate) if predicate is not None else self.last_or_none()
        if v is None:
            raise ValueError('Sequence is empty.')
        return v

    @overload
    def last_or_none(self) -> Optional[T]:
        ...
    @overload
    def last_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...
    @overload
    def last_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[T]:
        ...
    @overload
    def last_or_none(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[T]:
        ...
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
        for i, x in enumerate(self._iter):
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
    def index_of_first_or_none(self, predicate: Callable[[T], bool]) -> Optional[int]:
        ...
    @overload
    def index_of_first_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[int]:
        ...
    @overload
    def index_of_first_or_none(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[int]:
        ...
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
        predicate = self._callback_overload_warpper(predicate)
        for i, x in enumerate(self._iter):
            if predicate(x):
                return i
        return None

    @overload
    def index_of_first(self, predicate: Callable[[T], bool]) -> int:
        ...
    @overload
    def index_of_first(self, predicate: Callable[[T, int], bool]) -> int:
        ...
    @overload
    def index_of_first(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> int:
        ...
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
    def index_of_first_or(self, predicate: Callable[[T], bool], default: int) -> int:
        ...
    @overload
    def index_of_first_or(self, predicate: Callable[[T, int], bool], default: int) -> int:
        ...
    @overload
    def index_of_first_or(self, predicate: Callable[[T, int, Sequence[T]], bool], default: int) -> int:
        ...
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
    def index_of_first_or_else(self, predicate: Callable[[T], bool], f: Callable[[], int]) -> int:
        ...
    @overload
    def index_of_first_or_else(self, predicate: Callable[[T, int], bool], f: Callable[[], int]) -> int:
        ...
    @overload
    def index_of_first_or_else(self, predicate: Callable[[T, int, Sequence[T]], bool], f: Callable[[], int]) -> int:
        ...
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
    def index_of_last_or_none(self, predicate: Callable[[T], bool]) -> Optional[int]:
        ...
    @overload
    def index_of_last_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[int]:
        ...
    @overload
    def index_of_last_or_none(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[int]:
        ...
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
        predicate = self._callback_overload_warpper(predicate)
        for i, x in enumerate(seq):
            if predicate(x):
                return last_idx - i
        return None

    @overload
    def index_of_last(self, predicate: Callable[[T], bool]) -> int:
        ...
    @overload
    def index_of_last(self, predicate: Callable[[T, int], bool]) -> int:
        ...
    @overload
    def index_of_last(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> int:
        ...
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
    def index_of_last_or(self, predicate: Callable[[T], bool], default: int) -> int:
        ...
    @overload
    def index_of_last_or(self, predicate: Callable[[T, int], bool], default: int) -> int:
        ...
    @overload
    def index_of_last_or(self, predicate: Callable[[T, int, Sequence[T]], bool], default: int) -> int:
        ...
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
    def index_of_last_o_else(self, predicate: Callable[[T], bool], f: Callable[[], int]) -> int:
        ...
    @overload
    def index_of_last_o_else(self, predicate: Callable[[T, int], bool], f: Callable[[], int]) -> int:
        ...
    @overload
    def index_of_last_o_else(self, predicate: Callable[[T, int, Sequence[T]], bool], f: Callable[[], int]) -> int:
        ...
    def index_of_last_o_else(self, predicate: Callable[..., bool], f: Callable[[], int]) -> int:
        """
        Returns last index of element matching the given [predicate], or default value if no such element was found.

        Example 1:
        >>> lst = ['a', 'b', 'c', 'b']
        >>> it(lst).index_of_last_o_else(lambda x: x == 'b', lambda: -len(lst))
        3

        Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_o_else(lambda x: x == 'd', lambda: -len(lst))
        -3

        Example 3:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).index_of_last_o_else(lambda x: x == 'a', lambda: -len(lst))
        0
        """
        return none_or_else(self.index_of_last_or_none(predicate), f)

    @overload
    def single(self) -> T:
        ...
    @overload
    def single(self, predicate: Callable[[T], bool]) -> T:
        ...
    @overload
    def single(self, predicate: Callable[[T, int], bool]) -> T:
        ...
    @overload
    def single(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> T:
        ...
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
        ValueError: Sequence contains no element matching the predicate.

        Exmaple 2:
        >>> lst = ['a', 'b']
        >>> it(lst).single() is None
        Traceback (most recent call last):
        ...
        ValueError: Sequence contains more than one matching element.
        """
        single: Optional[T] = None
        found = False
        for i in self if predicate is None else self.filter(predicate):
            if found:
                raise ValueError('Sequence contains more than one matching element.')
            single = i
            found = True
        if single is None:
            raise ValueError('Sequence contains no element matching the predicate.')
        return single

    @overload
    def single_or_none(self) -> Optional[T]:
        ...
    @overload
    def single_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...
    @overload
    def single_or_none(self, predicate: Callable[[T, int], bool]) -> Optional[T]:
        ...
    @overload
    def single_or_none(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[T]:
        ...
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
    def drop(self, n: int) -> Sequence[T]:
        """
        Returns a Sequence containing all elements except first [n] elements.

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
            raise ValueError(f'Requested element count {n} is less than zero.')
        if n == 0:
            return self

        return DropTransform(self, n).as_sequence()

    # noinspection PyShadowingNames
    @overload
    def drop_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def drop_while(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def drop_while(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def drop_while(self, predicate: Callable[..., bool]) -> Sequence[T]:
        """
        Returns a Sequence containing all elements except first elements that satisfy the given [predicate].

        Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).drop_while(lambda x: x < 3 ).to_list()
        [3, 4, 1]
        """
        return DropWhileTransform(self, self._callback_overload_warpper(predicate)).as_sequence()
    
    def skip(self, n: int) -> Sequence[T]:
        """
        Returns a Sequence containing all elements except first [n] elements.

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
    def skip_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def skip_while(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def skip_while(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def skip_while(self, predicate: Callable[..., bool]) -> Sequence[T]:
        """
        Returns a Sequence containing all elements except first elements that satisfy the given [predicate].

        Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).skip_while(lambda x: x < 3 ).to_list()
        [3, 4, 1]
        """
        return self.drop_while(predicate)


    def take(self, n: int) -> Sequence[T]:
        """
        Returns an Sequence containing first [n] elements.

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
            raise ValueError(f'Requested element count {n} is less than zero.')
        if n == 0:
            return Sequence([])
        return TakeTransform(self, n).as_sequence()

    # noinspection PyShadowingNames
    @overload
    def take_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def take_while(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def take_while(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def take_while(self, predicate: Callable[..., bool]) -> Sequence[T]:
        """
        Returns an Sequence containing first elements satisfying the given [predicate].

        Example 1:
        >>> lst = ['a', 'b', 'c', 'd']
        >>> it(lst).take_while(lambda x: x in ['a', 'b']).to_list()
        ['a', 'b']
        """
        return TakeWhileTransform(self, self._callback_overload_warpper(predicate)).as_sequence()
    
    def take_last(self, n: int) -> Sequence[T]:
        """
        Returns an Sequence containing last [n] elements.

        Example 1:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(0).to_list()
        []

        Example 2:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(2).to_list()
        ['b', 'c']
        """
        if n < 0:
            raise ValueError(f'Requested element count {n} is less than zero.')
        if n == 0:
            return Sequence([])

        return self.drop(len(self) - n)

    # noinspection PyShadowingNames
    def sorted(self) -> Sequence[T]:
        """
        Returns an Sequence that yields elements of this Sequence sorted according to their natural sort order.

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
        lst.sort() # type: ignore
        return it(lst)

    # noinspection PyShadowingNames
    @overload
    def sorted_by(self, key_selector: Callable[[T], SupportsRichComparisonT]) -> Sequence[T]:
        ...
    @overload
    def sorted_by(self, key_selector: Callable[[T, int], SupportsRichComparisonT]) -> Sequence[T]:
        ...
    @overload
    def sorted_by(self, key_selector: Callable[[T, int, Sequence[T]], SupportsRichComparisonT]) -> Sequence[T]:
        ...
    def sorted_by(self, key_selector: Callable[..., SupportsRichComparisonT]) -> Sequence[T]:
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
        lst.sort(key=self._callback_overload_warpper(key_selector))
        return it(lst)

    def sorted_descending(self) -> Sequence[T]:
        """
        Returns a Sequence of all elements sorted descending according to their natural sort order.

        Example 1:
        >>> lst = ['b', 'c', 'a']
        >>> it(lst).sorted_descending().to_list()
        ['c', 'b', 'a']
        """
        return self.sorted().reversed()

    @overload
    def sorted_by_descending(self, key_selector: Callable[[T], SupportsRichComparisonT]) -> Sequence[T]:
        ...
    @overload
    def sorted_by_descending(self, key_selector: Callable[[T, int], SupportsRichComparisonT]) -> Sequence[T]:
        ...
    @overload
    def sorted_by_descending(self, key_selector: Callable[[T, int, Sequence[T]], SupportsRichComparisonT]) -> Sequence[T]:
        ...
    def sorted_by_descending(self, key_selector: Callable[..., SupportsRichComparisonT]) -> Sequence[T]:
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
    def sorted_with(self, comparator: Callable[[T, T], int]) -> Sequence[T]:
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
    def associate(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]:
        ...
    @overload
    def associate(self, transform: Callable[[T, int], Tuple[K, V]]) -> Dict[K, V]:
        ...
    @overload
    def associate(self, transform: Callable[[T, int, Sequence[T]], Tuple[K, V]]) -> Dict[K, V]:
        ...
    def associate(self, transform: Callable[..., Tuple[K, V]]) -> Dict[K, V]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}
        """
        transform = self._callback_overload_warpper(transform)
        dic: Dict[K, V] = dict()
        for i in self:
            k, v = transform(i)
            dic[k] = v
        return dic

    @overload
    def associate_by(self, key_selector: Callable[[T], K]) -> Dict[K, T]:
        ...
    @overload
    def associate_by(self, key_selector: Callable[[T, int], K]) -> Dict[K, T]:
        ...
    @overload
    def associate_by(self, key_selector: Callable[[T, int, Sequence[T]], K]) -> Dict[K, T]:
        ...
    @overload
    def associate_by(self, key_selector: Callable[[T], K], value_transform: Callable[[T], V]) -> Dict[K, V]:
        ...
    def associate_by(self, key_selector: Callable[..., K],
                     value_transform: Optional[Callable[[T], V]] = None) -> Union[Dict[K, T], Dict[K, V]]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate_by(lambda x: int(x))
        {1: '1', 2: '2', 3: '3'}

        Example 2:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate_by(lambda x: int(x), lambda x: x+x)
        {1: '11', 2: '22', 3: '33'}

        """
        key_selector = self._callback_overload_warpper(key_selector)

        dic: Dict[K, Any] = dict()
        for i in self:
            k = key_selector(i)
            dic[k] = i if value_transform is None else value_transform(i)
        return dic

    @overload
    def associate_by_to(self, destination: Dict[K, T], key_selector: Callable[[T], K]) -> Dict[K, T]:
        ...
    @overload
    def associate_by_to(self, destination: Dict[K, V], key_selector: Callable[[T], K], value_transform: Callable[[T], V]) -> Dict[K, V]:
        ...
    def associate_by_to(self, destination: Dict[K, Any], key_selector: Callable[[T], K],
                        value_transform: Optional[Callable[[T], Any]] = None) -> Dict[K, Any]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

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
    def all(self, predicate: Callable[[T], bool]) -> bool:
        ...
    @overload
    def all(self, predicate: Callable[[T, int], bool]) -> bool:
        ...
    @overload
    def all(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> bool:
        ...
    def all(self, predicate: Callable[..., bool]) -> bool:
        """
        Returns True if all elements of the Sequence satisfy the specified [predicate] function.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).all(lambda x: x > 0)
        True
        >>> it(lst).all(lambda x: x > 1)
        False
        """
        predicate = self._callback_overload_warpper(predicate)
        for i in self:
            if not predicate(i):
                return False
        return True
    
    @overload
    def any(self, predicate: Callable[[T], bool]) -> bool:
        ...
    @overload
    def any(self, predicate: Callable[[T, int], bool]) -> bool:
        ...
    @overload
    def any(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> bool:
        ...
    def any(self, predicate: Callable[..., bool]) -> bool:
        """
        Returns True if any elements of the Sequence satisfy the specified [predicate] function.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).any(lambda x: x > 0)
        True
        >>> it(lst).any(lambda x: x > 3)
        False
        """
        predicate = self._callback_overload_warpper(predicate)
        for i in self:
            if predicate(i):
                return True
        return False
    
    @overload
    def count(self) -> int:
        ...
    @overload
    def count(self, predicate: Callable[[T], bool]) -> int:
        ...
    @overload
    def count(self, predicate: Callable[[T, int], bool]) -> int:
        ...
    @overload
    def count(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> int:
        ...
    def count(self, predicate: Optional[Callable[..., bool]] = None) -> int:
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
        predicate = self._callback_overload_warpper(predicate)
        return sum(1 for i in self if predicate(i))
    
    def contains(self, value: T) -> bool:
        """
        Returns True if the Sequence contains the specified [value].

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
        Returns the element at the specified [index] in the Sequence.

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
        return self.element_at_or_else(index, lambda index: throw(IndexError(f'Index {index} out of range')))
    
    @overload
    def element_at_or_else(self, index: int, default: T) -> T:
        """
        Returns the element at the specified [index] in the Sequence or the [default] value if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_else(1, 'default')
        2
        >>> it(lst).element_at_or_else(4, lambda x: 'default')
        'default'
        """
        ...
    
    @overload
    def element_at_or_else(self, index: int, default: None) -> Optional[T]:
        ...
    @overload
    def element_at_or_else(self, index: int, default: Callable[[int], T]) -> T:
        ...
    def element_at_or_else(self, index: int, default: Union[Callable[[int], T], T, None]) -> Union[Optional[T], T]:
        """
        Returns the element at the specified [index] in the Sequence or the [default] value if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_else(1, lambda x: 'default')
        2
        >>> it(lst).element_at_or_else(4, lambda x: 'default')
        'default'

        """
        if (index < 0):
            return default(index) if callable(default) else default
        
        if type(self._iter) == SequenceTransform and isinstance(self._iter._iter, list) and index < len(self._iter._iter): # type: ignore
            return self._iter._iter[index] # type: ignore
        for i, e in enumerate(self):
            if i == index:
                return e
        return default(index) if callable(default) else default
    

    def element_at_or_default(self, index: int, default: T) -> T:
        """
        Returns the element at the specified [index] in the Sequence or the [default] value if the index is out of bounds.

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
        Returns the element at the specified [index] in the Sequence or None if the index is out of bounds.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).element_at_or_none(1)
        2
        >>> it(lst).element_at_or_none(4) is None
        True
        """
        return self.element_at_or_else(index, None)
    
    def distinct(self) -> Sequence[T]:
        """
        Returns a new Sequence containing the distinct elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3, 1, 2, 3]
        >>> it(lst).distinct().to_list()
        [1, 2, 3]

        Example 2:
        >>> lst = [(1, 'A'), (1, 'A'), (1, 'A'), (2, 'A'), (3, 'C'), (3, 'D')]
        >>> it(lst).distinct().sorted().to_list()
        [(1, 'A'), (2, 'A'), (3, 'C'), (3, 'D')]

        """
        return DistinctTransform(self).as_sequence()
    
    @overload
    def distinct_by(self, key_selector: Callable[[T], Any]) -> Sequence[T]:
        ...
    @overload
    def distinct_by(self, key_selector: Callable[[T, int], Any]) -> Sequence[T]:
        ...
    @overload
    def distinct_by(self, key_selector: Callable[[T, int, Sequence[T]], Any]) -> Sequence[T]:
        ...
    def distinct_by(self, key_selector: Callable[..., Any]) -> Sequence[T]:
        """
        Returns a new Sequence containing the distinct elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3, 1, 2, 3]
        >>> it(lst).distinct_by(lambda x: x%2).to_list()
        [1, 2]
        """
        return DistinctTransform(self, self._callback_overload_warpper(key_selector)).as_sequence()
    
    @overload
    def reduce(self, accumulator: Callable[[T, T], T]) -> T:
        ...
    @overload
    def reduce(self, accumulator: Callable[[R, T], R], initial: R) -> R:
        ...
    def reduce(self, accumulator: Callable[..., R], initial: Optional[R] = None) -> Optional[R]:
        """
        Returns the result of applying the specified [accumulator] function to the given Sequence's elements.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).reduce(lambda x, y: x+y)
        6
        """
        result: Optional[R] = initial
        for i, e in enumerate(self):
            if i == 0 and initial is None:
                result = e # type: ignore
                continue
            
            result = accumulator(result, e)
        return result
    
    def fold(self, initial: R, accumulator: Callable[[R, T], R]) -> R:
        """
        Returns the result of applying the specified [accumulator] function to the given Sequence's elements.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).fold(0, lambda x, y: x+y)
        6
        """
        return self.reduce(accumulator, initial)
    
    @overload
    def sum_of(self, selector: Callable[[T], int]) -> int:
        ...
    @overload
    def sum_of(self, selector: Callable[[T], float]) -> float:
        ...
    def sum_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        """
        Returns the sum of the elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum_of(lambda x: x)
        6
        """
        return sum(selector(i) for i in self)
    
    @overload
    def max_of(self, selector: Callable[[T], int]) -> int:
        ...
    @overload
    def max_of(self, selector: Callable[[T], float]) -> float:
        ...
    def max_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        """
        Returns the maximum element of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max_of(lambda x: x)
        3
        """
        return max(selector(i) for i in self)
    
    @overload
    def max_by_or_none(self, selector: Callable[[T], int]) -> Optional[T]:
        ...
    @overload
    def max_by_or_none(self, selector: Callable[[T], float]) -> Optional[T]:
        ...
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
    def max_by(self, selector: Callable[[T], int]) -> T:
        ...
    @overload
    def max_by(self, selector: Callable[[T], float]) -> T:
        ...
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
        ValueError: Sequence is empty.
        """
        max_item = self.max_by_or_none(selector)
        if max_item is None:
            raise ValueError("Sequence is empty.")
        return max_item

    @overload
    def min_of(self, selector: Callable[[T], int]) -> int:
        """
        Returns the minimum element of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min_of(lambda x: x)
        1
        """
        ...
    @overload
    def min_of(self, selector: Callable[[T], float]) -> float:
        ...
    def min_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        return min(selector(i) for i in self)
    
    
    @overload
    def min_by_or_none(self, selector: Callable[[T], int]) -> Optional[T]:
        ...
    @overload
    def min_by_or_none(self, selector: Callable[[T], float]) -> Optional[T]:
        ...
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
    def min_by(self, selector: Callable[[T], int]) -> T:
        ...
    @overload
    def min_by(self, selector: Callable[[T], float]) -> T:
        ...
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
        ValueError: Sequence is empty.
        """        
        min_item = self.min_by_or_none(selector)
        if min_item is None:
            raise ValueError("Sequence is empty.")
        
        return min_item
    
    def mean_of(self, selector: Callable[[T], Union[int, float]]) -> Union[int, float]:
        """
        Returns the mean of the elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).mean_of(lambda x: x)
        2.0
        """
        return self.sum_of(selector) / len(self)
    
    @overload
    def sum(self: Sequence[int]) -> int:
        """
        Returns the sum of the elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum()
        6
        """
        ...
    @overload
    def sum(self: Sequence[float]) -> float:
        ...
    def sum(self: Union[Sequence[int], Sequence[float]]) -> Union[float, int]:
        """
        Returns the sum of the elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum()
        6
        """
        return sum(self)
    
    @overload
    def max(self: Sequence[int]) -> int:
        ...
    @overload
    def max(self: Sequence[float]) -> float:
        ...
    def max(self: Union[Sequence[int], Sequence[float]]) -> Union[float, int]:
        """
        Returns the maximum element of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max()
        3
        """
        return max(self)
    
    @overload
    def min(self: Sequence[int]) -> int:
        """
        Returns the minimum element of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min()
        1
        """
        ...
    @overload
    def min(self: Sequence[float]) -> float:
        ...
    def min(self: Union[Sequence[int], Sequence[float]]) -> Union[float, int]:
        """
        Returns the minimum element of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min()
        1
        """
        return min(self)
    
    @overload
    def mean(self: Sequence[int]) -> float:
        """
        Returns the mean of the elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).mean()
        2.0
        """
        ...
    @overload
    def mean(self: Sequence[float]) -> float:
        ...
    def mean(self: Union[Sequence[int], Sequence[float]]) -> float:
        """
        Returns the mean of the elements of the given Sequence.

        Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).mean()
        2.0
        """
        return self.sum() / len(self)
    

    # noinspection PyShadowingNames
    def reversed(self) -> Sequence[T]:
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
    def flat_map(self, transform: Callable[[T], Iterable[R]]) -> Sequence[R]:
        ...
    @overload
    def flat_map(self, transform: Callable[[T, int], Iterable[R]]) -> Sequence[R]:
        ...
    @overload
    def flat_map(self, transform: Callable[[T, int, Sequence[T]], Iterable[R]]) -> Sequence[R]:
        ...
    def flat_map(self, transform: Callable[..., Iterable[R]]) -> Sequence[R]:
        """
        Returns a single list of all elements yielded from results of [transform]
        function being invoked on each element of original collection.

        Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flat_map(lambda x: x).to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        return self.map(transform).flatten()
    
    def flatten(self: Iterable[Iterable[R]]) -> Sequence[R]:
        """
        Returns a sequence of all elements from all sequences in this sequence.

        Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flatten().to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        return FlatteningTransform(self).as_sequence()
    
    @overload
    def group_by(self, key_selector: Callable[[T], K]) -> Sequence[Grouping[K, T]]:
        ...
    @overload
    def group_by(self, key_selector: Callable[[T, int], K]) -> Sequence[Grouping[K, T]]:
        ...
    @overload
    def group_by(self, key_selector: Callable[[T, int, Sequence[T]], K]) -> Sequence[Grouping[K, T]]:
        ...
    def group_by(self, key_selector: Callable[..., K]) -> Sequence[Grouping[K, T]]:
        """
        Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

        Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by(lambda x: x%2).map(lambda x: (x.key, x.values.to_list())).to_list()
        [(1, [1, 3, 5]), (0, [2, 4])]
        """
        return GroupingTransform(self, self._callback_overload_warpper(key_selector)).as_sequence()
    
    @overload
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T], K]) -> Dict[K, List[T]]:
        ...
    @overload
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T, int], K]) -> Dict[K, List[T]]:
        ...
    @overload
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T, int, Sequence[T]], K]) -> Dict[K, List[T]]:
        ...
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[..., K]) -> Dict[K, List[T]]:
        """
        Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

        Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by_to({}, lambda x: x%2)
        {1: [1, 3, 5], 0: [2, 4]}
        """
        key_selector = self._callback_overload_warpper(key_selector)
        for e in self:
            k = key_selector(e)
            if k not in destination:
                destination[k] = []
            destination[k].append(e)
        return destination

    @overload
    def for_each(self, action: Callable[[T], None]) -> None:
        ...
    @overload
    def for_each(self, action: Callable[[T, int], None]) -> None:
        ...
    @overload
    def for_each(self, action: Callable[[T, int, Sequence[T]], None]) -> None:
        ...
    def for_each(self, action: Callable[..., None]) -> None:
        """
        Invokes [action] function on each element of the given Sequence.

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
    def parallel_for_each(self, action: Callable[[T], None], max_workers: Optional[int]=None) -> None:
        ...
    @overload
    def parallel_for_each(self, action: Callable[[T, int], None], max_workers: Optional[int]=None) -> None:
        ...
    @overload
    def parallel_for_each(self, action: Callable[[T, int, Sequence[T]], None], max_workers: Optional[int]=None) -> None:
        ...
    def parallel_for_each(self, action: Callable[..., None], max_workers: Optional[int]=None) -> None:
        """
        Invokes [action] function on each element of the given Sequence in parallel.

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
    
    @deprecated("use `.for_each(lambda x, idx: ... )` instead.", category=None)
    def foreach_indexed(self, action: Callable[[T, int], None]) -> None:
        self.on_each(action)
    
    @overload
    def on_each(self, action: Callable[[T], None]) -> Sequence[T]:
        ...
    @overload
    def on_each(self, action: Callable[[T, int], None]) -> Sequence[T]:
        ...
    @overload
    def on_each(self, action: Callable[[T, int, Sequence[T]], None]) -> Sequence[T]:
        ...
    def on_each(self, action: Callable[..., None]) -> Sequence[T]:
        """
        Invokes [action] function on each element of the given Sequence.

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
        action = self._callback_overload_warpper(action)
        for i in self:
            action(i)
        return self
    
    @overload
    def parallel_on_each(self, action: Callable[[T], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread' ) -> Sequence[T]:
        ...
    @overload
    def parallel_on_each(self, action: Callable[[T, int], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread' ) -> Sequence[T]:
        ...
    @overload
    def parallel_on_each(self, action: Callable[[T, int, Sequence[T]], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread' ) -> Sequence[T]:
        ...
    def parallel_on_each(self, action: Callable[..., None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingTransform.Executor='Thread' ) -> Sequence[T]:
        """
        Invokes [action] function on each element of the given Sequence.

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
        action = self._callback_overload_warpper(action)
        for _ in ParallelMappingTransform(self, action, max_workers, chunksize, executor):
            pass
        return self
    

    @deprecated("use `.on_each(lambda x, idx: ... )` instead.", category=None)
    def on_each_indexed(self, action: Callable[[T, int], None]) -> Sequence[T]:
        return self.on_each(action)
    
    @overload
    def zip(self, other: Iterable[R]) -> Sequence[Tuple[T, R]]:
        ...
    @overload
    def zip(self, other: Iterable[R], transform: Callable[[T, R], V]) -> Sequence[V]:
        ...
    def zip(self, other: Iterable[Any], transform: Optional[Callable[..., Any]] = None) -> Sequence[Any]:
        """
        Returns a new Sequence of tuples, where each tuple contains two elements.
        
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
            transform = lambda *x:(*x,)
        return MergingTransform(self, other, transform).as_sequence()
    
    @overload
    def zip_with_next(self) -> Sequence[Tuple[T, T]]:
        ...
    @overload
    def zip_with_next(self, transform: Callable[[T, T], V]) -> Sequence[V]:
        ...
    def zip_with_next(self, transform: Optional[Callable[[T, T], Any]]=None) -> Sequence[Any]:
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
        return MergingWithNextTransform(self, transform or (lambda a, b: (a, b))).as_sequence()
    
    
    @overload
    def unzip(self: Sequence[Tuple[R, V]], as_sequence: Literal[True]) -> Tuple[Sequence[R], Sequence[V]] :
        ...
    @overload
    def unzip(self: Sequence[Tuple[R, V]]) -> Tuple[List[R], List[V]]:
        ...
    @overload
    def unzip(self, transform: Callable[[T], Tuple[R, V]], as_sequence: Literal[True]) -> Tuple[Sequence[R], Sequence[V]]:
        ...
    @overload
    def unzip(self, transform: Callable[[T, int], Tuple[R, V]], as_sequence: Literal[True]) -> Tuple[Sequence[R], Sequence[V]]:
        ...
    @overload
    def unzip(self, transform: Callable[[T, int, Sequence[T]], Tuple[R, V]], as_sequence: Literal[True]) -> Tuple[Sequence[R], Sequence[V]]:
        ...
    @overload
    def unzip(self, transform: Callable[[T], Tuple[R, V]]) -> Tuple[List[R], List[V]]:
        ...
    @overload
    def unzip(self, transform: Callable[[T, int], Tuple[R, V]]) -> Tuple[List[R], List[V]]:
        ...
    @overload
    def unzip(self, transform: Callable[[T, int, Sequence[T]], Tuple[R, V]]) -> Tuple[List[R], List[V]]:
        ...
    def unzip(self, transform: Union[Optional[Callable[..., Tuple[Any, Any]]], bool]=None, as_sequence: bool=False) -> Any: # type: ignore
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
        it = self
        if isinstance(transform, bool):
            as_sequence = transform
            transform = None

        if transform is not None:
            transform = self._callback_overload_warpper(transform)
            it = it.map(transform)
        
        a = it.map(lambda x: x[0]) # type: ignore
        b = it.map(lambda x: x[1]) # type: ignore

        if not as_sequence:
            return a.to_list(), b.to_list()
        return a, b


    def with_index(self):
        """
        Returns a sequence containing the elements of this sequence and their indexes.
        
        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).with_index().to_list()
        [IndexedValue(0, a), IndexedValue(1, b), IndexedValue(2, c)]
        """
        return self.indexed()
    

    @overload
    def shuffled(self) -> Sequence[T]:
        ...
    @overload
    def shuffled(self, seed: int) -> Sequence[T]:
        ...
    @overload
    def shuffled(self, seed: str) -> Sequence[T]:
        ...
    @overload
    def shuffled(self, random: Random) -> Sequence[T]:
        ...
    def shuffled(self, random: Optional[Union[Random, int, str]]=None) -> Sequence[T]: # type: ignore
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
        return ShufflingTransform(self, random).as_sequence()
    
    @overload
    def partition(self, predicate: Callable[[T], bool]) -> Tuple[List[T], List[T]]:
        ...
    @overload
    def partition(self, predicate: Callable[[T, int], bool]) -> Tuple[List[T], List[T]]:
        ...
    @overload
    def partition(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Tuple[List[T], List[T]]:
        ...
    @overload
    def partition(self, predicate: Callable[[T], bool], as_sequence: Literal[True]) -> Tuple[Sequence[T], Sequence[T]]:
        ...
    @overload
    def partition(self, predicate: Callable[[T, int], bool], as_sequence: Literal[True]) -> Tuple[Sequence[T], Sequence[T]]:
        ...
    @overload
    def partition(self, predicate: Callable[[T, int], bool], as_sequence: Literal[False]) -> Tuple[List[T], List[T]]:
        ...
    @overload
    def partition(self, predicate: Callable[[T, int, Sequence[T]], bool], as_sequence: Literal[True]) -> Tuple[Sequence[T], Sequence[T]]:
        ...
    def partition(self, predicate: Callable[..., bool], as_sequence: bool=False):
        """
        Partitions the elements of the given Sequence into two groups,
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
        predicate_a = self._callback_overload_warpper(predicate)
        predicate_b = self._callback_overload_warpper(predicate)
        part_a = self.filter(predicate_a)
        part_b = self.filter(lambda x: not predicate_b(x))
        if not as_sequence:
            return part_a.to_list(), part_b.to_list()
        return part_a, part_b 
    
    
    @overload
    def partition_indexed(self, predicate: Callable[[T, int], bool]) -> Tuple[List[T], List[T]]:
        ...
    @overload
    def partition_indexed(self, predicate: Callable[[T, int], bool], as_sequence: Literal[True]) -> Tuple[Sequence[T], Sequence[T]]:
        ...
    @deprecated("use `.partition(lambda x, idx: ... )` instead.", category=None)
    def partition_indexed(self, predicate: Callable[[T, int], bool], as_sequence: bool=False) -> Any:
        return self.partition(predicate, as_sequence) # type: ignore
    
    def indexed(self) -> Sequence[IndexedValue[T]]:
        return self.map(lambda x, i: IndexedValue(x, i))

    @overload
    def combinations(self, n: Literal[2]) -> Sequence[Tuple[T, T]]:
        ...
    @overload
    def combinations(self, n: Literal[3]) -> Sequence[Tuple[T, T, T]]: 
        ...
    @overload
    def combinations(self, n: Literal[4]) -> Sequence[Tuple[T, T, T, T]]: 
        ...
    @overload
    def combinations(self, n: Literal[5]) -> Sequence[Tuple[T, T, T, T, T]]: 
        ...
    def combinations(self, n: int) -> Sequence[Tuple[T, ...]]:
        """
        Returns a Sequence of all possible combinations of size [n] from the given Sequence.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).combinations(2).to_list()
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        """
        return CombinationTransform(self, n).as_sequence()
    

    def nth(self, n: int) -> T:
        """
        Returns the nth element of the given Sequence.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).nth(2)
        'c'
        """
        return self.skip(n).first()
    

    def windowed(self, size: int, step: int = 1, partialWindows: bool = False) -> Sequence[List[T]]:
        """
         Returns a Sequence of all possible sliding windows of size [size] from the given Sequence.

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
        return WindowedTransform(self, size, step, partialWindows).as_sequence()
    

    def chunked(self, size: int) -> Sequence[List[T]]:
        """
        Returns a Sequence of all possible chunks of size [size] from the given Sequence.

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

    def repeat(self, n: int) -> Sequence[T]:
        """
        Returns a Sequence containing this sequence repeated n times.

        Example 1:
        >>> lst = ['a', 'b']
        >>> it(lst).repeat(3).to_list()
        ['a', 'b', 'a', 'b', 'a', 'b']
        """

        return ConcatTransform([self] * n).as_sequence()
    
    def concat(self, *other: Sequence[T]) -> Sequence[T]:
        """
        Returns a Sequence of all elements of the given Sequence, followed by all elements of the given Sequence.

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
        return ConcatTransform([self, *other]).as_sequence()
    
    def intersect(self, *other: Sequence[T]) -> Sequence[T]:
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
        return IntersectionTransform([self, *other]).as_sequence()
    
    def union(self, *other: Sequence[T]) -> Sequence[T]:
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
    

    def join(self: Sequence[str], separator: str = ' ') -> str:
        """
        Joins the elements of the given Sequence into a string.

        Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).join(', ')
        'a, b, c'
        """
        return separator.join(self)
    
    @overload
    def progress(self) -> Sequence[T]:
        ...
    @overload
    def progress(self, progress_func: Union[Literal['tqdm'], Literal['tqdm_rich']]) -> Sequence[T]:
        ...
    @overload
    def progress(self, progress_func: Callable[[Iterable[T]], Iterable[T]]) -> Sequence[T]:
        ...
    def progress(self, progress_func: Union[Callable[[Iterable[T]], Iterable[T]], Literal['tqdm'], Literal['tqdm_rich'], None ] = None) -> Sequence[T]:
        """
        Returns a Sequence that enable a progress bar for the given Sequence.
        
        Example 1:
        >>> from tqdm import tqdm
        >>> from time import sleep
        >>> it(range(10)).progress(lambda x: tqdm(x, total=len(x))).parallel_map(lambda x: sleep(0.), max_workers=5).to_list() and None
        >>> for _ in it(list(range(10))).progress(lambda x: tqdm(x, total=len(x))).to_list(): pass
        """
        if progress_func is not None and callable(progress_func):
            return ProgressTransform(self, progress_func).as_sequence()
        
        def import_tqdm():
            if progress_func == 'tqdm_rich':
                from tqdm.rich import tqdm
            else:
                from tqdm import tqdm
            return tqdm

        try:
            tqdm = import_tqdm()
        except ImportError:
            from pip import main as pip
            pip(['install', 'tqdm'])
            tqdm = import_tqdm()
        
        return it(tqdm(self, total=len(self)))
        
    
    def typing_as(self, typ: Type[R]) -> Sequence[R]:
        """
        Cast the element as specific Type to gain code completion base on type annotations.
        """
        el = self.first_not_none_of_or_none()
        if el is None or isinstance(el, typ) or not isinstance(el, dict):
            return self # type: ignore

        class AttrDict(Dict[str, Any]):
            def __init__(self, value: Dict[str, Any]) -> None:
                super().__init__(**value)
                setattr(self, '__dict__', value)
                self.__getattr__  = value.__getitem__
                self.__setattr__ = value.__setattr__ # type: ignore
        return self.map(AttrDict) # type: ignore  # use https://github.com/cdgriffith/Box ?

    def to_set(self) -> Set[T]:
        """
        Returns a set containing all elements of this Sequence.

        Example 1:
        >>> it(['a', 'b', 'c', 'c']).to_set() == {'a', 'b', 'c'}
        True
        """
        return set(self)


    @overload
    def to_dict(self: Sequence[Tuple[K, V]]) -> Dict[K, V]:
        ...
    @overload
    def to_dict(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]:
        ...
    @overload
    def to_dict(self, transform: Callable[[T, int], Tuple[K, V]]) -> Dict[K, V]:
        ...
    @overload
    def to_dict(self, transform: Callable[[T, int, Sequence[T]], Tuple[K, V]]) -> Dict[K, V]:
        ...
    def to_dict(self, transform: Optional[Callable[..., Tuple[K, V]]]=None) -> Dict[K, V]:
        """
        Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

        Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).to_dict(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}

        Example 2:
        >>> lst = [(1, '1'), (2, '2'), (3, '3')]
        >>> it(lst).to_dict()
        {1: '1', 2: '2', 3: '3'}
        """
        return self.associate(transform or (lambda x: x)) # type: ignore

    def to_list(self) -> List[T]:
        """
        Returns a list with elements of the given Sequence.

        Example 1:
        >>> it(['b', 'c', 'a']).to_list()
        ['b', 'c', 'a']
        """
        return list(self)
    
    async def to_list_async(self: Iterable[Awaitable[T]]) -> List[T]:
        """
        Returns a list with elements of the given Sequence.

        Example 1:
        >>> it(['b', 'c', 'a']).to_list()
        ['b', 'c', 'a']
        """
        from asyncio import gather
        return await gather(*self) # type: ignore
    
    def let(self, block: Callable[[Sequence[T]], R]) -> R:
        """
        Calls the specified function [block] with `self` value as its argument and returns its result.
            
        Example 1:
        >>> it(['a', 'b', 'c']).let(lambda x: x.map(lambda y: y + '!')).to_list()
        ['a!', 'b!', 'c!']
        """
        return block(self)
    
    def also(self, block: Callable[[Sequence[T]], Any]) -> Sequence[T]:
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
        Returns the size of the given Sequence.
        """
        return len(self)

    @property
    def len(self) -> int:
        """
        Returns the length of the given Sequence.

        Example 1:
        >>> it(['a', 'b', 'c']).len
        3
        
        Example 1:
        >>> it(['a', 'b', 'c']).filter(lambda _, i: i % 2 == 0).len
        2
        """
        return len(self)
    
    def is_empty(self) -> bool:
        """
        Returns True if the Sequence is empty, False otherwise.
        """
        return len(self) == 0
    
    def __iter__(self) -> Iterator[T]:
        return self.__do_iter()
    
    def __do_iter(self) -> Iterator[T]:
        yield from self._iter

    def __len__(self):
        return len(self._iter)

    def __repr__(self):
        return repr(self.to_list())

    def __getitem__(self, key: int):
        """
        Returns the element at the specified [index] in the Sequence.

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
    def _callback_overload_warpper(self, callback: Callable[[T], R]) -> Callable[[T], R]:
        ...
    @overload
    def _callback_overload_warpper(self, callback: Callable[[T, int], R]) -> Callable[[T], R]:
        ...
    @overload
    def _callback_overload_warpper(self, callback: Callable[[T, int, Sequence[T]], R]) -> Callable[[T], R]:
        ...
    def _callback_overload_warpper(self, callback: Callable[..., R]) -> Callable[[T], R]:
        if hasattr(callback, '__code__'):
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


class SequenceTransform(Generic[IterableS, T], Iterable[T]):
    _iter: IterableS
    _cache: Optional[List[T]]

    def __init__(self, iterable: IterableS) -> None:
        super().__init__()
        self._iter = iterable
        self._cache = None
    
    def __iter__(self) -> Iterator[T]:
        if self._cache:
            yield from self._cache
        else:
            cache: List[T] = []
            for x in self.__do_iter__():
                cache.append(x)
                yield x
            self._cache = cache

    def __do_iter__(self) -> Iterator[T] :
        yield from self._iter

    def __len__(self) -> int:
        if not isinstance(self._iter, SequenceTransform):
            # not Sequence, just a wrapper of List, Tuple.etc.
            # we can get lenght of it directly.
            if hasattr(self._iter, '__len__'):
                return len(self._iter) # type: ignore
            elif hasattr(self._iter, '__length_hint__'):
                return self._iter.__length_hint__() # type: ignore
        # we need iterate all to get length
        cache = self._cache
        if cache is None:
            for _ in self:
                pass
            cache = self._cache
        if cache is not None:
            return len(cache)
        return 0

    def as_sequence(self) -> Sequence[T]:
        return it(self)


class FilteringTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        super().__init__(iterable)
        self._predicate = predicate
    
    def __do_iter__(self) -> Iterator[T]:
        for i in self._iter:
            if self._predicate(i):
                yield i
    

class MappingTransform(SequenceTransform[Iterable[T], R]):
    def __init__(self, iterable: Iterable[T], transform: Callable[[T], R]) -> None:
        super().__init__(iterable)
        self._transform = transform
    
    def __do_iter__(self) -> Iterator[R]:
        for i in self._iter:
            yield self._transform(i)


class ParallelMappingTransform(SequenceTransform[Sequence[T], R]):
    Executor = Literal['Thread', 'Process']

    def __init__(
            self, 
            iterable: Sequence[T], 
            transformer: Callable[[T], R], 
            max_workers: Optional[int]=None, 
            chunksize: int=1,
            executor: ParallelMappingTransform.Executor = 'Thread') -> None:

        super().__init__(iterable)
        self._transformer = transformer
        self._max_workers = max_workers
        self._executor = executor
        self._chunksize = chunksize
    
    def __do_iter__(self) -> Iterator[R]:
        import os

        def create_executor(max_workers: int):
            if self._executor == 'Process':
                from concurrent.futures import ProcessPoolExecutor
                return ProcessPoolExecutor(max_workers=max_workers)
            else:
                from concurrent.futures import ThreadPoolExecutor
                return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='PyIter worker')

        chunksize = self._chunksize
        max_workers = self._max_workers or min(32, (os.cpu_count() or 1) + 4)
        batch_size = max_workers * chunksize
        
        for batch in self._iter.chunked(batch_size):
            with create_executor(max_workers) as executor:
                yield from executor.map(self._transformer, batch, chunksize=chunksize)



class IndexedValue(NamedTuple, Generic[T]):
    val: T
    idx: int

    def __repr__(self) -> str:
        return f'IndexedValue({self.idx}, {self.val})'


class FlatteningTransform(SequenceTransform[Iterable[Iterable[T]], T]):
    def __init__(self, iterable: Iterable[Iterable[T]]) -> None:
        super().__init__(iterable)
    
    def __do_iter__(self) -> Iterator[T]:
        for i in self._iter:
            yield from i


class DropTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        super().__init__(iterable)
        self._n = n
        
    def __do_iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            if i < self._n:
                continue
            yield e


class DropWhileTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        super().__init__(iterable)
        self._predicate = predicate
    
    def __do_iter__(self) -> Iterator[T]:
        drop_state = True
        for e in self._iter:
            if drop_state and self._predicate(e):
                continue
            else:
                drop_state = False
            yield e


class TakeTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        super().__init__(iterable)
        self._n = n
    
    def __do_iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            if i >= self._n:
                break
            yield e


class TakeWhileTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        super().__init__(iterable)
        self._predicate = predicate
    
    def __do_iter__(self) -> Iterator[T]:
        take_state = True
        for e in self._iter:
            if take_state and self._predicate(e):
                yield e
            else:
                break


class MergingTransform(SequenceTransform[Iterable[T], V]):
    def __init__(self, 
        iterable: Iterable[T], 
        other: Iterable[R],
        transformer: Callable[[T, R], V] = lambda a,b:(a,b)
    ) -> None:
        super().__init__(iterable)
        self._other = other
        self._transformer = transformer
    
    def __do_iter__(self) -> Iterator[V]:
        iter1 = iter(self._iter)
        iter2 = iter(self._other)
        while True:
            try:
                yield self._transformer(next(iter1), next(iter2))
            except StopIteration:
                break


class IntersectionTransform(SequenceTransform[Iterable[Iterable[T]], T]):
    def __init__(self, iterable: Iterable[Iterable[T]]) -> None:
        super().__init__(iterable)
    
    def __do_iter__(self) -> Iterator[T]:
        iters = it(self._iter)
        seen: Set[T] = set()
        for v in iters.first():
            if v not in seen and iters.all(lambda iter: v in iter):
                yield v
                seen.add(v)


class MergingWithNextTransform(SequenceTransform[Iterable[T], V]):
    def __init__(self, 
    iterable: Iterable[T], 
    transformer: Callable[[T, T], V] = lambda a,b:(a,b)) -> None:
        super().__init__(iterable)
        self._transformer = transformer
    
    def __do_iter__(self) -> Iterator[V]:
        it = iter(self._iter)
        try:
            c = next(it)
            while True:
                n = next(it)
                yield self._transformer(c, n)
                c = n
        except StopIteration:
            pass


class DistinctTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], key_selector: Optional[Callable[[T], Any]]=None) -> None:
        super().__init__(iterable)
        self._key_selector = key_selector
    
    def __do_iter__(self) -> Iterator[T]:
        seen: Set[Any] = set()
        for e in self._iter:
            k = self._key_selector(e) if self._key_selector else e
            if k not in seen:
                seen.add(k)
                yield e


class DedupTransform(SequenceTransform[Iterable[T], List[T]]):
    def __init__(self, iterable: Iterable[T], key_selector: Optional[Callable[[T], Any]]=None) -> None:
        super().__init__(iterable)
        self._key_selector = key_selector

    def __do_iter__(self) -> Iterator[List[T]]:
        duplicates: List[T] = []
        seen: Optional[Any] = None

        for e in self._iter:
            k = self._key_selector(e) if self._key_selector else e
            if k != seen:
                if len(duplicates) > 0:
                    yield duplicates
                duplicates = [e]
                seen = k
                continue
            duplicates.append(e)

        if len(duplicates) > 0:
            yield duplicates



class Grouping(NamedTuple, Generic[K, T]):
    key: K
    values: Sequence[T]


class GroupingTransform(SequenceTransform[Iterable[T], Grouping[K, T]]):
    def __init__(self, iterable: Iterable[T], key_func: Callable[[T], K]) -> None:
        super().__init__(iterable)
        self._key_func = key_func
    
    @property
    def keys(self) -> Sequence[K]:
        return it(self).map(lambda x: x.key)
    
    @property
    def values(self) -> Sequence[Sequence[T]]:
        return it(self).map(lambda x: x.values)
    
    @property
    def items(self) -> Sequence[Grouping[K, T]]:
        return it(self)
    
    def __do_iter__(self) -> Iterator[Grouping[K, T]]:
        from collections import defaultdict
        d: DefaultDict[K, List[T]] = defaultdict(list)
        for e in self._iter:
            d[self._key_func(e)].append(e)
        yield from it(d.items()).map(lambda x: Grouping(x[0], it(x[1])))


class CombinationTransform(SequenceTransform[Iterable[T], Tuple[T, ...]]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        super().__init__(iterable)
        self._n = n
    
    def __do_iter__(self) -> Iterator[Tuple[T, ...]]:
        from itertools import combinations
        yield from combinations(self._iter, self._n)


class WindowedTransform(SequenceTransform[Iterable[T], List[T]]):
    def __init__(self, iterable: Iterable[T], size: int, step: int, partialWindows :bool) -> None:
        super().__init__(iterable)
        self._size = size
        self._step = step
        self._partialWindows = partialWindows
    
    def __do_iter__(self) -> Iterator[List[T]]:
        from collections import deque
        window: Deque[T] = deque(maxlen=self._size)
        for e in self._iter:
            window.append(e)
            if len(window) == self._size:
                yield list(window)
            if len(window) == self._size:
                for _ in range(self._step):
                    window.popleft()
        
        if self._partialWindows and len(window) > 0:
            yield list(window)


class ConcatTransform(SequenceTransform[Iterable[Iterable[T]], T]):
    def __init__(self, iterable: Iterable[Iterable[T]]) -> None:
        super().__init__(iterable)
    
    def __do_iter__(self) -> Iterator[T]:
        for i in self._iter:
            yield from i


class ShufflingTransform(SequenceTransform[Iterable[T], T]):
    def __init__(self, iterable: Iterable[T], random: Optional[Union[Random, str, int]]=None) -> None:
        super().__init__(iterable)
        if random is None or isinstance(random, (str, int)):
            from random import Random
            self._random = Random(random)
        else:
            self._random = random
    
    def __do_iter__(self) -> Iterator[T]:
        lst = list(self._iter)
        self._random.shuffle(lst)
        yield from lst


class ProgressTransform(SequenceTransform[Sequence[T], T]):
    def __init__(self, iterable: Sequence[T], progress_func: Callable[[Iterable[T]], Iterable[T]]) -> None:
        super().__init__(iterable)
        self._progress_func = progress_func
    
    @property
    def len(self) -> int:
        return len(self._iter)
    
    def __do_iter__(self) -> Iterator[T]:
        yield from self._progress_func(self._iter)
    
    # def to_list(self) -> List[T]:
    #     progress_func = self._progress_func
    #     class ListLike(List[T]):
    #         def __init__(self, iterable: Iterable[T]) -> None:
    #             super().__init__(iterable)
        
    #         def __iter__(self) -> Iterator[T]:
    #             yield from progress_func(it(super().__iter__()))

    #     return ListLike(self._iter)

    
def throw(exception: Exception) -> Any:
    raise exception


def none_or(value: Optional[T], default: T) -> T:
    return value if value is not None else default

def none_or_else(value: Optional[T], f: Callable[[], T]) -> T:
    return value if value is not None else f()

class SequenceProducer:
    @overload
    def __call__(self, elements: List[T]) -> Sequence[T]:
        ...
    @overload
    def __call__(self, elements: Iterable[T]) -> Sequence[T]:
        ...
    @overload
    def __call__(self, *elements: T) -> Sequence[T]:
        ...
    def __call__(self, *iterable: Union[Iterable[T], List[T], T]) -> Sequence[T]: # type: ignore
        if len(iterable) == 1:
            iter = iterable[0]
            if isinstance(iter, Sequence):
                return iter # type: ignore
            if isinstance(iter, Iterable) and not isinstance(iter, str):
                return Sequence(iter) # type: ignore
        return Sequence(iterable) # type: ignore
    
    def json(self, filepath: str, **kwargs: Dict[str, Any]) -> Sequence[Any]:
        """
        Reads and parses the input of a json file.
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f, **kwargs) # type: ignore
            return self(data)
    
    def csv(self, filepath: str):
        """
        Reads and parses the input of a csv file.
        """
        return self.read_csv(filepath)

    def read_csv(self, filepath: str, header: Optional[int]=0):
        """
        Reads and parses the input of a csv file.

        Example 1:
        >>> it.read_csv('tests/data/a.csv')
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

                return iter.map(lambda row: it(row).associate_by(
                    lambda _, ordinal: headers[ordinal] if ordinal < len(headers) else f'undefined_{ordinal}')
                )
            return iter


sequence = SequenceProducer()
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> sequence('hello', 'world').map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 2:
>>> sequence(['hello', 'world']).map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 3:
>>> sequence(range(10)).map(lambda x: x*x).to_list()
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""

seq = sequence
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> seq('hello', 'world').map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 2:
>>> seq(['hello', 'world']).map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 3:
>>> seq(range(10)).map(lambda x: x*x).to_list()
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""


iterate = sequence
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> iterate('hello', 'world').map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 2:
>>> iterate(['hello', 'world']).map(lambda x: x.upper()).to_list()
['HELLO', 'WORLD']

    Example 3:
>>> iterate(range(10)).map(lambda x: x*x).to_list()
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""


it = sequence
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
