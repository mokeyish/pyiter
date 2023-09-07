from __future__ import annotations
from typing import (
    overload, List, Set, Dict, Generic, Iterable, Iterator, Union, 
     Optional, Tuple, Type, TypeVar, Callable, Literal, NamedTuple,
     TYPE_CHECKING
)
if TYPE_CHECKING:
    from random import Random

import sys
if sys.version_info.major == 3 and sys.version_info.minor < 11:
    # Generic NamedTuple 
    origin__namedtuple_mro_entries =  NamedTuple.__mro_entries__
    NamedTuple.__mro_entries__ = lambda bases: origin__namedtuple_mro_entries(bases[:1])


T = TypeVar("T")
R = TypeVar("R")

K = TypeVar("K")
V = TypeVar("V")


class Sequence(Generic[T], Iterable[T]):
    """
    Given an [iterator] function constructs a [Sequence] that returns values through the [Iterator]
    provided by that function.
    
    The values are evaluated lazily, and the sequence is potentially infinite.
    """
    _iter: Iterable[T]
    _cache: Optional[List[T]]

    def __init__(self, v: Iterable[T]) -> None:
        super().__init__()
        self._iter = v
        self._cache = v if type(self) == Sequence and type(v) == list else None

    @overload
    def filter(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def filter(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def filter(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def filter(self, predicate: Callable[[T], bool]) -> Sequence[T]:
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
        return FilteringSequence(self, callback_overload_warpper(predicate, self))

    def filter_indexed(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing only elements matching the given [predicate].

         Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter_indexed(lambda x, i: i == 2).to_list()
        ['b2']
        """
        return self.indexed().filter(lambda x: predicate(*x)).map(lambda x: x.value)

    def filter_is_instance(self, typ: Type[R]) -> Sequence[R]:
        """
         Returns a Sequence containing all elements that are instances of specified type parameter typ.

        Example 1:
        >>> lst = [ 'a1', 1, 'b2', 3]
        >>> it(lst).filter_is_instance(int).to_list()
        [1, 3]

        """
        return self.filter(lambda x: type(x) == typ)

    @overload
    def filter_not(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        ...
    @overload
    def filter_not(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        ...
    @overload
    def filter_not(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Sequence[T]:
        ...
    def filter_not(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing all elements not matching the given [predicate].

         Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter_not(lambda x: x.startswith('a')).to_list()
        ['b1', 'b2']

        
         Example 1:
        >>> lst = [ 'a1', 'a2', 'b1', 'b2']
        >>> it(lst).filter_not(lambda x, i: x.startswith('a') and i % 2 == 0 ).to_list()
        ['a2', 'b1', 'b2']
        """
        predicate = callback_overload_warpper(predicate, self)
        return self.filter(lambda x: not predicate(x))

    def filter_not_none(self) -> Sequence[T]:
        """
         Returns a Sequence containing all elements that are not `None`.

         Example 1:
        >>> lst = [ 'a', None, 'b']
        >>> it(lst).filter_not_none().to_list()
        ['a', 'b']
        """
        return self.filter(lambda x: x is not None)
    
    @overload
    def map(self, transform: Callable[[T], R]) -> Sequence[R]:
        ...
    @overload
    def map(self, transform: Callable[[T, int], R]) -> Sequence[R]:
        ...
    @overload
    def map(self, transform: Callable[[T, int, Sequence[T]], R]) -> Sequence[R]:
        ...
    def map(self, transform: Callable[[T], R]) -> Sequence[R]:
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
        """
        return MappingSequence(self, callback_overload_warpper(transform, self))
    

    def map_indexed(self, transform: Callable[[T, int], R]) -> Sequence[R]:
        """
         Returns a Sequence containing the results of applying the given [transform] function
         to each element in the original Sequence.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).map_indexed(lambda x, i: x['age'] + i).to_list()
        [12, 14]
        """
        return self.indexed().map(lambda x: transform(x.value, x.index))

    @overload
    def map_not_none(self, transform: Callable[[T], Optional[R]]) -> Sequence[R]:
        ...
    @overload
    def map_not_none(self, transform: Callable[[T, int], Optional[R]]) -> Sequence[R]:
        ...
    @overload
    def map_not_none(self, transform: Callable[[T, int, Sequence[T]], Optional[R]]) -> Sequence[R]:
        ...
    def map_not_none(self, transform: Callable[[T], Optional[R]]) -> Sequence[R]:
        """
         Returns a Sequence containing only the non-none results of applying the given [transform] function
        to each element in the original collection.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': None}]
        >>> it(lst).map_not_none(lambda x: x['age']).to_list()
        [12]
        """
        return self.map(transform).filter_not_none()
    
    @overload
    def parallel_map(self, transform: Callable[[T], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread') -> Sequence[R]:
        ...
    @overload
    def parallel_map(self, transform: Callable[[T, int], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread') -> Sequence[R]:
        ...
    @overload
    def parallel_map(self, transform: Callable[[T, int, Sequence[T]], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread') -> Sequence[R]:
        ...
    def parallel_map(self, transform: Callable[[T], R], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread') -> Sequence[R]:
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
        return ParallelMappingSequence(self, callback_overload_warpper(transform, self), max_workers, chunksize, executor)

    @overload
    def find(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...
    @overload
    def find(self, predicate: Callable[[T, int], bool]) -> Optional[T]:
        ...
    @overload
    def find(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> Optional[T]:
        ...
    def find(self, predicate: Callable[[T], bool]) -> Optional[T]:
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
    def first(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
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
    def first_not_none_of(self, transform: Callable[[T], Optional[R]]) -> R:
        ...
    @overload
    def first_not_none_of(self, transform: Callable[[T, int], Optional[R]]) -> R:
        ...
    @overload
    def first_not_none_of(self, transform: Callable[[T, int, Sequence[T]], Optional[R]]) -> R:
        ...
    def first_not_none_of(self, transform: Callable[[T], Optional[R]]) -> R:
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
        v = self.first_not_none_of_or_none(transform)
        if v is None:
            raise ValueError('No element of the Sequence was transformed to a non-none value.')
        return v
    
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T], Optional[R]]) -> Optional[R]:
        ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T, int], Optional[R]]) -> Optional[R]:
        ...
    @overload
    def first_not_none_of_or_none(self, transform: Callable[[T, int, Sequence[T]], Optional[R]]) -> Optional[R]:
        ...
    def first_not_none_of_or_none(self, transform: Callable[[T], Optional[R]]) -> Optional[R]:
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
        return self.map_not_none(transform).first_or_none()

    @overload
    def first_or_none(self) -> Optional[T]:
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
    @overload
    def first_or_none(self) -> Optional[T]:
        ...
    def first_or_none(self, predicate: Optional[Callable[[T], bool]] = None) -> Optional[T]:
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
    def first_or_default(self, predicate: Union[Callable[[T], bool], T], default: Optional[T] = None) -> T:
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
            return self.first_or_none(predicate) or default
        else:
            default = predicate
        return self.first_or_none() or default

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
    def last(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
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
        v = self.last_or_none(predicate)
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
    def last_or_none(self, predicate: Optional[Callable[[T], bool]] = None) -> Optional[T]:
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
        for i, x in enumerate(self._iter):
            if x == element:
                return i
        return -1
    
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
        seq = self.reversed()
        last_idx = len(seq) - 1;
        for i, x in enumerate(seq):
            if x == element:
                return last_idx - i
        return -1
    
    @overload
    def index_of_first(self, predicate: Callable[[T], bool]) -> int:
        ...
    @overload
    def index_of_first(self, predicate: Callable[[T, int], bool]) -> int:
        ...
    @overload
    def index_of_first(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> int:
        ...
    def index_of_first(self, predicate: Callable[[T], bool]) -> int:
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
        """
        predicate = callback_overload_warpper(predicate, self)
        for i, x in enumerate(self._iter):
            if predicate(x):
                return i
        return -1
    
    @overload
    def index_of_last(self, predicate: Callable[[T], bool]) -> int:
        ...
    @overload
    def index_of_last(self, predicate: Callable[[T, int], bool]) -> int:
        ...
    @overload
    def index_of_last(self, predicate: Callable[[T, int, Sequence[T]], bool]) -> int:
        ...
    def index_of_last(self, predicate: Callable[[T], bool]) -> int:
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
        """
        seq = self.reversed()
        last_idx = len(seq) - 1;
        predicate = callback_overload_warpper(predicate, self)
        for i, x in enumerate(seq):
            if predicate(x):
                return last_idx - i
        return -1

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
    def single(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
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
        if not found:
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
    def single_or_none(self, predicate: Optional[Callable[[T], bool]] = None) -> Optional[T]:
        """
         Returns the single element matching the given [predicate], or `None` if element was not found
        or more than one element was found.

         Exmaple 1:
        >>> lst = ['a']
        >>> it(lst).single_or_none()
        'a'

         Exmaple 2:
        >>> lst = []
        >>> it(lst).single_or_none() is None
        True

         Exmaple 2:
        >>> lst = ['a', 'b']
        >>> it(lst).single_or_none() is None
        True
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

        return DropSequence(self, n)

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
    def drop_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing all elements except first elements that satisfy the given [predicate].

         Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).drop_while(lambda x: x < 3 ).to_list()
        [3, 4, 1]
        """
        return DropWhileSequence(self, callback_overload_warpper(predicate, self))
    
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
    def skip_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
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
        return TakeSequence(self, n)

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
    def take_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns an Sequence containing first elements satisfying the given [predicate].

         Example 1:
        >>> lst = ['a', 'b', 'c', 'd']
        >>> it(lst).take_while(lambda x: x in ['a', 'b']).to_list()
        ['a', 'b']

        """
        return TakeWhileSequence(self, callback_overload_warpper(predicate, self))
    
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
        lst.sort()
        return it(lst)

    # noinspection PyShadowingNames
    @overload
    def sorted_by(self, key_selector: Callable[[T], R]) -> Sequence[T]:
        ...
    @overload
    def sorted_by(self, key_selector: Callable[[T, int], R]) -> Sequence[T]:
        ...
    @overload
    def sorted_by(self, key_selector: Callable[[T, int, Sequence[T]], R]) -> Sequence[T]:
        ...
    def sorted_by(self, key_selector: Callable[[T], R]) -> Sequence[T]:
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
        lst.sort(key=callback_overload_warpper(key_selector, self))
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
    def sorted_by_descending(self, key_selector: Callable[[T], R]) -> Sequence[T]:
        ...
    @overload
    def sorted_by_descending(self, key_selector: Callable[[T, int], R]) -> Sequence[T]:
        ...
    @overload
    def sorted_by_descending(self, key_selector: Callable[[T, int, Sequence[T]], R]) -> Sequence[T]:
        ...
    def sorted_by_descending(self, key_selector: Callable[[T], R]) -> Sequence[T]:
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
    def associate(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]:
        """
         Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

         Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}

        """
        transform = callback_overload_warpper(transform, self)
        dic = dict()
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
    def associate_by(self, key_selector: Callable[[T], K],
                     value_transform: Optional[Callable[[T], V]] = None) -> Dict[K, Union[V, T]]:
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
        key_selector = callback_overload_warpper(key_selector, self)

        dic = dict()
        for i in self:
            k = key_selector(i)
            dic[k] = i if value_transform is None else value_transform(i)
        return dic

    @overload
    def associate_by_to(self, destination: Dict[K, T], key_selector: Callable[[T], K]) -> Dict[K, T]:
        ...
    @overload
    def associate_by_to(self, destination: Dict[K, V], key_selector: Callable[[T], K],
                        value_transform: Callable[[T], V]) -> Dict[K, V]:
        ...
    def associate_by_to(self, destination: Dict[K, V], key_selector: Callable[[T], K],
                        value_transform: Optional[Callable[[T], V]] = None) -> Dict[K, Union[V, T]]:
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
    def all(self, predicate: Callable[[T], bool]) -> bool:
        """
         Returns True if all elements of the Sequence satisfy the specified [predicate] function.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).all(lambda x: x > 0)
        True
        >>> it(lst).all(lambda x: x > 1)
        False

        """
        predicate = callback_overload_warpper(predicate, self)
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
    def any(self, predicate: Callable[[T], bool]) -> bool:
        """
         Returns True if any elements of the Sequence satisfy the specified [predicate] function.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).any(lambda x: x > 0)
        True
        >>> it(lst).any(lambda x: x > 3)
        False

        """
        predicate = callback_overload_warpper(predicate, self)
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
    def count(self, predicate: Optional[Callable[[T], bool]] = None) -> int:
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
        predicate = callback_overload_warpper(predicate, self)
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
        return DistinctSequence(self)
    
    @overload
    def distinct_by(self, key_selector: Callable[[T], K]) -> Sequence[T]:
        ...
    @overload
    def distinct_by(self, key_selector: Callable[[T, int], K]) -> Sequence[T]:
        ...
    @overload
    def distinct_by(self, key_selector: Callable[[T, int, Sequence[T]], K]) -> Sequence[T]:
        ...
    def distinct_by(self, key_selector: Callable[[T], K]) -> Sequence[T]:
        """
         Returns a new Sequence containing the distinct elements of the given Sequence.

         Example 1:
        >>> lst = [1, 2, 3, 1, 2, 3]
        >>> it(lst).distinct_by(lambda x: x%2).to_list()
        [1, 2]

        """
        return DistinctSequence(self, callback_overload_warpper(key_selector, self))
    
    def reduce(self, accumulator: Callable[[R, T], R], initial: Optional[R] = None) -> T:
        """
         Returns the result of applying the specified [accumulator] function to the given Sequence's elements.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).reduce(lambda x, y: x+y)
        6

        """
        result = initial
        for i, e in enumerate(self):
            if i == 0 and initial is None:
                result = e
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

         Exmaple 2:
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
    def sum(self: Sequence[Union[float, int]]) -> Union[float, int]:
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
        """
         Returns the maximum element of the given Sequence.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max()
        3

        """
        ...
    @overload
    def max(self: Sequence[float]) -> float:
        ...
    def max(self: Sequence[Union[float, int]]) -> Union[float, int]:
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
    def min(self: Sequence[Union[float, int]]) -> Union[float, int]:
        """
         Returns the minimum element of the given Sequence.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).min()
        1

        """
        return min(self)
    
    @overload
    def mean(self: Sequence[int]) -> int:
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
    def mean(self: Sequence[Union[int, float]]) -> Union[int, float]:
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
    def flat_map(self, transform: Callable[[T], Iterable[R]]) -> Sequence[R]:
        """
         Returns a single list of all elements yielded from results of [transform]
        function being invoked on each element of original collection.

         Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flat_map(lambda x: x).to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        return FlatteningSequence(self, callback_overload_warpper(transform, self))
    
    @overload
    def flatten(self: Sequence[List[R]]) -> Sequence[R]:
        """
         Returns a sequence of all elements from all sequences in this sequence.

         Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flatten().to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        ...
    @overload
    def flatten(self: Sequence[Set[R]]) -> Sequence[R]:
        ...
    @overload
    def flatten(self: Sequence[Tuple[R, ...]]) -> Sequence[R]:
        ...
    @overload
    def flatten(self: Sequence[Sequence[R]]) -> Sequence[R]:
        ...
    def flatten(self: Sequence[Iterable[R]]) -> Sequence[R]:
        """
         Returns a sequence of all elements from all sequences in this sequence.

         Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flatten().to_list()
        ['a', 'b', 'c', 'd', 'e']
        """
        return FlatteningSequence(self)
    
    @overload
    def group_by(self, key_selector: Callable[[T], K]) -> GroupingSequence[K, T]:
        ...
    @overload
    def group_by(self, key_selector: Callable[[T, int], K]) -> GroupingSequence[K, T]:
        ...
    @overload
    def group_by(self, key_selector: Callable[[T, int, Sequence[T]], K]) -> GroupingSequence[K, T]:
        ...
    def group_by(self, key_selector: Callable[[T], K]) -> GroupingSequence[K, T]:
        """
         Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

         Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by(lambda x: x%2).map(lambda x: (x.key, x.values.to_list())).to_list()
        [(1, [1, 3, 5]), (0, [2, 4])]
        """
        return GroupingSequence(self, callback_overload_warpper(key_selector, self))
    
    @overload
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T], K]) -> Dict[K, List[T]]:
        ...
    @overload
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T, int], K]) -> Dict[K, List[T]]:
        ...
    @overload
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T, int, Sequence[T]], K]) -> Dict[K, List[T]]:
        ...
    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T], K]) -> Dict[K, List[T]]:
        """
         Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

         Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by_to({}, lambda x: x%2)
        {1: [1, 3, 5], 0: [2, 4]}
        """
        key_selector = callback_overload_warpper(key_selector, self)
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
    def for_each(self, action: Callable[[T], None]) -> None:
        """
         Invokes [action] function on each element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).for_each(lambda x: print(x))
        a
        b
        c
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
    def parallel_for_each(self, action: Callable[[T], None], max_workers: Optional[int]=None) -> None:
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
    
    def foreach_indexed(self, action: Callable[[T, int], None]) -> None:
        """
         Invokes [action] function on each element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).foreach_indexed(lambda x, i: print(x, i))
        a 0
        b 1
        c 2
        """
        self.on_each_indexed(action)
    
    @overload
    def on_each(self, action: Callable[[T], None]) -> Sequence[T]:
        ...
    @overload
    def on_each(self, action: Callable[[T, int], None]) -> Sequence[T]:
        ...
    @overload
    def on_each(self, action: Callable[[T, int, Sequence[T]], None]) -> Sequence[T]:
        ...
    def on_each(self, action: Callable[[T], None]) -> Sequence[T]:
        """
         Invokes [action] function on each element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).on_each(lambda x: print(x)) and None
        a
        b
        c
        """
        action = callback_overload_warpper(action, self)
        for i in self:
            action(i)
        return self
    
    def parallel_on_each(self, action: Callable[[T], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread' ) -> Sequence[T]:
        ...
    def parallel_on_each(self, action: Callable[[T, int], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread' ) -> Sequence[T]:
        ...
    def parallel_on_each(self, action: Callable[[T, int, Sequence[T]], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread' ) -> Sequence[T]:
        ...
    def parallel_on_each(self, action: Callable[[T], None], max_workers: Optional[int]=None, chunksize: int=1, executor: ParallelMappingSequence.Executor='Thread' ) -> Sequence[T]:
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
        action = callback_overload_warpper(action, self)
        for _ in ParallelMappingSequence(self, action, max_workers, chunksize, executor):
            pass
        return self
    
    def on_each_indexed(self, action: Callable[[T, int], None]) -> Sequence[T]:
        """
         Invokes [action] function on each element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).on_each_indexed(lambda x, i: print(x, i)) and None
        a 0
        b 1
        c 2
        """
        for i, e in enumerate(self):
            action(e, i)
        return self
    
    @overload
    def zip(self, other: Sequence[T]) -> Sequence[Tuple[T, T]]:
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
        ...
    @overload
    def zip(self, other: Iterable[R], transform: Callable[[T, R], V]) -> Sequence[V]:
        ...
    def zip(self, other: Iterable[R], transform: Callable[[T, R], V] = lambda a, b:(a, b)) -> Sequence[V]:
        return MergingSequence(self, other, transform)
    
    @overload
    def zip_with_next(self) -> Sequence[Tuple[T, T]]:
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
        ...
    @overload
    def zip_with_next(self, transform: Callable[[T, T], V]) -> Sequence[V]:
        ...
    def zip_with_next(self, transform: Optional[Callable[[T, T], V]]=None) -> Tuple[Sequence[V], Sequence[Tuple[T, T]]]:
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
        return MergingWithNextSequence(self, transform or (lambda a, b: (a, b)))
    
    
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
    def unzip(self, transform: Union[Optional[Callable[[T], Tuple[R, V]]], bool]=None, as_sequence: bool=False):
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
        if type(transform) == bool:
            as_sequence = transform
            transform = None

        if transform is not None:
            transform = callback_overload_warpper(transform, self)
            it = it.map(transform)
        
        a = it.map(lambda x: x[0])
        b = it.map(lambda x: x[1])

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
        return IndexingSequence(self)
    

    @overload
    def shuffled(self) -> Sequence[T]:
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
        ...
    @overload
    def shuffled(self, random: Random) -> Sequence[T]:
        ...
    @overload
    def shuffled(self, seed: Union[int, str]) -> Sequence[T]:
        ...
    def shuffled(self, random: Optional[Union[Random, str, int]]=None) -> Sequence[T]:
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
        return ShufflingSequence(self, random)
    
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
    def partition(self, predicate: Callable[[T, int, Sequence[T]], bool], as_sequence: Literal[True]) -> Tuple[Sequence[T], Sequence[T]]:
        ...
    def partition(self, predicate: Callable[[T], bool], as_sequence: bool=False) -> Tuple[Sequence[T], Sequence[T]]:
        """
         Partitions the elements of the given Sequence into two groups,
         the first group containing the elements for which the predicate returns true,
         and the second containing the rest.

         Example 1:
        >>> lst = ['a', 'b', 'c', '2']
        >>> it(lst).partition(lambda x: x.isalpha())
        (['a', 'b', 'c'], ['2'])
        """
        predicate_a = callback_overload_warpper(predicate, self)
        predicate_b = callback_overload_warpper(predicate, self)
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
    def partition_indexed(self, predicate: Callable[[T, int], bool], as_sequence: bool=False) -> Tuple[Sequence[T], Sequence[T]]:
        """
         Partitions the elements of the given Sequence into two groups,
         the first group containing the elements for which the predicate returns true,
         and the second containing the rest.

         Example 1:
        >>> lst = ['a', 'b', 'c', '2']
        >>> it(lst).partition_indexed(lambda x, i: i % 2 == 0)
        (['a', 'c'], ['b', '2'])
        """
        indexed = self.indexed()
        part_a = indexed.filter(lambda x: predicate(x.value, x.index)).map(lambda x: x.value)
        part_b = indexed.filter(lambda x: not predicate(x.value, x.index)).map(lambda x: x.value)
        if not as_sequence:
            return part_a.to_list(), part_b.to_list()
        return part_a, part_b 
    
    def indexed(self) -> IndexingSequence[T]:
        return IndexingSequence(self)

    @overload
    def combinations(self, n: Literal[2]) -> Sequence[Tuple[T, T]]: 
        """
         Returns a Sequence of all possible combinations of size [n] from the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).combinations(2).to_list()
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        """
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
        return CombinationSequence(self, n)
    

    def nth(self, n: int) -> T:
        """
         Returns the nth element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).nth(2)
        'c'
        """
        return self.skip(n).first( )
    

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
        return WindowedSequence(self, size, step, partialWindows)
    

    def chunked(self, size: int) -> Sequence[List[T]]:
        """
         Returns a Sequence of all possible chunks of size [size] from the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).chunked(3).to_list()
        [['a', 'b', 'c'], ['d', 'e']]
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

        return ConcatSequence([self] * n)
    
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
        return ConcatSequence([self, *other])
    

    def join(self, separator: str = ' ') -> str:
        """
         Joins the elements of the given Sequence into a string.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).join(', ')
        'a, b, c'
        """
        return separator.join(self)
    

    def progress(self, progress_func: Callable[[Iterable[T]], Iterable[T]]) -> Sequence[T]:
        """
         Returns a Sequence that enable a progress bar for the given Sequence.
        
         Example 1:
        >>> from tqdm import tqdm
        >>> from time import sleep
        >>> it(range(10)).progress(lambda x: tqdm(x, total=len(x))).parallel_map(lambda x: sleep(0.), max_workers=5).to_list() and None
        >>> for _ in it(list(range(10))).progress(lambda x: tqdm(x, total=len(x))).to_list(): pass
        """
        return ProgressSequence(self, progress_func)


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
    def to_dict(self, transform: Optional[Callable[[T], Tuple[K, V]]]=None) -> Dict[K, V]:
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
        return self.associate(transform or (lambda x: x))

    def to_list(self) -> List[T]:
        """
         Returns a list with elements of the given Sequence.

         Example 1:
        >>> it(['b', 'c', 'a']).to_list()
        ['b', 'c', 'a']
        """
        return list(self)
    
    def let(self, block: Callable[[Sequence[T]], R]) -> R:
        """
         Calls the specified function [block] with `self` value as its argument and returns its result.
            
         Example 1:
        >>> it(['a', 'b', 'c']).let(lambda x: x.map(lambda y: y + '!')).to_list()
        ['a!', 'b!', 'c!']
        """
        return block(self)
    
    def also(self, block: Callable[[Sequence[T]], R]) -> R:
        """
         Calls the specified function [block] with `self` value as its argument and returns `self` value.
            
         Example 1:
        >>> it(['a', 'b', 'c']).also(lambda x: x.map(lambda y: y + '!')).to_list()
        ['a', 'b', 'c']
        """
        block(self)
        return self
    
    def __len__(self) -> int:
        if not isinstance(self._iter, Sequence):
            if hasattr(self._iter, '__len__'):
                return len(self._iter)
            elif hasattr(self._iter, '__length_hint__'):
                return self._iter.__length_hint__()
        if self._cache is None:
            for _ in self:
                pass
        return len(self._cache)

    def __repr__(self) -> str:
        if self._cache:
            return str(self._cache)
        return 'Sequence(...)'

    def __iter__(self) -> Iterator[T]:
        if self._cache:
            yield from self._cache
        else:
            cache = []
            for x in self.__do_iter__():
                cache.append(x)
                yield x
            self._cache = cache
    
    def __do_iter__(self) -> Iterator[T] :
        yield from self._iter
    


class Index:
    idx = 0
    def __call__(self):
        val = self.idx
        self.idx += 1
        return val


def callback_overload_warpper(callback: Callable, seq: Sequence) -> Callable:
    if callback.__code__.co_argcount == 2:
        index = Index()
        return lambda x: callback(x, index())
    if callback.__code__.co_argcount == 3:
        index = Index()
        return lambda x: callback(x, index(), seq)
    return callback


class FilteringSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        super().__init__(iterable)
        self._predicate = predicate
    
    def __do_iter__(self) -> Iterator[T]:
        for i in self._iter:
            if self._predicate(i):
                yield i


class MappingSequence(Sequence[R]):
    def __init__(self, iterable: Iterable[T], transform: Callable[[T], R]) -> None:
        super().__init__(iterable)
        self._transform = transform
    
    def __do_iter__(self) -> Iterator[R]:
        for i in self._iter:
            yield self._transform(i)


class ParallelMappingSequence(Sequence[R]):
    Executor = Literal['Thread', 'Process']

    def __init__(
            self, 
            iterable: Iterable[Sequence[T]], 
            transformer: Callable[[T], R], 
            max_workers: Optional[int]=None, 
            chunksize: int=1,
            executor: ParallelMappingSequence.Executor = 'Thread') -> None:

        super().__init__(iterable)
        self._transformer = transformer
        self._max_workers = max_workers
        self._executor = executor
        self._chunksize = chunksize
    
    def __do_iter__(self) -> Iterator[R]:
        import os

        def create_executor(max_workers: int, ):
            if self._executor == 'Process':
                from concurrent.futures import ProcessPoolExecutor
                return ProcessPoolExecutor(max_workers=max_workers)
            else:
                from concurrent.futures import ThreadPoolExecutor
                return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='PyIter worker')

        size = len(self._iter)
        chunksize = self._chunksize
        max_workers = self._max_workers or min(32, (os.cpu_count() or 1) + 4)
        batch_count = -1 * (-size // chunksize)
        max_workers = min(max_workers, batch_count)

        batch_size = max_workers * chunksize

        if batch_size < size:
            for batch in it(self._iter).chunked(batch_size):
                with create_executor(max_workers) as executor:
                    yield from executor.map(self._transformer, batch, chunksize=chunksize)
        else:
            with create_executor(max_workers) as executor:
                yield from executor.map(self._transformer, self._iter, chunksize=chunksize)


class IndexedValue(NamedTuple, Generic[T]):
    value: T
    index: int
    
    def __repr__(self) -> str:
        return f'IndexedValue({self.index}, {self.value})'


class IndexingSequence(Sequence[IndexedValue[T]]):
    def __init__(self, iterable: Iterable[T]) -> None:
        super().__init__(iterable)
    
    def __do_iter__(self) -> Iterator[T]:
        for i, v in enumerate(self._iter):
            yield IndexedValue(v, i)


class FlatteningSequence(Sequence[R]):
    def __init__(self, iterable: Iterable[Sequence[T]], transformer: Optional[Callable[[T], R]]=None) -> None:
        super().__init__(iterable)
        self._transformer = transformer
    
    def __do_iter__(self) -> Iterator[R]:
        if self._transformer:
            for i in self._iter:
                yield from self._transformer(i)
        else:
            for i in self._iter:
                yield from i


class DropSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        super().__init__(iterable)
        self._n = n
        
    def __do_iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            if i < self._n:
                continue
            yield e


class DropWhileSequence(Sequence[T]):
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


class TakeSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        super().__init__(iterable)
        self._n = n
    
    def __do_iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            if i >= self._n:
                break
            yield e


class TakeWhileSequence(Sequence[T]):
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


class MergingSequence(Sequence[V]):
    def __init__(self, 
    iterable: Iterable[T], 
    other: Iterable[R],
    transformer: Callable[[T, R], V] = lambda a,b:(a,b)) -> None:
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


class MergingWithNextSequence(Sequence[V]):
    def __init__(self, 
    iterable: Iterable[T], 
    transformer: Callable[[T, R], V] = lambda a,b:(a,b)) -> None:
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


class DistinctSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], key_selector: Optional[Callable[[T], K]]=None) -> None:
        super().__init__(iterable)
        self._key_selector = key_selector
    
    def __do_iter__(self) -> Iterator[T]:
        seen = set()
        for e in self._iter:
            k = self._key_selector(e) if self._key_selector else e
            if k not in seen:
                seen.add(k)
                yield e


class Grouping(NamedTuple, Generic[K, T]):
    key: K
    values: Sequence[T]


class GroupingSequence(Sequence[Grouping[K, T]]):
    def __init__(self, iterable: Iterable[T], key_func: Callable[[T], K]) -> None:
        super().__init__(iterable)
        self._key_func = key_func
    
    @property
    def keys(self) -> Sequence[K]:
        return self.map(lambda x: x.key)
    
    @property
    def values(self) -> Sequence[Sequence[T]]:
        return self.map(lambda x: x.values)
    
    @property
    def items(self) -> Sequence[Grouping[K, T]]:
        return self
    
    def __do_iter__(self) -> Iterator[Grouping[K, T]]:
        from collections import defaultdict
        d = defaultdict(list)
        for e in self._iter:
            d[self._key_func(e)].append(e)
        yield from it(d.items()).map(lambda x: Grouping(x[0], it(x[1])))


class CombinationSequence(Sequence[Tuple[T, ...]]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        super().__init__(iterable)
        self._n = n
    
    def __do_iter__(self) -> Iterator[Tuple[T, ...]]:
        from itertools import combinations
        yield from combinations(self._iter, self._n)


class WindowedSequence(Sequence[List[T]]):
    def __init__(self, iterable: Iterable[T], size: int, step: int, partialWindows :bool) -> None:
        super().__init__(iterable)
        self._size = size
        self._step = step
        self._partialWindows = partialWindows
    
    def __do_iter__(self) -> Iterator[List[T]]:
        from collections import deque
        window = deque(maxlen=self._size)
        for e in self._iter:
            window.append(e)
            if len(window) == self._size:
                yield list(window)
            if len(window) == self._size:
                for _ in range(self._step):
                    window.popleft()
        
        if self._partialWindows:
            yield list(window)


class ConcatSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[Iterable[T]]) -> None:
        super().__init__(iterable)
    
    def __do_iter__(self) -> Iterator[T]:
        for i in self._iter:
            yield from i


class ShufflingSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], random: Optional[Union[Random, str, int]]=None) -> None:
        super().__init__(iterable)
        if random is None or isinstance(random, (str, int)):
            from random import Random
            self._random = Random(random)
        else:
            self._random = random
    
    def __do_iter__(self) -> Iterator[T]:
        lst = self._iter._cache.copy() if isinstance(self._iter, Sequence) and self._iter._cache is not None else list(self._iter)
        self._random.shuffle(lst)
        yield from lst


class ProgressSequence(Sequence[T]):
    def __init__(self, iterable: Sequence[T], progress_func: Callable[[Iterable[T]], Iterable[T]]) -> None:
        super().__init__(iterable)
        self._progress_func = progress_func
    
    @property
    def len(self) -> int:
        return len(self._iter)
    
    def __do_iter__(self) -> Iterator[T]:
        yield from self._progress_func(self._iter)
    
    def to_list(self) -> List[T]:
        progress_func = self._progress_func
        class ListLike(List[T]):
            def __init__(self, iterable: Iterable[T]) -> None:
                super().__init__(iterable)
        
            def __iter__(self) -> Iterator[T]:
                yield from progress_func(it(super().__iter__()))

        return ListLike(self._iter)

    
def throw(exception: Exception) -> None:
    raise exception


@overload
def sequence(elements: List[T]) -> Sequence[T]:
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
    ...
@overload
def sequence(elements: Iterable[T]) -> Sequence[T]:
    ...
@overload
def sequence(*elements: T) -> Sequence[T]:
    ...
def sequence(*iterable: Iterable[T]) -> Sequence[T]:
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
    if len(iterable) == 1 and isinstance(iterable[0], Iterable) and not isinstance(iterable[0], str):
        return Sequence(iterable[0])
    return Sequence(iterable)


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
