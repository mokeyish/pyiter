from __future__ import annotations
from typing import Dict, Generic, Iterable, Iterator, Union, List, Set, Optional, Tuple, Type, TypeVar, Callable, overload, Literal

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

    def __init__(self, v: Iterable[T]) -> None:
        super().__init__()
        self._iter = v

    def filter(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
        Returns a Sequence containing only elements matching the given [predicate].

         Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter(lambda x: x.startswith('a'))
        ['a1', 'a2']
        """
        return FilteringSequence(self, predicate)

    def filter_indexed(self, predicate: Callable[[T, int], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing only elements matching the given [predicate].

         Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter_indexed(lambda x, i: i == 2)
        ['b2']
        """
        return IndexingSequence(self).filter(lambda x: predicate(x.value, x.index)).map(lambda x: x.value)

    def filter_is_instance(self, r_type: Type[R]) -> Sequence[R]:
        """
         Returns a Sequence containing all elements that are instances of specified type parameter r_type.

        Example 1:
        >>> lst = [ 'a1', 1, 'b2', 3]
        >>> it(lst).filter_is_instance(int)
        [1, 3]

        """
        return self.filter(lambda x: type(x) == r_type)

    def filter_not(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing all elements not matching the given [predicate].

         Example 1:
        >>> lst = [ 'a1', 'b1', 'b2', 'a2']
        >>> it(lst).filter_not(lambda x: x.startswith('a'))
        ['b1', 'b2']
        """
        return self.filter(lambda x: not predicate(x))

    def filter_not_none(self) -> Sequence[T]:
        """
         Returns a Sequence containing all elements that are not `None`.

         Example 1:
        >>> lst = [ 'a', None, 'b']
        >>> it(lst).filter_not_none()
        ['a', 'b']
        """
        return self.filter(lambda x: x is not None)

    def map(self, transform: Callable[[T], R]) -> Sequence[R]:
        """
         Returns a Sequence containing the results of applying the given [transform] function
         to each element in the original Sequence.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).map(lambda x: x['age'])
        [12, 13]
        """
        return MappingSequence(self, transform)
    

    def map_indexed(self, transform: Callable[[T, int], R]) -> Sequence[R]:
        """
         Returns a Sequence containing the results of applying the given [transform] function
         to each element in the original Sequence.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).map_indexed(lambda x, i: x['age'] + i)
        [12, 14]
        """
        return IndexingSequence(self).map(lambda x: transform(x.value, x.index))

    
    def map_not_none(self, transform: Callable[[T], Optional[R]]) -> Sequence[R]:
        """
         Returns a Sequence containing only the non-none results of applying the given [transform] function
        to each element in the original collection.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': None}]
        >>> it(lst).map_not_none(lambda x: x['age'])
        [12]
        """
        return self.map(transform).filter_not_none()
    

    def paralell_map(self, transform: Callable[[T], R]) -> Sequence[R]:
        """
         Returns a Sequence containing the results of applying the given [transform] function
         to each element in the original Sequence.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': 12}, { 'name': 'B', 'age': 13}]
        >>> it(lst).paralell_map(lambda x: x['age'])
        [12, 13]
        """
        return ParallelMappingSequence(self, transform)

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
        ...

    @overload
    def first(self, predicate: Callable[[T], bool]) -> T:
        ...

    def first(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
        for e in self:
            if predicate is None or predicate(e):
                return e
        raise ValueError("Sequence is empty.")
    
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
        v = self.first_not_null_of_or_none(transform)
        if v is None:
            raise ValueError('No element of the Sequence was transformed to a non-none value.')
        return v
    
    def first_not_null_of_or_none(self, transform: Callable[[T], Optional[R]]) -> Optional[R]:
        """
         Returns the first non-`None` result of applying the given [transform] function to each element in the original collection.

         Example 1:
        >>> lst = [{ 'name': 'A', 'age': None}, { 'name': 'B', 'age': 12}]
        >>> it(lst).first_not_null_of_or_none(lambda x: x['age'])
        12

         Example 2:
        >>> lst = [{ 'name': 'A', 'age': None}, { 'name': 'B', 'age': None}]
        >>> it(lst).first_not_null_of_or_none(lambda x: x['age']) is None
        True
        """
        return self.map_not_none(transform).first_or_none()

    @overload
    def first_or_none(self) -> Optional[T]:
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
        ...

    @overload
    def first_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...

    def first_or_none(self, predicate: Optional[Callable[[T], bool]] = None) -> Optional[T]:
        return next(iter(self if predicate is None else self.filter(predicate)), None)


    @overload
    def first_or_default(self, default: T) -> T :
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
        ...

    @overload
    def first_or_default(self, predicate: Callable[[T], bool], default: T) -> T :
        ...
    
    def first_or_default(self, predicate: Union[Callable[[T], bool], T], default: T) -> T:
        if isinstance(predicate, Callable):
            return self.first_or_none(predicate) or default
        else:
            default = predicate
        return self.first_or_none() or default

    @overload
    def last(self) -> T:
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
        ...

    @overload
    def last(self, predicate: Callable[[T], bool]) -> T:
        ...

    def last(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
        v = self.last_or_none(predicate)
        if v is None:
            raise ValueError('Sequence is empty.')
        return v

    @overload
    def last_or_none(self) -> Optional[T]:
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
        ...

    @overload
    def last_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...

    def last_or_none(self, predicate: Optional[Callable[[T], bool]] = None) -> Optional[T]:
        last: Optional[T] = None
        for i in self if predicate is None else self.filter(predicate):
            last = i
        return last

    @overload
    def single(self) -> T:
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
        ...

    @overload
    def single(self, predicate: Callable[[T], bool]) -> T:
        ...

    def single(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
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
        ...

    @overload
    def single_or_none(self, predicate: Callable[[T], bool]) -> Optional[T]:
        ...

    def single_or_none(self, predicate: Optional[Callable[[T], bool]] = None) -> Optional[T]:
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
        >>> it(lst).drop(0)
        ['a', 'b', 'c']

         Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop(1)
        ['b', 'c']


         Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).drop(4)
        []
        """
        if n < 0:
            raise ValueError(f'Requested element count {n} is less than zero.')
        if n == 0:
            return self

        return DropSequence(self, n)

    # noinspection PyShadowingNames
    def drop_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing all elements except first elements that satisfy the given [predicate].

         Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).drop_while(lambda x: x < 3 )
        [3, 4, 1]
        """
        return DropWhileSequence(self, predicate)
    
    def skip(self, n: int) -> Sequence[T]:
        """
         Returns a Sequence containing all elements except first [n] elements.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).skip(0)
        ['a', 'b', 'c']

         Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).skip(1)
        ['b', 'c']


         Example 2:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).skip(4)
        []
        """
        return self.drop(n)
    
    def skip_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns a Sequence containing all elements except first elements that satisfy the given [predicate].

         Example 1:
        >>> lst = [1, 2, 3, 4, 1]
        >>> it(lst).skip_while(lambda x: x < 3 )
        [3, 4, 1]
        """
        return self.drop_while(predicate)


    def take(self, n: int) -> Sequence[T]:
        """
         Returns an Sequence containing first [n] elements.

         Example 1:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take(0)
        []


         Example 2:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take(2)
        ['a', 'b']

        """
        if n < 0:
            raise ValueError(f'Requested element count {n} is less than zero.')
        if n == 0:
            return Sequence([])
        return TakeSequence(self, n)

    # noinspection PyShadowingNames
    def take_while(self, predicate: Callable[[T], bool]) -> Sequence[T]:
        """
         Returns an Sequence containing first elements satisfying the given [predicate].

         Example 1:
        >>> lst = ['a', 'b', 'c', 'd']
        >>> it(lst).take_while(lambda x: x in ['a', 'b'])
        ['a', 'b']

        """
        return TakeWhileSequence(self, predicate)
    
    def take_last(self, n: int) -> Sequence[T]:
        """
         Returns an Sequence containing last [n] elements.

         Example 1:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(0)
        []


         Example 2:
        >>> a = ['a', 'b', 'c']
        >>> it(a).take_last(2)
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
        >>> it(lst).sorted()
        ['a', 'b', 'c', 'e']

         Example 2:
        >>> lst = [2, 1, 4, 3]
        >>> it(lst).sorted()
        [1, 2, 3, 4]
        """
        lst = list(self)
        lst.sort()
        return it(lst)

    # noinspection PyShadowingNames
    def sorted_by(self, key_selector: Callable[[T], R]) -> Sequence[T]:
        """
         Returns a sequence that yields elements of this sequence sorted according to natural sort
        order of the value returned by specified [key_selector] function.

         Example 1:
        >>> lst = [ {'name': 'A', 'age': 12 }, {'name': 'C', 'age': 10 }, {'name': 'B', 'age': 11 } ]
        >>> it(lst).sorted_by(lambda x: x['name'])
        [{'name': 'A', 'age': 12}, {'name': 'B', 'age': 11}, {'name': 'C', 'age': 10}]
        >>> it(lst).sorted_by(lambda x: x['age'])
        [{'name': 'C', 'age': 10}, {'name': 'B', 'age': 11}, {'name': 'A', 'age': 12}]

        """
        lst = list(self)
        lst.sort(key=key_selector)
        return it(lst)

    def sorted_descending(self) -> Sequence[T]:
        """
         Returns a Sequence of all elements sorted descending according to their natural sort order.

         Example 1:
        >>> lst = ['b', 'c', 'a']
        >>> it(lst).sorted_descending()
        ['c', 'b', 'a']
        """
        return self.sorted().reversed()

    def sorted_by_descending(self, key_selector: Callable[[T], R]) -> Sequence[T]:
        """
         Returns a sequence that yields elements of this sequence sorted descending according
        to natural sort order of the value returned by specified [key_selector] function.

         Example 1:
        >>> lst = [ {'name': 'A', 'age': 12 }, {'name': 'C', 'age': 10 }, {'name': 'B', 'age': 11 } ]
        >>> it(lst).sorted_by_descending(lambda x: x['name'])
        [{'name': 'C', 'age': 10}, {'name': 'B', 'age': 11}, {'name': 'A', 'age': 12}]
        >>> it(lst).sorted_by_descending(lambda x: x['age'])
        [{'name': 'A', 'age': 12}, {'name': 'B', 'age': 11}, {'name': 'C', 'age': 10}]
        """
        return self.sorted_by(key_selector).reversed()

    # noinspection PyShadowingNames
    def sorted_with(self, comparator: Callable[[T, T], int]) -> Sequence[T]:
        """
         Returns a sequence that yields elements of this sequence sorted according to the specified [comparator].

        Example 1:
        >>> lst = ['aa', 'bbb', 'c']
        >>> it(lst).sorted_with(lambda a, b: len(a)-len(b))
        ['c', 'aa', 'bbb']
        """
        from functools import cmp_to_key
        lst = list(self)
        lst.sort(key=cmp_to_key(comparator))
        return it(lst)

    def associate(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]:
        """
         Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

         Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).associate(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}

        """
        dic = dict()
        for i in self:
            k, v = transform(i)
            dic[k] = v
        return dic

    @overload
    def associate_by(self, key_selector: Callable[[T], K]) -> Dict[K, T]:
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
        ...

    @overload
    def associate_by(self, key_selector: Callable[[T], K], value_transform: Callable[[T], V]) -> Dict[K, V]:
        ...

    def associate_by(self, key_selector: Callable[[T], K],
                     value_transform: Optional[Callable[[T], V]] = None) -> Dict[K, Union[V, T]]:
        dic = dict()
        for i in self:
            k = key_selector(i)
            dic[k] = i if value_transform is None else value_transform(i)
        return dic

    @overload
    def associate_by_to(self, destination: Dict[K, T], key_selector: Callable[[T], K]) -> Dict[K, T]:
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
        ...

    @overload
    def associate_by_to(self, destination: Dict[K, V], key_selector: Callable[[T], K],
                        value_transform: Callable[[T], V]) -> Dict[K, V]:
        ...

    def associate_by_to(self, destination: Dict[K, V], key_selector: Callable[[T], K],
                        value_transform: Optional[Callable[[T], V]] = None) -> Dict[K, Union[V, T]]:
        for i in self:
            k = key_selector(i)
            destination[k] = i if value_transform is None else value_transform(i)
        return destination
    
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
        for i in self:
            if not predicate(i):
                return False
        return True
    
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
        for i in self:
            if predicate(i):
                return True
        return False
    
    @overload
    def count(self) -> int:
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
        ...
    @overload
    def count(self, predicate: Callable[[T], bool]) -> int:
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
        >>> it(lst).distinct()
        [1, 2, 3]

        """
        return Sequence(set(self))
    
    def distinct_by(self, key_selector: Callable[[T], K]) -> Sequence[T]:
        """
         Returns a new Sequence containing the distinct elements of the given Sequence.

         Example 1:
        >>> lst = [1, 2, 3, 1, 2, 3]
        >>> it(lst).distinct_by(lambda x: x%2)
        [3, 2]

        """
        return Sequence(self.associate_by(key_selector).values())
    
    def reduce(self, accumulator: Callable[[T, T], T], initial: Optional[T] = None) -> T:
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
    
    def fold(self, initial: R, accumulator: Callable[[R, T], T]) -> R:
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
        """
         Returns the sum of the elements of the given Sequence.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).sum_of(lambda x: x)
        6

        """
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
        """
         Returns the maximum element of the given Sequence.

         Example 1:
        >>> lst = [1, 2, 3]
        >>> it(lst).max_of(lambda x: x)
        3

        """
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
        >>> it(lst).reversed()
        ['a', 'c', 'b']
        """
        lst = list(self)
        lst.reverse()
        return it(lst)

    def flat_map(self, transform: Callable[[T], Iterable[R]]) -> Sequence[R]:
        """
         Returns a single list of all elements yielded from results of [transform]
        function being invoked on each element of original collection.

         Example 1:
        >>> lst = [['a', 'b'], ['c'], ['d', 'e']]
        >>> it(lst).flat_map(lambda x: x)
        ['a', 'b', 'c', 'd', 'e']
        """
        return FlatteningSequence(self, transform)
    

    def group_by(self, key_selector: Callable[[T], K]) -> GroupingSequence[K, T]:
        """
         Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

         Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by(lambda x: x%2)
        [(1, [1, 3, 5]), (0, [2, 4])]
        """
        return GroupingSequence(self, key_selector)
    

    def group_by_to(self, destination: Dict[K, List[T]], key_selector: Callable[[T], K]) -> Dict[K, List[T]]:
        """
         Returns a dictionary with keys being the result of [key_selector] function being invoked on each element of original collection
        and values being the corresponding elements of original collection.

         Example 1:
        >>> lst = [1, 2, 3, 4, 5]
        >>> it(lst).group_by_to({}, lambda x: x%2)
        {1: [1, 3, 5], 0: [2, 4]}
        """
        for e in self:
            k = key_selector(e)
            if k not in destination:
                destination[k] = []
            destination[k].append(e)
        return destination

    def foreach(self, action: Callable[[T], None]) -> None:
        """
         Invokes [action] function on each element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).foreach(lambda x: print(x))
        a
        b
        c
        """
        self.on_each(action)
    
    def parallel_foreach(self, action: Callable[[T], None]) -> None:
        """
         Invokes [action] function on each element of the given Sequence in parallel.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).parallel_foreach(lambda x: print(x))
        a
        b
        c
        """
        self.parallel_on_each(action)
    
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
        for i in self:
            action(i)
        return self
    
    def parallel_on_each(self, action: Callable[[T], None]) -> Sequence[T]:
        """
         Invokes [action] function on each element of the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).parallel_on_each(lambda x: print(x)) and None
        a
        b
        c
        """
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            for i in self:
                executor.submit(action, i)
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
        >>> it(lst1).zip(lst2)
        [('a', 1), ('b', 2), ('c', 3)]

         Example 2:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> it(lst1).zip(lst2, lambda x, y: x + '__' +str( y))
        ['a__1', 'b__2', 'c__3']
        """
        ...
    @overload
    def zip(self, other: Iterable[R], transform: Callable[[T, R], V]) -> Sequence[V]:
        ...
    def zip(self, other: Iterable[R], transform: Callable[[T, R], V] = lambda a, b:(a, b)) -> Sequence[V]:
        return MergingSequence(self, other, transform)
    
    def partition(self, predicate: Callable[[T], bool]) -> Tuple[Sequence[T], Sequence[T]]:
        """
         Partitions the elements of the given Sequence into two groups,
         the first group containing the elements for which the predicate returns true,
         and the second containing the rest.

         Example 1:
        >>> lst = ['a', 'b', 'c', '2']
        >>> it(lst).partition(lambda x: x.isalpha())
        (['a', 'b', 'c'], ['2'])
        """
        return self.filter(predicate), self.filter(lambda x: not predicate(x))
    

    @overload
    def combinations(self, n: Literal[2]) -> Sequence[Tuple[T, T]]: 
        """
         Returns a Sequence of all possible combinations of size [n] from the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c']
        >>> it(lst).combinations(2)
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
        >>> it(lst).combinations(2)
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
    

    def windowed(self, size: int, step: int = 1, partialWindows: bool = False) -> Sequence[Sequence[T]]:
        """
         Returns a Sequence of all possible sliding windows of size [size] from the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).windowed(3)
        [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]

         Example 2:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).windowed(3, 2)
        [['a', 'b', 'c'], ['c', 'd', 'e']]

         Example 3:
        >>> lst = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> it(lst).windowed(3, 2, True)
        [['a', 'b', 'c'], ['c', 'd', 'e'], ['e', 'f']]
        """
        return WindowedSequence(self, size, step, partialWindows)
    

    def chunked(self, size: int) -> Sequence[Sequence[T]]:
        """
         Returns a Sequence of all possible chunks of size [size] from the given Sequence.

         Example 1:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> it(lst).chunked(3)
        [['a', 'b', 'c'], ['d', 'e']]
        """
        return self.windowed(size, size, True)

    
    def concat(self, *other: Sequence[T]) -> Sequence[T]:
        """
         Returns a Sequence of all elements of the given Sequence, followed by all elements of the given Sequence.

         Example 1:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> it(lst1).concat(lst2)
        ['a', 'b', 'c', 1, 2, 3]

         Example 2:
        >>> lst1 = ['a', 'b', 'c']
        >>> lst2 = [1, 2, 3]
        >>> lst3 = [4, 5, 6]
        >>> it(lst1).concat(lst2, lst3)
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

    def to_set(self) -> Set[T]:
        """
         Returns a set containing all elements of this Sequence.

         Example 1:
        >>> it(['a', 'b', 'c', 'c']).to_set() == {'a', 'b', 'c'}
        True
        """
        return set(self)

    def to_dict(self, transform: Callable[[T], Tuple[K, V]]) -> Dict[K, V]:
        """
         Returns a [Dict] containing key-value Tuple provided by [transform] function
        applied to elements of the given Sequence.

         Example 1:
        >>> lst = ['1', '2', '3']
        >>> it(lst).to_dict(lambda x: (int(x), x))
        {1: '1', 2: '2', 3: '3'}

        """
        return self.associate(transform)

    def to_list(self) -> List[T]:
        """
         Returns a list with elements of the given Sequence.

         Example 1:
        >>> it(['b', 'c', 'a']).to_list()
        ['b', 'c', 'a']
        """
        return list(self)
    
    def __len__(self) -> int:
        return len(self._iter)

    def __repr__(self) -> str:
        return str(self.to_list())

    def __iter__(self) -> Iterator[T]:
        return iter(self._iter)


class FilteringSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        self._iter = iterable
        self._predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        for i in self._iter:
            if self._predicate(i):
                yield i


class MappingSequence(Sequence[R]):
    def __init__(self, iterable: Iterable[T], transform: Callable[[T], R]) -> None:
        self._iter = iterable
        self._transform = transform
    
    def __iter__(self) -> Iterator[R]:
        for i in self._iter:
            yield self._transform(i)


class ParallelMappingSequence(Sequence[R]):
    def __init__(self, iterable: Iterable[Sequence[T]], transformer: Callable[[T], R] , max_workers: Optional[int]=None) -> None:
        self._iter = iterable
        self._transformer = transformer
        self._max_workers = max_workers
    
    def __iter__(self) -> Iterator[R]:
        from concurrent.futures import ThreadPoolExecutor
        from multiprocessing import cpu_count
        size = len(self._iter)
        max_workers = cpu_count() -1 if self._max_workers is None else self._max_workers
        
        with ThreadPoolExecutor(max_workers) as executor:
            yield from executor.map(self._transformer, self._iter, chunksize=size//max_workers)

class IndexedValue(Generic[T]):
    def __init__(self, index: int, value: T) -> None:
        self.index = index
        self.value = value
    
    def __repr__(self) -> str:
        return f'IndexedValue({self.index}, {self.value})'


class IndexingSequence(Sequence[IndexedValue[T]]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._iter = iterable
    
    def __iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            yield IndexedValue(i, e)


class FlatteningSequence(Sequence[R]):
    def __init__(self, iterable: Iterable[Sequence[T]], transformer: Callable[[T], R]) -> None:
        self._iter = iterable
        self._transformer = transformer
    
    def __iter__(self) -> Iterator[R]:
        for i in self._iter:
            yield from self._transformer(i)


class DropSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        self._iter = iterable
        self._n = n
        
    def __iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            if i < self._n:
                continue
            yield e


class DropWhileSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        self._iter = iterable
        self._predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        drop_state = True
        for e in self._iter:
            if drop_state and self._predicate(e):
                continue
            else:
                drop_state = False
            yield e


class TakeSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        self._iter = iterable
        self._n = n
    
    def __iter__(self) -> Iterator[T]:
        for i, e in enumerate(self._iter):
            if i >= self._n:
                break
            yield e


class TakeWhileSequence(Sequence[T]):
    def __init__(self, iterable: Iterable[T], predicate: Callable[[T], bool]) -> None:
        self._iter = iterable
        self._predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
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
        self._iter = iterable
        self._other = other
        self._transformer = transformer
    
    def __iter__(self) -> Iterator[V]:
        iter1 = iter(self._iter)
        iter2 = iter(self._other)
        while True:
            try:
                yield self._transformer(next(iter1), next(iter2))
            except StopIteration:
                break


class GroupingSequence(Sequence[Tuple[K, List[T]]]):
    def __init__(self, iterable: Iterable[T], key_func: Callable[[T], K]) -> None:
        self._iter = iterable
        self._key_func = key_func
    
    @property
    def keys(self) -> Sequence[K]:
        return self.map(lambda x: x[0])
    
    @property
    def values(self) -> Sequence[List[T]]:
        return self.map(lambda x: x[1])
    
    @property
    def items(self) -> Sequence[Tuple[K, List[T]]]:
        return self
    
    def __iter__(self) -> Iterator[Tuple[K, List[T]]]:
        from collections import defaultdict
        d = defaultdict(list)
        for e in self._iter:
            d[self._key_func(e)].append(e)
        yield from d.items()



class CombinationSequence(Sequence[Tuple[T, ...]]):
    def __init__(self, iterable: Iterable[T], n: int) -> None:
        self._iter = iterable
        self._n = n
    
    def __iter__(self) -> Iterator[Tuple[T, ...]]:
        from itertools import combinations
        yield from combinations(self._iter, self._n)


class WindowedSequence(Sequence[List[T]]):
    def __init__(self, iterable: Iterable[T], size: int, step: int, partialWindows :bool) -> None:
        self._iter = iterable
        self._size = size
        self._step = step
        self._partialWindows = partialWindows
    
    def __iter__(self) -> Iterator[List[T]]:
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
        self._iter = iterable
    
    def __iter__(self) -> Iterator[T]:
        for i in self._iter:
            yield from i



def throw(exception: Exception) -> None:
    raise exception


@overload
def sequence(*elements: T) -> Sequence[T]:
    """
     Creates an iterator from a list of elements or given Iterable.
    
     Example 1:
    >>> sequence('hello', 'world').map(lambda x: x.upper())
    ['HELLO', 'WORLD']

     Example 2:
    >>> sequence(['hello', 'world']).map(lambda x: x.upper())
    ['HELLO', 'WORLD']

     Example 3:
    >>> sequence(range(10)).map(lambda x: x*x)
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    """
    ...
@overload
def sequence(elements: Iterable[T]) -> Sequence[T]:
    ...
def sequence(*iterable: Iterable[T]) -> Sequence[T]:
    """
     Creates an iterator from a list of elements or given Iterable.
    
     Example 1:
    >>> sequence('hello', 'world').map(lambda x: x.upper())
    ['HELLO', 'WORLD']

     Example 2:
    >>> sequence(['hello', 'world']).map(lambda x: x.upper())
    ['HELLO', 'WORLD']

     Example 3:
    >>> sequence(range(10)).map(lambda x: x*x)
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    """
    if len(iterable) == 1 and isinstance(iterable[0], Iterable):
        return Sequence(iterable[0])
    return Sequence(iterable)


seq = sequence
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> seq('hello', 'world').map(lambda x: x.upper())
['HELLO', 'WORLD']

    Example 2:
>>> seq(['hello', 'world']).map(lambda x: x.upper())
['HELLO', 'WORLD']

    Example 3:
>>> seq(range(10)).map(lambda x: x*x)
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""


iterate = sequence
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> iterate('hello', 'world').map(lambda x: x.upper())
['HELLO', 'WORLD']

    Example 2:
>>> iterate(['hello', 'world']).map(lambda x: x.upper())
['HELLO', 'WORLD']

    Example 3:
>>> iterate(range(10)).map(lambda x: x*x)
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""


it = sequence
"""
    Creates an iterator from a list of elements or given Iterable.

    Example 1:
>>> it('hello', 'world').map(lambda x: x.upper())
['HELLO', 'WORLD']

    Example 2:
>>> it(['hello', 'world']).map(lambda x: x.upper())
['HELLO', 'WORLD']

    Example 3:
>>> it(range(10)).map(lambda x: x*x)
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
"""


if __name__ == "__main__":
    import doctest

    doctest.testmod()
