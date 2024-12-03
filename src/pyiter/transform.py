from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Iterator, List, Optional, TypeVar


T = TypeVar("T")

U = TypeVar("U")

O = TypeVar("O")

K = TypeVar("K")


class Transform(ABC, Generic[T, U], Iterable[U]):
    """A transform that applies a function to an iterable."""
    iter: Iterable[T]
    cache: Optional[List[U]]
    def __init__(self, iter: Iterable[T]):
        self.iter = iter
        self.cache = None
    
    def __iter__(self) -> Iterator[U]:
        if self.cache:
            yield from self.cache
        else:
            cache: List[U] = []
            for x in self.__do_iter__():
                cache.append(x)
                yield x
            self.cache = cache

    def __len__(self) -> int:
        # if not isinstance(self.iter, Transform):
        #     # not Sequence, just a wrapper of List, Tuple.etc.
        #     # we can get lenght of it directly.
        #     if hasattr(self.iter, '__len__'):
        #         return len(self.iter) # type: ignore
        #     elif hasattr(self.iter, '__length_hint__'):
        #         return self._iter.__length_hint__() # type: ignore
        # we need iterate all to get length
        cache = self.cache
        if cache is None:
            for _ in self:
                pass
            cache = self.cache
        if cache is not None:
            return len(cache)
        return 0
    
    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __do_iter__(self) -> Iterator[U] :
        raise NotImplementedError
    
    def transforms(self) -> "Iterable[Transform[Any, Any]]":
        from .sequence import Sequence
        inner = self.iter
        if isinstance(inner, Sequence):
            yield from inner.__transform__.transforms()
        elif isinstance(inner, Transform):
            yield from inner.transforms()
        yield self

    

    


class NonTransform(Transform[T, T]):
    """
    A [Transform] that does not transform the values.
    """
    def __init__(self, iter: Iterable[T]) -> None:
        super().__init__(iter)
    
    def __do_iter__(self) -> Iterator[T]:
        yield from self.iter

    def __len__(self) -> int:
        if not isinstance(self.iter, Transform):
            # not Sequence, just a wrapper of List, Tuple.etc.
            # we can get lenght of it directly.
            if hasattr(self.iter, '__len__'):
                return len(self.iter) # type: ignore
            elif hasattr(self.iter, '__length_hint__'):
                return self.iter.__length_hint__()  # type: ignore
        return super().__len__()