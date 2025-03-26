from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Iterator, List, Optional, TypeVar, Generator


T = TypeVar("T")

U = TypeVar("U")

O = TypeVar("O")

K = TypeVar("K")


class Transform(ABC, Generic[T, U], Iterable[U]):
    """A transform that applies a function to an iterable."""

    iter: Iterable[T]
    cache: Optional[List[U]]

    def __init__(self, iter: Iterable[T]):
        from .sequence import Sequence

        self.iter = iter.__transform__ if isinstance(iter, Sequence) else iter
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
    def __do_iter__(self) -> Iterator[U]:
        raise NotImplementedError

    def transforms(self) -> "Iterable[Transform[Any, Any]]":
        inner = self.iter
        if isinstance(inner, Transform):
            yield from inner.transforms()
        yield self


class NonTransform(Transform[T, T]):
    """
    A [Transform] that does not transform the values.
    """

    def __init__(self, iter: Iterable[T]) -> None:
        super().__init__(iter)
        self.cache = iter if isinstance(iter, list) else [*iter]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.iter.__class__.__name__}]"

    def __do_iter__(self) -> Iterator[T]:
        yield from self.iter

    def __len__(self) -> int:
        if not isinstance(self.iter, Transform):
            # not Sequence, just a wrapper of List, Tuple.etc.
            # we can get lenght of it directly.
            if hasattr(self.iter, "__len__"):
                return len(self.iter)  # type: ignore
            elif hasattr(self.iter, "__length_hint__"):
                return self.iter.__length_hint__()  # type: ignore
        return super().__len__()


class InfinityTransform(Transform[T, T]):
    """Transform that iterates over an infinite sequence."""

    __cache__: List[T]

    def __init__(self, iter: Iterable[T]) -> None:
        super().__init__(iter)
        self.__cache__ = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.iter.__class__.__name__}]"

    def __do_iter__(self) -> Iterator[T]:
        cache = self.__cache__
        yield from cache
        for x in self.iter:
            cache.append(x)
            yield x

    def __len__(self) -> int:
        if self.cache is not None:
            return len(self.cache)
        raise OverflowError("Cannot determine the length of an infinite sequence.")


def new_transform(iter: Iterable[T]) -> Transform[Any, T]:
    from .sequence import Sequence

    if isinstance(iter, Sequence):
        return iter.__transform__
    if isinstance(iter, Transform):
        return iter  # type: ignore

    if isinstance(iter, Generator):
        return InfinityTransform(iter)  # type: ignore

    return NonTransform(iter)
