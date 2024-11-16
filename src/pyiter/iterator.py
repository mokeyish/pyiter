from typing import Iterator as IteratorBase, TypeVar

T = TypeVar("T")

class Iterator(IteratorBase[T]):

    def next(self) -> T:
        return next(self)

    def has_next(self) -> bool:
        try:
            peek
            next(self)
            return True
        except StopIteration:
            return False