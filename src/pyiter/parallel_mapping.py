from __future__ import annotations
from typing import Callable, Iterable, Iterator, Literal, Optional
from .transform import Transform, T, U


class ParallelMappingTransform(Transform[T, U]):
    """A transform that applies a function to each element of an iterable in parallel."""
    
    Executor = Literal['Thread', 'Process']
    def __init__(
            self, 
            iter: Iterable[T], 
            transform: Callable[[T], U], 
            max_workers: Optional[int]=None, 
            chunksize: int=1,
            executor: ParallelMappingTransform.Executor = 'Thread'):
        super().__init__(iter)
        self.transform = transform
        self.max_workers = max_workers
        self.executor = executor
        self.chunksize = chunksize

    def __do_iter__(self) -> Iterator[U]:
        import os
        from .sequence import it

        def create_executor(max_workers: int):
            if self.executor == 'Process':
                from concurrent.futures import ProcessPoolExecutor
                return ProcessPoolExecutor(max_workers=max_workers)
            else:
                from concurrent.futures import ThreadPoolExecutor
                return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='PyIter worker')

        chunksize = self.chunksize
        max_workers = self.max_workers or min(32, (os.cpu_count() or 1) + 4)
        batch_size = max_workers * chunksize

        for batch in it(self.iter).chunked(batch_size):
            with create_executor(max_workers) as executor:
                yield from executor.map(self.transform, batch, chunksize=chunksize)
