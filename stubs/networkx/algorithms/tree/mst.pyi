from _typeshed import Incomplete
from enum import Enum

class EdgePartition(Enum):
    OPEN: int
    INCLUDED: int
    EXCLUDED: int

def minimum_spanning_edges(G, algorithm: str = ..., weight: str = ..., keys: bool = ..., data: bool = ..., ignore_nan: bool = ...): ...
def maximum_spanning_edges(G, algorithm: str = ..., weight: str = ..., keys: bool = ..., data: bool = ..., ignore_nan: bool = ...): ...
def minimum_spanning_tree(G, weight: str = ..., algorithm: str = ..., ignore_nan: bool = ...): ...
def partition_spanning_tree(G, minimum: bool = ..., weight: str = ..., partition: str = ..., ignore_nan: bool = ...): ...
def maximum_spanning_tree(G, weight: str = ..., algorithm: str = ..., ignore_nan: bool = ...): ...
def random_spanning_tree(G, weight: Incomplete | None = ..., *, multiplicative: bool = ..., seed: Incomplete | None = ...): ...

class SpanningTreeIterator:
    class Partition:
        mst_weight: float
        partition_dict: dict
        def __copy__(self): ...
        def __init__(self, mst_weight, partition_dict) -> None: ...
        def __lt__(self, other): ...
        def __gt__(self, other): ...
        def __le__(self, other): ...
        def __ge__(self, other): ...
    G: Incomplete
    weight: Incomplete
    minimum: Incomplete
    ignore_nan: Incomplete
    partition_key: str
    def __init__(self, G, weight: str = ..., minimum: bool = ..., ignore_nan: bool = ...) -> None: ...
    partition_queue: Incomplete
    def __iter__(self): ...
    def __next__(self): ...
