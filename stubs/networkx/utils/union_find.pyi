from _typeshed import Incomplete
from networkx.utils import groups as groups

class UnionFind:
    parents: Incomplete
    weights: Incomplete
    def __init__(self, elements: Incomplete | None = ...) -> None: ...
    def __getitem__(self, object): ...
    def __iter__(self): ...
    def to_sets(self) -> None: ...
    def union(self, *objects): ...