from _typeshed import Incomplete
from pandas.core.base import PandasObject as PandasObject
from pandas.io.formats.printing import pprint_thing as pprint_thing
from typing import Any

class FrozenList(PandasObject, list):
    def union(self, other) -> FrozenList: ...
    def difference(self, other) -> FrozenList: ...
    __add__: Incomplete
    __iadd__: Incomplete
    def __getitem__(self, n): ...
    def __radd__(self, other): ...
    def __eq__(self, other: Any) -> bool: ...
    __req__: Incomplete
    def __mul__(self, other): ...
    __imul__: Incomplete
    def __reduce__(self): ...
    def __hash__(self) -> int: ...
    __setitem__: Incomplete
    __setslice__: Incomplete
    __delitem__: Incomplete
    __delslice__: Incomplete
    pop: Incomplete
    append: Incomplete
    extend: Incomplete
    remove: Incomplete
    sort: Incomplete
    insert: Incomplete
