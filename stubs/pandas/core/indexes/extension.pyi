from _typeshed import Incomplete
from pandas._typing import ArrayLike as ArrayLike, npt as npt
from pandas.core.arrays import IntervalArray as IntervalArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame
from pandas.core.indexes.base import Index as Index
from pandas.util._decorators import cache_readonly as cache_readonly, doc as doc
from typing import Callable

def inherit_names(names: list[str], delegate: type, cache: bool = ..., wrap: bool = ...) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]: ...

class ExtensionIndex(Index):
    def map(self, mapper, na_action: Incomplete | None = ...): ...

class NDArrayBackedExtensionIndex(ExtensionIndex): ...
