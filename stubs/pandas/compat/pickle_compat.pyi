import pickle as pkl
from pandas import DataFrame as DataFrame, Index as Index, Series as Series
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.tslibs import BaseOffset as BaseOffset
from pandas.core.arrays import DatetimeArray as DatetimeArray, PeriodArray as PeriodArray, TimedeltaArray as TimedeltaArray
from pandas.core.internals import BlockManager as BlockManager
from typing import Iterator

def load_reduce(self) -> None: ...

class _LoadSparseSeries:
    def __new__(cls) -> Series: ...

class _LoadSparseFrame:
    def __new__(cls) -> DataFrame: ...

class Unpickler(pkl._Unpickler):
    def find_class(self, module, name): ...

def load_newobj(self) -> None: ...
def load_newobj_ex(self) -> None: ...
def load(fh, encoding: Union[str, None] = ..., is_verbose: bool = ...): ...
def loads(bytes_object: bytes, *, fix_imports: bool = ..., encoding: str = ..., errors: str = ...): ...
def patch_pickle() -> Iterator[None]: ...
