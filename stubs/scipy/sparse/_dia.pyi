from ._base import sparray
from ._data import _data_matrix
from ._matrix import spmatrix
from _typeshed import Incomplete

class _dia_base(_data_matrix):
    data: Incomplete
    offsets: Incomplete
    def __init__(self, arg1, shape: Incomplete | None = ..., dtype: Incomplete | None = ..., copy: bool = ...) -> None: ...
    def count_nonzero(self): ...
    def sum(self, axis: Incomplete | None = ..., dtype: Incomplete | None = ..., out: Incomplete | None = ...): ...
    def todia(self, copy: bool = ...): ...
    def transpose(self, axes: Incomplete | None = ..., copy: bool = ...): ...
    def diagonal(self, k: int = ...): ...
    def tocsc(self, copy: bool = ...): ...
    def tocoo(self, copy: bool = ...): ...
    def resize(self, *shape) -> None: ...

def isspmatrix_dia(x): ...

class dia_array(_dia_base, sparray): ...
class dia_matrix(spmatrix, _dia_base): ...