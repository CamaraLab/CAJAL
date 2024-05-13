import numpy as np
import pyarrow
from _typeshed import Incomplete
from pandas._libs import lib as lib
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, npt as npt, type_t as type_t
from pandas.core import ops as ops
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray, BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.common import is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype
from pandas.core.dtypes.dtypes import register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.missing import isna as isna

class BooleanDtype(BaseMaskedDtype):
    name: str
    @property
    def type(self) -> type: ...
    @property
    def kind(self) -> str: ...
    @property
    def numpy_dtype(self) -> np.dtype: ...
    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]: ...
    def __from_arrow__(self, array: Union[pyarrow.Array, pyarrow.ChunkedArray]) -> BooleanArray: ...

def coerce_to_array(values, mask: Incomplete | None = ..., copy: bool = ...) -> tuple[np.ndarray, np.ndarray]: ...

class BooleanArray(BaseMaskedArray):
    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = ...) -> None: ...
    @property
    def dtype(self) -> BooleanDtype: ...
