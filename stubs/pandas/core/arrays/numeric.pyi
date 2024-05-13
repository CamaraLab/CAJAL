import numpy as np
import pyarrow
from pandas._libs import lib as lib
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, npt as npt
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray, BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import cache_readonly as cache_readonly
from typing import TypeVar

T = TypeVar('T', bound='NumericArray')

class NumericDtype(BaseMaskedDtype):
    def is_signed_integer(self) -> bool: ...
    def is_unsigned_integer(self) -> bool: ...
    def __from_arrow__(self, array: Union[pyarrow.Array, pyarrow.ChunkedArray]) -> BaseMaskedArray: ...

class NumericArray(BaseMaskedArray):
    def __init__(self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = ...) -> None: ...
    def dtype(self) -> NumericDtype: ...
