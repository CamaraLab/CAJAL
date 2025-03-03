import pyarrow
from _typeshed import Incomplete
from pandas import Series as Series
from pandas._config import get_option as get_option
from pandas._libs import lib as lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._typing import Dtype as Dtype, Scalar as Scalar, npt as npt, type_t as type_t
from pandas.compat import pa_version_under1p01 as pa_version_under1p01
from pandas.core import ops as ops
from pandas.core.array_algos import masked_reductions as masked_reductions
from pandas.core.arrays import ExtensionArray as ExtensionArray, FloatingArray as FloatingArray, IntegerArray as IntegerArray
from pandas.core.arrays.floating import FloatingDtype as FloatingDtype
from pandas.core.arrays.integer import IntegerDtype as IntegerDtype
from pandas.core.arrays.numpy_ import PandasArray as PandasArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype, StorageExtensionDtype as StorageExtensionDtype, register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.common import is_array_like as is_array_like, is_bool_dtype as is_bool_dtype, is_dtype_equal as is_dtype_equal, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.indexers import check_array_indexer as check_array_indexer
from pandas.core.missing import isna as isna

class StringDtype(StorageExtensionDtype):
    name: str
    @property
    def na_value(self) -> libmissing.NAType: ...
    storage: Incomplete
    def __init__(self, storage: Incomplete | None = ...) -> None: ...
    @property
    def type(self) -> type[str]: ...
    @classmethod
    def construct_from_string(cls, string): ...
    def construct_array_type(self) -> type_t[BaseStringArray]: ...
    def __from_arrow__(self, array: Union[pyarrow.Array, pyarrow.ChunkedArray]) -> BaseStringArray: ...

class BaseStringArray(ExtensionArray): ...

class StringArray(BaseStringArray, PandasArray):
    def __init__(self, values, copy: bool = ...) -> None: ...
    def __arrow_array__(self, type: Incomplete | None = ...): ...
    def __setitem__(self, key, value) -> None: ...
    def astype(self, dtype, copy: bool = ...): ...
    def min(self, axis: Incomplete | None = ..., skipna: bool = ..., **kwargs) -> Scalar: ...
    def max(self, axis: Incomplete | None = ..., skipna: bool = ..., **kwargs) -> Scalar: ...
    def value_counts(self, dropna: bool = ...) -> Series: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
