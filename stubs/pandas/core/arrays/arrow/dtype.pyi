import numpy as np
import pyarrow as pa
from _typeshed import Incomplete
from pandas._typing import DtypeObj as DtypeObj
from pandas.compat import pa_version_under1p01 as pa_version_under1p01
from pandas.core.dtypes.base import StorageExtensionDtype as StorageExtensionDtype, register_extension_dtype as register_extension_dtype
from pandas.util._decorators import cache_readonly as cache_readonly

class ArrowDtype(StorageExtensionDtype):
    pyarrow_dtype: Incomplete
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
    @property
    def type(self): ...
    @property
    def name(self) -> str: ...
    def numpy_dtype(self) -> np.dtype: ...
    def kind(self) -> str: ...
    def itemsize(self) -> int: ...
    @classmethod
    def construct_array_type(cls): ...
    @classmethod
    def construct_from_string(cls, string: str) -> ArrowDtype: ...
    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]): ...