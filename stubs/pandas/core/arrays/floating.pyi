from _typeshed import Incomplete
from pandas.core.arrays.numeric import NumericArray as NumericArray, NumericDtype as NumericDtype
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype
from pandas.core.dtypes.dtypes import register_extension_dtype as register_extension_dtype

class FloatingDtype(NumericDtype):
    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]: ...

class FloatingArray(NumericArray): ...

class Float32Dtype(FloatingDtype):
    type: Incomplete
    name: str
    __doc__: Incomplete

class Float64Dtype(FloatingDtype):
    type: Incomplete
    name: str
    __doc__: Incomplete

FLOAT_STR_TO_DTYPE: Incomplete
