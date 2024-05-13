import numpy as np
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from pandas._libs import lib as lib
from pandas._typing import Dtype as Dtype, NpDtype as NpDtype, Scalar as Scalar, npt as npt
from pandas.compat import pa_version_under1p01 as pa_version_under1p01, pa_version_under2p0 as pa_version_under2p0, pa_version_under3p0 as pa_version_under3p0, pa_version_under4p0 as pa_version_under4p0
from pandas.core.arrays.arrow import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.arrow._arrow_utils import fallback_performancewarning as fallback_performancewarning
from pandas.core.arrays.boolean import BooleanDtype as BooleanDtype
from pandas.core.arrays.integer import Int64Dtype as Int64Dtype
from pandas.core.arrays.numeric import NumericDtype as NumericDtype
from pandas.core.arrays.string_ import BaseStringArray as BaseStringArray, StringDtype as StringDtype
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_dtype_equal as is_dtype_equal, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.missing import isna as isna
from pandas.core.strings.object_array import ObjectStringArrayMixin as ObjectStringArrayMixin

ArrowStringScalarOrNAT: Incomplete

class ArrowStringArray(ArrowExtensionArray, BaseStringArray, ObjectStringArrayMixin):
    def __init__(self, values) -> None: ...
    @property
    def dtype(self) -> StringDtype: ...
    def __array__(self, dtype: Union[NpDtype, None] = ...) -> np.ndarray: ...
    def to_numpy(self, dtype: Union[npt.DTypeLike, None] = ..., copy: bool = ..., na_value=...) -> np.ndarray: ...
    def insert(self, loc: int, item) -> ArrowStringArray: ...
    def isin(self, values) -> npt.NDArray[np.bool_]: ...
    def astype(self, dtype, copy: bool = ...): ...
