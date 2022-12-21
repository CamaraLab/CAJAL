import numpy as np
from pandas._libs import lib as lib
from pandas._libs.tslibs import is_unitless as is_unitless
from pandas._libs.tslibs.timedeltas import array_to_timedelta64 as array_to_timedelta64
from pandas._typing import ArrayLike as ArrayLike, DtypeObj as DtypeObj, IgnoreRaise as IgnoreRaise
from pandas.core.arrays import DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray
from pandas.core.dtypes.common import is_datetime64_dtype as is_datetime64_dtype, is_datetime64tz_dtype as is_datetime64tz_dtype, is_dtype_equal as is_dtype_equal, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_timedelta64_dtype as is_timedelta64_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype, PandasDtype as PandasDtype
from pandas.core.dtypes.missing import isna as isna
from pandas.errors import IntCastingNaNError as IntCastingNaNError
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import overload

@overload
def astype_nansafe(arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...) -> np.ndarray: ...
@overload
def astype_nansafe(arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...) -> ExtensionArray: ...
def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = ...) -> ArrayLike: ...
def astype_array_safe(values: ArrayLike, dtype, copy: bool = ..., errors: IgnoreRaise = ...) -> ArrayLike: ...
def astype_td64_unit_conversion(values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray: ...
def astype_dt64_to_dt64tz(values: ArrayLike, dtype: DtypeObj, copy: bool, via_utc: bool = ...) -> DatetimeArray: ...
