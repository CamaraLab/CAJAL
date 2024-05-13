import numpy as np
from _typeshed import Incomplete
from pandas import Categorical as Categorical, DataFrame as DataFrame, Index as Index, MultiIndex as MultiIndex, Series as Series
from pandas._libs import Timedelta as Timedelta, lib as lib
from pandas._typing import AnyArrayLike as AnyArrayLike, ArrayLike as ArrayLike, DtypeObj as DtypeObj, IndexLabel as IndexLabel, Suffixes as Suffixes, npt as npt
from pandas.core import groupby as groupby
from pandas.core.arrays import DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.cast import find_common_type as find_common_type
from pandas.core.dtypes.common import ensure_float64 as ensure_float64, ensure_int64 as ensure_int64, ensure_object as ensure_object, is_array_like as is_array_like, is_bool as is_bool, is_bool_dtype as is_bool_dtype, is_categorical_dtype as is_categorical_dtype, is_dtype_equal as is_dtype_equal, is_extension_array_dtype as is_extension_array_dtype, is_float_dtype as is_float_dtype, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_number as is_number, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.sorting import is_int64_overflow_possible as is_int64_overflow_possible
from pandas.errors import MergeError as MergeError
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, cache_readonly as cache_readonly
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Hashable, Sequence

def merge(left: Union[DataFrame, Series], right: Union[DataFrame, Series], how: str = ..., on: Union[IndexLabel, None] = ..., left_on: Union[IndexLabel, None] = ..., right_on: Union[IndexLabel, None] = ..., left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., copy: bool = ..., indicator: bool = ..., validate: Union[str, None] = ...) -> DataFrame: ...
def merge_ordered(left: DataFrame, right: DataFrame, on: Union[IndexLabel, None] = ..., left_on: Union[IndexLabel, None] = ..., right_on: Union[IndexLabel, None] = ..., left_by: Incomplete | None = ..., right_by: Incomplete | None = ..., fill_method: Union[str, None] = ..., suffixes: Suffixes = ..., how: str = ...) -> DataFrame: ...
def merge_asof(left: Union[DataFrame, Series], right: Union[DataFrame, Series], on: Union[IndexLabel, None] = ..., left_on: Union[IndexLabel, None] = ..., right_on: Union[IndexLabel, None] = ..., left_index: bool = ..., right_index: bool = ..., by: Incomplete | None = ..., left_by: Incomplete | None = ..., right_by: Incomplete | None = ..., suffixes: Suffixes = ..., tolerance: Incomplete | None = ..., allow_exact_matches: bool = ..., direction: str = ...) -> DataFrame: ...

class _MergeOperation:
    how: str
    on: Union[IndexLabel, None]
    left_on: Sequence[Union[Hashable, AnyArrayLike]]
    right_on: Sequence[Union[Hashable, AnyArrayLike]]
    left_index: bool
    right_index: bool
    axis: int
    bm_axis: int
    sort: bool
    suffixes: Suffixes
    copy: bool
    indicator: bool
    validate: Union[str, None]
    left: Incomplete
    right: Incomplete
    def __init__(self, left: Union[DataFrame, Series], right: Union[DataFrame, Series], how: str = ..., on: Union[IndexLabel, None] = ..., left_on: Union[IndexLabel, None] = ..., right_on: Union[IndexLabel, None] = ..., axis: int = ..., left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., indicator: bool = ..., validate: Union[str, None] = ...) -> None: ...
    def get_result(self, copy: bool = ...) -> DataFrame: ...

def get_join_indexers(left_keys, right_keys, sort: bool = ..., how: str = ..., **kwargs) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
def restore_dropped_levels_multijoin(left: MultiIndex, right: MultiIndex, dropped_level_names, join_index: Index, lindexer: npt.NDArray[np.intp], rindexer: npt.NDArray[np.intp]) -> tuple[list[Index], npt.NDArray[np.intp], list[Hashable]]: ...

class _OrderedMerge(_MergeOperation):
    fill_method: Incomplete
    def __init__(self, left: Union[DataFrame, Series], right: Union[DataFrame, Series], on: Union[IndexLabel, None] = ..., left_on: Union[IndexLabel, None] = ..., right_on: Union[IndexLabel, None] = ..., left_index: bool = ..., right_index: bool = ..., axis: int = ..., suffixes: Suffixes = ..., fill_method: Union[str, None] = ..., how: str = ...) -> None: ...
    def get_result(self, copy: bool = ...) -> DataFrame: ...

class _AsOfMerge(_OrderedMerge):
    by: Incomplete
    left_by: Incomplete
    right_by: Incomplete
    tolerance: Incomplete
    allow_exact_matches: Incomplete
    direction: Incomplete
    def __init__(self, left: Union[DataFrame, Series], right: Union[DataFrame, Series], on: Union[IndexLabel, None] = ..., left_on: Union[IndexLabel, None] = ..., right_on: Union[IndexLabel, None] = ..., left_index: bool = ..., right_index: bool = ..., by: Incomplete | None = ..., left_by: Incomplete | None = ..., right_by: Incomplete | None = ..., axis: int = ..., suffixes: Suffixes = ..., copy: bool = ..., fill_method: Union[str, None] = ..., how: str = ..., tolerance: Incomplete | None = ..., allow_exact_matches: bool = ..., direction: str = ...) -> None: ...
