from _typeshed import Incomplete
from numpy import ndarray
from pandas._libs.lib import is_bool as is_bool, is_integer as is_integer
from pandas._typing import Axis as Axis
from pandas.errors import UnsupportedFunctionCall as UnsupportedFunctionCall
from pandas.util._validators import validate_args as validate_args, validate_args_and_kwargs as validate_args_and_kwargs, validate_kwargs as validate_kwargs
from typing import Any, TypeVar, overload

AxisNoneT = TypeVar('AxisNoneT', Axis, None)

class CompatValidator:
    fname: Incomplete
    method: Incomplete
    defaults: Incomplete
    max_fname_arg_count: Incomplete
    def __init__(self, defaults, fname: Incomplete | None = ..., method: Union[str, None] = ..., max_fname_arg_count: Incomplete | None = ...) -> None: ...
    def __call__(self, args, kwargs, fname: Incomplete | None = ..., max_fname_arg_count: Incomplete | None = ..., method: Union[str, None] = ...) -> None: ...

ARGMINMAX_DEFAULTS: Incomplete
validate_argmin: Incomplete
validate_argmax: Incomplete

def process_skipna(skipna: Union[bool, ndarray, None], args) -> tuple[bool, Any]: ...
def validate_argmin_with_skipna(skipna: Union[bool, ndarray, None], args, kwargs) -> bool: ...
def validate_argmax_with_skipna(skipna: Union[bool, ndarray, None], args, kwargs) -> bool: ...

ARGSORT_DEFAULTS: dict[str, Union[int, str, None]]
validate_argsort: Incomplete
ARGSORT_DEFAULTS_KIND: dict[str, Union[int, None]]
validate_argsort_kind: Incomplete

def validate_argsort_with_ascending(ascending: Union[bool, int, None], args, kwargs) -> bool: ...

CLIP_DEFAULTS: dict[str, Any]
validate_clip: Incomplete


@overload
def validate_clip_with_axis(axis: ndarray, args, kwargs) -> None: ...
@overload
def validate_clip_with_axis(axis: AxisNoneT, args, kwargs) -> AxisNoneT: ...

CUM_FUNC_DEFAULTS: dict[str, Any]
validate_cum_func: Incomplete
validate_cumsum: Incomplete

def validate_cum_func_with_skipna(skipna, args, kwargs, name) -> bool: ...

ALLANY_DEFAULTS: dict[str, Union[bool, None]]
validate_all: Incomplete
validate_any: Incomplete
LOGICAL_FUNC_DEFAULTS: Incomplete
validate_logical_func: Incomplete
MINMAX_DEFAULTS: Incomplete
validate_min: Incomplete
validate_max: Incomplete
RESHAPE_DEFAULTS: dict[str, str]
validate_reshape: Incomplete
REPEAT_DEFAULTS: dict[str, Any]
validate_repeat: Incomplete
ROUND_DEFAULTS: dict[str, Any]
validate_round: Incomplete
SORT_DEFAULTS: dict[str, Union[int, str, None]]
validate_sort: Incomplete
STAT_FUNC_DEFAULTS: dict[str, Union[Any, None]]
SUM_DEFAULTS: Incomplete
PROD_DEFAULTS: Incomplete
MEDIAN_DEFAULTS: Incomplete
validate_stat_func: Incomplete
validate_sum: Incomplete
validate_prod: Incomplete
validate_mean: Incomplete
validate_median: Incomplete
STAT_DDOF_FUNC_DEFAULTS: dict[str, Union[bool, None]]
validate_stat_ddof_func: Incomplete
TAKE_DEFAULTS: dict[str, Union[str, None]]
validate_take: Incomplete

def validate_take_with_convert(convert: Union[ndarray, bool, None], args, kwargs) -> bool: ...

TRANSPOSE_DEFAULTS: Incomplete
validate_transpose: Incomplete

def validate_window_func(name, args, kwargs) -> None: ...
def validate_rolling_func(name, args, kwargs) -> None: ...
def validate_expanding_func(name, args, kwargs) -> None: ...
def validate_groupby_func(name, args, kwargs, allowed: Incomplete | None = ...) -> None: ...

RESAMPLER_NUMPY_OPS: Incomplete

def validate_resampler_func(method: str, args, kwargs) -> None: ...
def validate_minmax_axis(axis: Union[int, None], ndim: int = ...) -> None: ...
