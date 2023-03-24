from _typeshed import Incomplete
from datetime import datetime
from pandas import Series
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.parsing import DateParseError as DateParseError
from pandas._typing import AnyArrayLike, ArrayLike, DateTimeErrorChoices
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from typing import List, Tuple, TypedDict, Union, overload

ArrayConvertible = Union[List, Tuple, AnyArrayLike]
Scalar = Union[float, str]
DatetimeScalar = Union[Scalar, datetime]
DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]
DatetimeDictArg = Union[List[Scalar], Tuple[Scalar, ...], AnyArrayLike]

class YearMonthDayDict(TypedDict):
    year: DatetimeDictArg
    month: DatetimeDictArg
    day: DatetimeDictArg

class FulldatetimeDict(YearMonthDayDict):
    hour: DatetimeDictArg
    hours: DatetimeDictArg
    minute: DatetimeDictArg
    minutes: DatetimeDictArg
    second: DatetimeDictArg
    seconds: DatetimeDictArg
    ms: DatetimeDictArg
    us: DatetimeDictArg
    ns: DatetimeDictArg

def should_cache(arg: ArrayConvertible, unique_share: float = ..., check_count: Union[int, None] = ...) -> bool: ...
@overload
def to_datetime(arg: DatetimeScalar, errors: DateTimeErrorChoices = ..., dayfirst: bool = ..., yearfirst: bool = ..., utc: Union[bool, None] = ..., format: Union[str, None] = ..., exact: bool = ..., unit: Union[str, None] = ..., infer_datetime_format: bool = ..., origin=..., cache: bool = ...) -> Timestamp: ...
@overload
def to_datetime(arg: Union[Series, DictConvertible], errors: DateTimeErrorChoices = ..., dayfirst: bool = ..., yearfirst: bool = ..., utc: Union[bool, None] = ..., format: Union[str, None] = ..., exact: bool = ..., unit: Union[str, None] = ..., infer_datetime_format: bool = ..., origin=..., cache: bool = ...) -> Series: ...
@overload
def to_datetime(arg: Union[list, tuple, Index, ArrayLike], errors: DateTimeErrorChoices = ..., dayfirst: bool = ..., yearfirst: bool = ..., utc: Union[bool, None] = ..., format: Union[str, None] = ..., exact: bool = ..., unit: Union[str, None] = ..., infer_datetime_format: bool = ..., origin=..., cache: bool = ...) -> DatetimeIndex: ...
def to_time(arg, format: Incomplete | None = ..., infer_time_format: bool = ..., errors: str = ...): ...
