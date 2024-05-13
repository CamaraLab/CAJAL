import numpy as np
from _typeshed import Incomplete
from datetime import timedelta
from pandas._libs import lib as lib
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.tslibs import BaseOffset as BaseOffset, NaT as NaT, NaTType as NaTType, Timedelta as Timedelta, astype_overflowsafe as astype_overflowsafe, get_unit_from_dtype as get_unit_from_dtype, iNaT as iNaT, parsing as parsing, period as libperiod, to_offset as to_offset
from pandas._libs.tslibs.dtypes import FreqGroup as FreqGroup
from pandas._libs.tslibs.fields import isleapyear_arr as isleapyear_arr
from pandas._libs.tslibs.offsets import Tick as Tick, delta_to_tick as delta_to_tick
from pandas._libs.tslibs.period import DIFFERENT_FREQ as DIFFERENT_FREQ, IncompatibleFrequency as IncompatibleFrequency, Period as Period, get_period_field_arr as get_period_field_arr, period_asfreq_arr as period_asfreq_arr
from pandas._typing import AnyArrayLike as AnyArrayLike, Dtype as Dtype, NpDtype as NpDtype, NumpySorter as NumpySorter, NumpyValueArrayLike as NumpyValueArrayLike, npt as npt
from pandas.core.arrays import DatetimeArray as DatetimeArray, TimedeltaArray as TimedeltaArray, datetimelike as dtl
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.dtypes.common import ensure_object as ensure_object, is_datetime64_any_dtype as is_datetime64_any_dtype, is_datetime64_dtype as is_datetime64_dtype, is_dtype_equal as is_dtype_equal, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_period_dtype as is_period_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCPeriodIndex as ABCPeriodIndex, ABCSeries as ABCSeries, ABCTimedeltaArray as ABCTimedeltaArray
from pandas.core.dtypes.missing import isna as isna
from pandas.util._decorators import cache_readonly as cache_readonly, doc as doc
from typing import Literal, Sequence, TypeVar, overload

BaseOffsetT = TypeVar('BaseOffsetT', bound=BaseOffset)

class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    __array_priority__: int
    def __init__(self, values, dtype: Union[Dtype, None] = ..., freq: Incomplete | None = ..., copy: bool = ...) -> None: ...
    def dtype(self) -> PeriodDtype: ...
    @property
    def freq(self) -> BaseOffset: ...
    def __array__(self, dtype: Union[NpDtype, None] = ...) -> np.ndarray: ...
    def __arrow_array__(self, type: Incomplete | None = ...): ...
    year: Incomplete
    month: Incomplete
    day: Incomplete
    hour: Incomplete
    minute: Incomplete
    second: Incomplete
    weekofyear: Incomplete
    week: Incomplete
    day_of_week: Incomplete
    dayofweek: Incomplete
    weekday: Incomplete
    dayofyear: Incomplete
    day_of_year: Incomplete
    quarter: Incomplete
    qyear: Incomplete
    days_in_month: Incomplete
    daysinmonth: Incomplete
    @property
    def is_leap_year(self) -> np.ndarray: ...
    def to_timestamp(self, freq: Incomplete | None = ..., how: str = ...) -> DatetimeArray: ...
    def asfreq(self, freq: Incomplete | None = ..., how: str = ...) -> PeriodArray: ...
    def astype(self, dtype, copy: bool = ...): ...
    def searchsorted(self, value: Union[NumpyValueArrayLike, ExtensionArray], side: Literal['left', 'right'] = ..., sorter: NumpySorter = ...) -> Union[npt.NDArray[np.intp], np.intp]: ...
    def fillna(self, value: Incomplete | None = ..., method: Incomplete | None = ..., limit: Incomplete | None = ...) -> PeriodArray: ...

def raise_on_incompatible(left, right): ...
def period_array(data: Union[Sequence[Union[Period, str, None]], AnyArrayLike], freq: Union[str, Tick, None] = ..., copy: bool = ...) -> PeriodArray: ...
@overload
def validate_dtype_freq(dtype, freq: BaseOffsetT) -> BaseOffsetT: ...
@overload
def validate_dtype_freq(dtype, freq: Union[timedelta, str, None]) -> BaseOffset: ...
def dt64arr_to_periodarr(data, freq, tz: Incomplete | None = ...) -> tuple[npt.NDArray[np.int64], BaseOffset]: ...
