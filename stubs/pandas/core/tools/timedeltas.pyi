from datetime import timedelta
from pandas import Index as Index, Series as Series, TimedeltaIndex as TimedeltaIndex
from pandas._libs import lib as lib
from pandas._libs.tslibs import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta, UnitChoices as UnitChoices, parse_timedelta_unit as parse_timedelta_unit
from pandas._typing import ArrayLike as ArrayLike, DateTimeErrorChoices as DateTimeErrorChoices
from pandas.core.arrays.timedeltas import sequence_to_td64ns as sequence_to_td64ns
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries
from typing import overload

@overload
def to_timedelta(arg: Union[str, float, timedelta], unit: Union[UnitChoices, None] = ..., errors: DateTimeErrorChoices = ...) -> Timedelta: ...
@overload
def to_timedelta(arg: Series, unit: Union[UnitChoices, None] = ..., errors: DateTimeErrorChoices = ...) -> Series: ...
@overload
def to_timedelta(arg: Union[list, tuple, range, ArrayLike, Index], unit: Union[UnitChoices, None] = ..., errors: DateTimeErrorChoices = ...) -> TimedeltaIndex: ...
