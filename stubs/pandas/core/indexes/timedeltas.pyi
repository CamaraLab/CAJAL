from _typeshed import Incomplete
from pandas._libs import lib as lib
from pandas._libs.tslibs import Timedelta as Timedelta, to_offset as to_offset
from pandas._typing import DtypeObj as DtypeObj
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.dtypes.common import TD64NS_DTYPE as TD64NS_DTYPE, is_scalar as is_scalar, is_timedelta64_dtype as is_timedelta64_dtype
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin as DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names as inherit_names

class TimedeltaIndex(DatetimeTimedeltaMixin):
    def __new__(cls, data: Incomplete | None = ..., unit: Incomplete | None = ..., freq=..., closed: Incomplete | None = ..., dtype=..., copy: bool = ..., name: Incomplete | None = ...): ...
    def get_loc(self, key, method: Incomplete | None = ..., tolerance: Incomplete | None = ...): ...
    @property
    def inferred_type(self) -> str: ...

def timedelta_range(start: Incomplete | None = ..., end: Incomplete | None = ..., periods: Union[int, None] = ..., freq: Incomplete | None = ..., name: Incomplete | None = ..., closed: Incomplete | None = ...) -> TimedeltaIndex: ...
