from pandas._libs import NaT as NaT
from pandas.core.indexes.base import Index as Index, _new_Index as _new_Index, ensure_index as ensure_index, ensure_index_from_sequences as ensure_index_from_sequences, get_unanimous_names as get_unanimous_names
from pandas.core.indexes.category import CategoricalIndex as CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex as IntervalIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.numeric import Float64Index as Float64Index, Int64Index as Int64Index, NumericIndex as NumericIndex, UInt64Index as UInt64Index
from pandas.core.indexes.period import PeriodIndex as PeriodIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.errors import InvalidIndexError as InvalidIndexError

def get_objs_combined_axis(objs, intersect: bool = ..., axis: int = ..., sort: bool = ..., copy: bool = ...) -> Index: ...
def safe_sort_index(index: Index) -> Index: ...
def union_indexes(indexes, sort: Union[bool, None] = ...) -> Index: ...
def all_indexes_same(indexes) -> bool: ...
def default_index(n: int) -> RangeIndex: ...
