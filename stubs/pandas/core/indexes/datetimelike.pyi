import numpy as np
from _typeshed import Incomplete
from pandas import CategoricalIndex as CategoricalIndex
from pandas._libs import NaT as NaT, Timedelta as Timedelta, lib as lib
from pandas._libs.tslibs import BaseOffset as BaseOffset, Resolution as Resolution, Tick as Tick, parsing as parsing, to_offset as to_offset
from pandas.core.arrays import DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray, PeriodArray as PeriodArray, TimedeltaArray as TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin as DatetimeLikeArrayMixin
from pandas.core.dtypes.common import is_categorical_dtype as is_categorical_dtype, is_dtype_equal as is_dtype_equal, is_integer as is_integer, is_list_like as is_list_like
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex as NDArrayBackedExtensionIndex, inherit_names as inherit_names
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.tools.timedeltas import to_timedelta as to_timedelta
from pandas.util._decorators import Appender as Appender, cache_readonly as cache_readonly, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Callable

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex):
    freq: Union[BaseOffset, None]
    freqstr: Union[str, None]
    def hasnans(self) -> bool: ...
    def equals(self, other: Any) -> bool: ...
    def __contains__(self, key: Any) -> bool: ...
    def format(self, name: bool = ..., formatter: Union[Callable, None] = ..., na_rep: str = ..., date_format: Union[str, None] = ...) -> list[str]: ...
    def shift(self, periods: int = ..., freq: Incomplete | None = ...) -> _T: ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin):
    def is_type_compatible(self, kind: str) -> bool: ...
    @property
    def values(self) -> np.ndarray: ...
    def delete(self, loc) -> DatetimeTimedeltaMixin: ...
    def insert(self, loc: int, item): ...
    def take(self, indices, axis: int = ..., allow_fill: bool = ..., fill_value: Incomplete | None = ..., **kwargs): ...
