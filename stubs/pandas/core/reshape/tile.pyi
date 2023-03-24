from _typeshed import Incomplete
from pandas import Categorical as Categorical, Index as Index, IntervalIndex as IntervalIndex, to_datetime as to_datetime, to_timedelta as to_timedelta
from pandas._libs import Timedelta as Timedelta, Timestamp as Timestamp
from pandas._libs.lib import infer_dtype as infer_dtype
from pandas._typing import IntervalLeftRight as IntervalLeftRight
from pandas.core.dtypes.common import DT64NS_DTYPE as DT64NS_DTYPE, ensure_platform_int as ensure_platform_int, is_bool_dtype as is_bool_dtype, is_categorical_dtype as is_categorical_dtype, is_datetime64_dtype as is_datetime64_dtype, is_datetime64tz_dtype as is_datetime64tz_dtype, is_datetime_or_timedelta_dtype as is_datetime_or_timedelta_dtype, is_extension_array_dtype as is_extension_array_dtype, is_integer as is_integer, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_scalar as is_scalar, is_timedelta64_dtype as is_timedelta64_dtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna

def cut(x, bins, right: bool = ..., labels: Incomplete | None = ..., retbins: bool = ..., precision: int = ..., include_lowest: bool = ..., duplicates: str = ..., ordered: bool = ...): ...
def qcut(x, q, labels: Incomplete | None = ..., retbins: bool = ..., precision: int = ..., duplicates: str = ...): ...
