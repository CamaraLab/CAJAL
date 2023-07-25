from _typeshed import Incomplete
from pandas import DataFrame as DataFrame
from pandas.core.arrays import Categorical as Categorical
from pandas.core.dtypes.common import is_extension_array_dtype as is_extension_array_dtype, is_list_like as is_list_like
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.missing import notna as notna
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex
from pandas.core.reshape.concat import concat as concat
from pandas.core.reshape.util import tile_compat as tile_compat
from pandas.core.tools.numeric import to_numeric as to_numeric
from pandas.util._decorators import Appender as Appender, deprecate_kwarg as deprecate_kwarg
from pandas.util._exceptions import find_stack_level as find_stack_level

def melt(frame: DataFrame, id_vars: Incomplete | None = ..., value_vars: Incomplete | None = ..., var_name: Incomplete | None = ..., value_name: str = ..., col_level: Incomplete | None = ..., ignore_index: bool = ...) -> DataFrame: ...
def lreshape(data: DataFrame, groups, dropna: bool = ..., label: Incomplete | None = ...) -> DataFrame: ...
def wide_to_long(df: DataFrame, stubnames, i, j, sep: str = ..., suffix: str = ...) -> DataFrame: ...
