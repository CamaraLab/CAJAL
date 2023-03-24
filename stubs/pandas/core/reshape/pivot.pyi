from _typeshed import Incomplete
from pandas import DataFrame as DataFrame
from pandas._typing import AggFuncType as AggFuncType, AggFuncTypeBase as AggFuncTypeBase, AggFuncTypeDict as AggFuncTypeDict, IndexLabel as IndexLabel
from pandas.core.dtypes.cast import maybe_downcast_to_dtype as maybe_downcast_to_dtype
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_nested_list_like as is_nested_list_like, is_scalar as is_scalar
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.groupby import Grouper as Grouper
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex, get_objs_combined_axis as get_objs_combined_axis
from pandas.core.reshape.concat import concat as concat
from pandas.core.reshape.util import cartesian_product as cartesian_product
from pandas.core.series import Series as Series
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments
from pandas.util._exceptions import rewrite_warning as rewrite_warning

def pivot_table(data: DataFrame, values: Incomplete | None = ..., index: Incomplete | None = ..., columns: Incomplete | None = ..., aggfunc: AggFuncType = ..., fill_value: Incomplete | None = ..., margins: bool = ..., dropna: bool = ..., margins_name: str = ..., observed: bool = ..., sort: bool = ...) -> DataFrame: ...
def pivot(data: DataFrame, index: Union[IndexLabel, None] = ..., columns: Union[IndexLabel, None] = ..., values: Union[IndexLabel, None] = ...) -> DataFrame: ...
def crosstab(index, columns, values: Incomplete | None = ..., rownames: Incomplete | None = ..., colnames: Incomplete | None = ..., aggfunc: Incomplete | None = ..., margins: bool = ..., margins_name: str = ..., dropna: bool = ..., normalize: bool = ...) -> DataFrame: ...
