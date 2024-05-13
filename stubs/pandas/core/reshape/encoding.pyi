from _typeshed import Incomplete
from pandas._libs.sparse import IntIndex as IntIndex
from pandas._typing import Dtype as Dtype
from pandas.core.arrays import SparseArray as SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable as factorize_from_iterable
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_object_dtype as is_object_dtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.api import Index as Index
from pandas.core.series import Series as Series
from typing import Hashable

def get_dummies(data, prefix: Incomplete | None = ..., prefix_sep: str = ..., dummy_na: bool = ..., columns: Incomplete | None = ..., sparse: bool = ..., drop_first: bool = ..., dtype: Union[Dtype, None] = ...) -> DataFrame: ...
def from_dummies(data: DataFrame, sep: Union[None, str] = ..., default_category: Union[None, Hashable, dict[str, Hashable]] = ...) -> DataFrame: ...
