import numpy as np
from _typeshed import Incomplete
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, npt as npt
from pandas.core.arrays.categorical import Categorical as Categorical, contains as contains
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import is_categorical_dtype as is_categorical_dtype, is_scalar as is_scalar, pandas_dtype as pandas_dtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, notna as notna
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex as NDArrayBackedExtensionIndex, inherit_names as inherit_names
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.util._decorators import cache_readonly as cache_readonly, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Hashable

class CategoricalIndex(NDArrayBackedExtensionIndex):
    codes: np.ndarray
    categories: Index
    ordered: Union[bool, None]
    def __new__(cls, data: Incomplete | None = ..., categories: Incomplete | None = ..., ordered: Incomplete | None = ..., dtype: Union[Dtype, None] = ..., copy: bool = ..., name: Hashable = ...) -> CategoricalIndex: ...
    def astype(self, dtype: Dtype, copy: bool = ...) -> Index: ...
    def equals(self, other: object) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def __contains__(self, key: Any) -> bool: ...
    def reindex(self, target, method: Incomplete | None = ..., level: Incomplete | None = ..., limit: Incomplete | None = ..., tolerance: Incomplete | None = ...) -> tuple[Index, Union[npt.NDArray[np.intp], None]]: ...
    def take_nd(self, *args, **kwargs) -> CategoricalIndex: ...
    def map(self, mapper): ...
