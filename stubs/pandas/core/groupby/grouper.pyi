import numpy as np
from _typeshed import Incomplete
from pandas._typing import ArrayLike as ArrayLike, NDFrameT as NDFrameT, npt as npt
from pandas.core.arrays import Categorical as Categorical, ExtensionArray as ExtensionArray
from pandas.core.dtypes.cast import sanitize_to_nanoseconds as sanitize_to_nanoseconds
from pandas.core.dtypes.common import is_categorical_dtype as is_categorical_dtype, is_list_like as is_list_like, is_scalar as is_scalar
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby import ops as ops
from pandas.core.groupby.categorical import recode_for_groupby as recode_for_groupby, recode_from_groupby as recode_from_groupby
from pandas.core.indexes.api import CategoricalIndex as CategoricalIndex, Index as Index, MultiIndex as MultiIndex
from pandas.core.series import Series as Series
from pandas.errors import InvalidIndexError as InvalidIndexError
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.util._decorators import cache_readonly as cache_readonly
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Hashable

class Grouper:
    axis: int
    sort: bool
    dropna: bool
    def __new__(cls, *args, **kwargs): ...
    key: Incomplete
    level: Incomplete
    freq: Incomplete
    grouper: Incomplete
    obj: Incomplete
    indexer: Incomplete
    binner: Incomplete
    def __init__(self, key: Incomplete | None = ..., level: Incomplete | None = ..., freq: Incomplete | None = ..., axis: int = ..., sort: bool = ..., dropna: bool = ...) -> None: ...
    @property
    def ax(self) -> Index: ...
    @property
    def groups(self): ...

class Grouping:
    level: Incomplete
    grouping_vector: Incomplete
    obj: Incomplete
    in_axis: Incomplete
    def __init__(self, index: Index, grouper: Incomplete | None = ..., obj: Union[NDFrame, None] = ..., level: Incomplete | None = ..., sort: bool = ..., observed: bool = ..., in_axis: bool = ..., dropna: bool = ...) -> None: ...
    def __iter__(self): ...
    def name(self) -> Hashable: ...
    @property
    def ngroups(self) -> int: ...
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]: ...
    @property
    def codes(self) -> npt.NDArray[np.signedinteger]: ...
    def group_arraylike(self) -> ArrayLike: ...
    def result_index(self) -> Index: ...
    def group_index(self) -> Index: ...
    def groups(self) -> dict[Hashable, np.ndarray]: ...

def get_grouper(obj: NDFrameT, key: Incomplete | None = ..., axis: int = ..., level: Incomplete | None = ..., sort: bool = ..., observed: bool = ..., mutated: bool = ..., validate: bool = ..., dropna: bool = ...) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]: ...
