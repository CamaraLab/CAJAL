from _typeshed import Incomplete
from pandas import Index as Index
from pandas._libs import NaT as NaT
from pandas._libs.missing import NA as NA
from pandas._typing import ArrayLike as ArrayLike, DtypeObj as DtypeObj, Manager as Manager, Shape as Shape
from pandas.core.arrays import DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray
from pandas.core.arrays.sparse import SparseDtype as SparseDtype
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na as ensure_dtype_can_hold_na, find_common_type as find_common_type
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_dtype_equal as is_dtype_equal, is_scalar as is_scalar, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.concat import cast_to_common_type as cast_to_common_type, concat_compat as concat_compat
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, isna_all as isna_all
from pandas.core.internals.array_manager import ArrayManager as ArrayManager, NullArrayProxy as NullArrayProxy
from pandas.core.internals.blocks import Block as Block, ensure_block_shape as ensure_block_shape, new_block_2d as new_block_2d
from pandas.core.internals.managers import BlockManager as BlockManager
from pandas.util._decorators import cache_readonly as cache_readonly

def concat_arrays(to_concat: list) -> ArrayLike: ...
def concatenate_managers(mgrs_indexers, axes: list[Index], concat_axis: int, copy: bool) -> Manager: ...

class JoinUnit:
    block: Incomplete
    indexers: Incomplete
    shape: Incomplete
    def __init__(self, block: Block, shape: Shape, indexers: Incomplete | None = ...) -> None: ...
    def needs_filling(self) -> bool: ...
    def dtype(self) -> DtypeObj: ...
    def is_na(self) -> bool: ...
    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike: ...
