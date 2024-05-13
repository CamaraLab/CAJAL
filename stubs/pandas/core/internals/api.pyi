from _typeshed import Incomplete
from pandas._libs.internals import BlockPlacement as BlockPlacement
from pandas._typing import Dtype as Dtype
from pandas.core.arrays import DatetimeArray as DatetimeArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import is_datetime64tz_dtype as is_datetime64tz_dtype, is_period_dtype as is_period_dtype, pandas_dtype as pandas_dtype
from pandas.core.internals.blocks import Block as Block, DatetimeTZBlock as DatetimeTZBlock, ExtensionBlock as ExtensionBlock, check_ndim as check_ndim, ensure_block_shape as ensure_block_shape, extract_pandas_array as extract_pandas_array, get_block_type as get_block_type, maybe_coerce_values as maybe_coerce_values

def make_block(values, placement, klass: Incomplete | None = ..., ndim: Incomplete | None = ..., dtype: Union[Dtype, None] = ...) -> Block: ...
def maybe_infer_ndim(values, placement: BlockPlacement, ndim: Union[int, None]) -> int: ...
