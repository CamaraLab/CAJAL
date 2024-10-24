import pandas as pd
from pandas._libs.lib import infer_dtype as infer_dtype
from pandas._libs.tslibs import iNaT as iNaT
from pandas.api.types import is_categorical_dtype as is_categorical_dtype, is_string_dtype as is_string_dtype
from pandas.core.interchange.buffer import PandasBuffer as PandasBuffer
from pandas.core.interchange.dataframe_protocol import Column as Column, ColumnBuffers as ColumnBuffers, ColumnNullType as ColumnNullType, DtypeKind as DtypeKind
from pandas.core.interchange.utils import ArrowCTypes as ArrowCTypes, Endianness as Endianness, NoBufferPresent as NoBufferPresent, dtype_to_arrow_c_fmt as dtype_to_arrow_c_fmt
from pandas.util._decorators import cache_readonly as cache_readonly

class PandasColumn(Column):
    def __init__(self, column: pd.Series, allow_copy: bool = ...) -> None: ...
    def size(self) -> int: ...
    @property
    def offset(self) -> int: ...
    def dtype(self) -> tuple[DtypeKind, int, str, str]: ...
    @property
    def describe_categorical(self): ...
    @property
    def describe_null(self): ...
    def null_count(self) -> int: ...
    @property
    def metadata(self) -> dict[str, pd.Index]: ...
    def num_chunks(self) -> int: ...
    def get_chunks(self, n_chunks: Union[int, None] = ...): ...
    def get_buffers(self) -> ColumnBuffers: ...
