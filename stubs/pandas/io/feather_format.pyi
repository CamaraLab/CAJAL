from pandas._typing import FilePath as FilePath, ReadBuffer as ReadBuffer, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.api import DataFrame as DataFrame, Int64Index as Int64Index, RangeIndex as RangeIndex
from pandas.io.common import get_handle as get_handle
from pandas.util._decorators import doc as doc
from typing import Hashable, Sequence

def to_feather(df: DataFrame, path: Union[FilePath, WriteBuffer[bytes]], storage_options: StorageOptions = ..., **kwargs) -> None: ...
def read_feather(path: Union[FilePath, ReadBuffer[bytes]], columns: Union[Sequence[Hashable], None] = ..., use_threads: bool = ..., storage_options: StorageOptions = ...): ...
