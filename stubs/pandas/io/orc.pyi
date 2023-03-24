from pandas import DataFrame as DataFrame
from pandas._typing import FilePath as FilePath, ReadBuffer as ReadBuffer, WriteBuffer as WriteBuffer
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import is_categorical_dtype as is_categorical_dtype, is_interval_dtype as is_interval_dtype, is_period_dtype as is_period_dtype, is_unsigned_integer_dtype as is_unsigned_integer_dtype
from pandas.io.common import get_handle as get_handle
from typing import Any, Literal

def read_orc(path: Union[FilePath, ReadBuffer[bytes]], columns: Union[list[str], None] = ..., **kwargs) -> DataFrame: ...
def to_orc(df: DataFrame, path: Union[FilePath, WriteBuffer[bytes], None] = ..., *, engine: Literal['pyarrow'] = ..., index: Union[bool, None] = ..., engine_kwargs: Union[dict[str, Any], None] = ...) -> Union[bytes, None]: ...
