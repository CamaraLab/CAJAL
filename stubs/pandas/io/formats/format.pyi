import numpy as np
from _typeshed import Incomplete
from io import StringIO
from pandas import DataFrame as DataFrame, Series as Series
from pandas._config.config import get_option as get_option, set_option as set_option
from pandas._libs import lib as lib
from pandas._libs.missing import NA as NA
from pandas._libs.tslibs import NaT as NaT, Timedelta as Timedelta, Timestamp as Timestamp, get_unit_from_dtype as get_unit_from_dtype, iNaT as iNaT, periods_per_day as periods_per_day
from pandas._libs.tslibs.nattype import NaTType as NaTType
from pandas._typing import ArrayLike as ArrayLike, Axes as Axes, ColspaceArgType as ColspaceArgType, ColspaceType as ColspaceType, CompressionOptions as CompressionOptions, FilePath as FilePath, FloatFormatType as FloatFormatType, FormattersType as FormattersType, IndexLabel as IndexLabel, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.core.arrays import Categorical as Categorical, DatetimeArray as DatetimeArray, TimedeltaArray as TimedeltaArray
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import is_categorical_dtype as is_categorical_dtype, is_complex_dtype as is_complex_dtype, is_datetime64_dtype as is_datetime64_dtype, is_extension_array_dtype as is_extension_array_dtype, is_float as is_float, is_float_dtype as is_float_dtype, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_scalar as is_scalar, is_timedelta64_dtype as is_timedelta64_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.missing import isna as isna, notna as notna
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex, PeriodIndex as PeriodIndex, ensure_index as ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.core.reshape.concat import concat as concat
from pandas.io.common import check_parent_directory as check_parent_directory, stringify_path as stringify_path
from pandas.io.formats.printing import adjoin as adjoin, justify as justify, pprint_thing as pprint_thing
from pandas.util._decorators import deprecate_kwarg as deprecate_kwarg
from typing import Any, Callable, Final, Hashable, IO, Iterable, Iterator, Sequence

common_docstring: Final[str]
return_docstring: Final[str]

class CategoricalFormatter:
    categorical: Incomplete
    buf: Incomplete
    na_rep: Incomplete
    length: Incomplete
    footer: Incomplete
    quoting: Incomplete
    def __init__(self, categorical: Categorical, buf: Union[IO[str], None] = ..., length: bool = ..., na_rep: str = ..., footer: bool = ...) -> None: ...
    def to_string(self) -> str: ...

class SeriesFormatter:
    series: Incomplete
    buf: Incomplete
    name: Incomplete
    na_rep: Incomplete
    header: Incomplete
    length: Incomplete
    index: Incomplete
    max_rows: Incomplete
    min_rows: Incomplete
    float_format: Incomplete
    dtype: Incomplete
    adj: Incomplete
    def __init__(self, series: Series, buf: Union[IO[str], None] = ..., length: Union[bool, str] = ..., header: bool = ..., index: bool = ..., na_rep: str = ..., name: bool = ..., float_format: Union[str, None] = ..., dtype: bool = ..., max_rows: Union[int, None] = ..., min_rows: Union[int, None] = ...) -> None: ...
    def to_string(self) -> str: ...

class TextAdjustment:
    encoding: Incomplete
    def __init__(self) -> None: ...
    def len(self, text: str) -> int: ...
    def justify(self, texts: Any, max_len: int, mode: str = ...) -> list[str]: ...
    def adjoin(self, space: int, *lists, **kwargs) -> str: ...

class EastAsianTextAdjustment(TextAdjustment):
    ambiguous_width: int
    def __init__(self) -> None: ...
    def len(self, text: str) -> int: ...
    def justify(self, texts: Iterable[str], max_len: int, mode: str = ...) -> list[str]: ...

def get_adjustment() -> TextAdjustment: ...
def get_dataframe_repr_params() -> dict[str, Any]: ...
def get_series_repr_params() -> dict[str, Any]: ...

class DataFrameFormatter:
    __doc__: Incomplete
    frame: Incomplete
    columns: Incomplete
    col_space: Incomplete
    header: Incomplete
    index: Incomplete
    na_rep: Incomplete
    formatters: Incomplete
    justify: Incomplete
    float_format: Incomplete
    sparsify: Incomplete
    show_index_names: Incomplete
    decimal: Incomplete
    bold_rows: Incomplete
    escape: Incomplete
    max_rows: Incomplete
    min_rows: Incomplete
    max_cols: Incomplete
    show_dimensions: Incomplete
    max_cols_fitted: Incomplete
    max_rows_fitted: Incomplete
    tr_frame: Incomplete
    adj: Incomplete
    def __init__(self, frame: DataFrame, columns: Union[Sequence[Hashable], None] = ..., col_space: Union[ColspaceArgType, None] = ..., header: Union[bool, Sequence[str]] = ..., index: bool = ..., na_rep: str = ..., formatters: Union[FormattersType, None] = ..., justify: Union[str, None] = ..., float_format: Union[FloatFormatType, None] = ..., sparsify: Union[bool, None] = ..., index_names: bool = ..., max_rows: Union[int, None] = ..., min_rows: Union[int, None] = ..., max_cols: Union[int, None] = ..., show_dimensions: Union[bool, str] = ..., decimal: str = ..., bold_rows: bool = ..., escape: bool = ...) -> None: ...
    def get_strcols(self) -> list[list[str]]: ...
    @property
    def should_show_dimensions(self) -> bool: ...
    @property
    def is_truncated(self) -> bool: ...
    @property
    def is_truncated_horizontally(self) -> bool: ...
    @property
    def is_truncated_vertically(self) -> bool: ...
    @property
    def dimensions_info(self) -> str: ...
    @property
    def has_index_names(self) -> bool: ...
    @property
    def has_column_names(self) -> bool: ...
    @property
    def show_row_idx_names(self) -> bool: ...
    @property
    def show_col_idx_names(self) -> bool: ...
    @property
    def max_rows_displayed(self) -> int: ...
    def truncate(self) -> None: ...
    def format_col(self, i: int) -> list[str]: ...

class DataFrameRenderer:
    fmt: Incomplete
    def __init__(self, fmt: DataFrameFormatter) -> None: ...
    def to_latex(self, buf: Union[FilePath, WriteBuffer[str], None] = ..., column_format: Union[str, None] = ..., longtable: bool = ..., encoding: Union[str, None] = ..., multicolumn: bool = ..., multicolumn_format: Union[str, None] = ..., multirow: bool = ..., caption: Union[str, tuple[str, str], None] = ..., label: Union[str, None] = ..., position: Union[str, None] = ...) -> Union[str, None]: ...
    def to_html(self, buf: Union[FilePath, WriteBuffer[str], None] = ..., encoding: Union[str, None] = ..., classes: Union[str, list, tuple, None] = ..., notebook: bool = ..., border: Union[int, bool, None] = ..., table_id: Union[str, None] = ..., render_links: bool = ...) -> Union[str, None]: ...
    def to_string(self, buf: Union[FilePath, WriteBuffer[str], None] = ..., encoding: Union[str, None] = ..., line_width: Union[int, None] = ...) -> Union[str, None]: ...
    def to_csv(self, path_or_buf: Union[FilePath, WriteBuffer[bytes], WriteBuffer[str], None] = ..., encoding: Union[str, None] = ..., sep: str = ..., columns: Union[Sequence[Hashable], None] = ..., index_label: Union[IndexLabel, None] = ..., mode: str = ..., compression: CompressionOptions = ..., quoting: Union[int, None] = ..., quotechar: str = ..., lineterminator: Union[str, None] = ..., chunksize: Union[int, None] = ..., date_format: Union[str, None] = ..., doublequote: bool = ..., escapechar: Union[str, None] = ..., errors: str = ..., storage_options: StorageOptions = ...) -> Union[str, None]: ...

def save_to_buffer(string: str, buf: Union[FilePath, WriteBuffer[str], None] = ..., encoding: Union[str, None] = ...) -> Union[str, None]: ...
def get_buffer(buf: Union[FilePath, WriteBuffer[str], None], encoding: Union[str, None] = ...) -> Union[Iterator[WriteBuffer[str]], Iterator[StringIO]]: ...
def format_array(values: Any, formatter: Union[Callable, None], float_format: Union[FloatFormatType, None] = ..., na_rep: str = ..., digits: Union[int, None] = ..., space: Union[str, int, None] = ..., justify: str = ..., decimal: str = ..., leading_space: Union[bool, None] = ..., quoting: Union[int, None] = ...) -> list[str]: ...

class GenericArrayFormatter:
    values: Incomplete
    digits: Incomplete
    na_rep: Incomplete
    space: Incomplete
    formatter: Incomplete
    float_format: Incomplete
    justify: Incomplete
    decimal: Incomplete
    quoting: Incomplete
    fixed_width: Incomplete
    leading_space: Incomplete
    def __init__(self, values: Any, digits: int = ..., formatter: Union[Callable, None] = ..., na_rep: str = ..., space: Union[str, int] = ..., float_format: Union[FloatFormatType, None] = ..., justify: str = ..., decimal: str = ..., quoting: Union[int, None] = ..., fixed_width: bool = ..., leading_space: Union[bool, None] = ...) -> None: ...
    def get_result(self) -> list[str]: ...

class FloatArrayFormatter(GenericArrayFormatter):
    fixed_width: bool
    formatter: Incomplete
    float_format: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def get_result_as_array(self) -> np.ndarray: ...

class IntArrayFormatter(GenericArrayFormatter): ...

class Datetime64Formatter(GenericArrayFormatter):
    nat_rep: Incomplete
    date_format: Incomplete
    def __init__(self, values: Union[np.ndarray, Series, DatetimeIndex, DatetimeArray], nat_rep: str = ..., date_format: None = ..., **kwargs) -> None: ...

class ExtensionArrayFormatter(GenericArrayFormatter): ...

def format_percentiles(percentiles: Union[np.ndarray, Sequence[float]]) -> list[str]: ...
def is_dates_only(values: Union[np.ndarray, DatetimeArray, Index, DatetimeIndex]) -> bool: ...
def get_format_datetime64(is_dates_only: bool, nat_rep: str = ..., date_format: Union[str, None] = ...) -> Callable: ...
def get_format_datetime64_from_values(values: Union[np.ndarray, DatetimeArray, DatetimeIndex], date_format: Union[str, None]) -> Union[str, None]: ...

class Datetime64TZFormatter(Datetime64Formatter): ...

class Timedelta64Formatter(GenericArrayFormatter):
    nat_rep: Incomplete
    box: Incomplete
    def __init__(self, values: Union[np.ndarray, TimedeltaIndex], nat_rep: str = ..., box: bool = ..., **kwargs) -> None: ...

def get_format_timedelta64(values: Union[np.ndarray, TimedeltaIndex, TimedeltaArray], nat_rep: str = ..., box: bool = ...) -> Callable: ...

class EngFormatter:
    ENG_PREFIXES: Incomplete
    accuracy: Incomplete
    use_eng_prefix: Incomplete
    def __init__(self, accuracy: Union[int, None] = ..., use_eng_prefix: bool = ...) -> None: ...
    def __call__(self, num: float) -> str: ...

def set_eng_float_format(accuracy: int = ..., use_eng_prefix: bool = ...) -> None: ...
def get_level_lengths(levels: Any, sentinel: Union[bool, object, str] = ...) -> list[dict[int, int]]: ...
def buffer_put_lines(buf: WriteBuffer[str], lines: list[str]) -> None: ...
