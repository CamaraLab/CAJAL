import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from pandas import DataFrame as DataFrame, MultiIndex as MultiIndex, Series as Series, isna as isna, notna as notna, to_datetime as to_datetime
from pandas._libs.tslibs import iNaT as iNaT
from pandas._typing import CompressionOptions as CompressionOptions, DtypeArg as DtypeArg, FilePath as FilePath, IndexLabel as IndexLabel, JSONSerializable as JSONSerializable, ReadBuffer as ReadBuffer, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.core.construction import create_series_with_explicit_dtype as create_series_with_explicit_dtype
from pandas.core.dtypes.common import ensure_str as ensure_str, is_period_dtype as is_period_dtype
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.reshape.concat import concat as concat
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.io.common import IOHandles as IOHandles, file_exists as file_exists, get_handle as get_handle, is_fsspec_url as is_fsspec_url, is_url as is_url, stringify_path as stringify_path
from pandas.io.json._normalize import convert_to_line_delimits as convert_to_line_delimits
from pandas.io.json._table_schema import build_table_schema as build_table_schema, parse_table_schema as parse_table_schema
from pandas.io.parsers.readers import validate_integer as validate_integer
from pandas.util._decorators import deprecate_kwarg as deprecate_kwarg, deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments, doc as doc
from typing import Any, Callable, Literal, Mapping, TypeVar, overload

FrameSeriesStrT = TypeVar('FrameSeriesStrT', bound=Literal['frame', 'series'])
loads: Incomplete
dumps: Incomplete


@overload
def to_json(path_or_buf: Union[FilePath, WriteBuffer[str], WriteBuffer[bytes]], obj: NDFrame, orient: Union[str, None] = ..., date_format: str = ..., double_precision: int = ..., force_ascii: bool = ..., date_unit: str = ..., default_handler: Union[Callable[[Any], JSONSerializable], None] = ..., lines: bool = ..., compression: CompressionOptions = ..., index: bool = ..., indent: int = ..., storage_options: StorageOptions = ...) -> None: ...
@overload
def to_json(path_or_buf: None, obj: NDFrame, orient: Union[str, None] = ..., date_format: str = ..., double_precision: int = ..., force_ascii: bool = ..., date_unit: str = ..., default_handler: Union[Callable[[Any], JSONSerializable], None] = ..., lines: bool = ..., compression: CompressionOptions = ..., index: bool = ..., indent: int = ..., storage_options: StorageOptions = ...) -> str: ...

class Writer(ABC, metaclass=abc.ABCMeta):
    obj: Incomplete
    orient: Incomplete
    date_format: Incomplete
    double_precision: Incomplete
    ensure_ascii: Incomplete
    date_unit: Incomplete
    default_handler: Incomplete
    index: Incomplete
    indent: Incomplete
    is_copy: Incomplete
    def __init__(self, obj, orient: Union[str, None], date_format: str, double_precision: int, ensure_ascii: bool, date_unit: str, index: bool, default_handler: Union[Callable[[Any], JSONSerializable], None] = ..., indent: int = ...) -> None: ...
    def write(self) -> str: ...
    @property
    @abstractmethod
    def obj_to_write(self) -> Union[NDFrame, Mapping[IndexLabel, Any]]: ...

class SeriesWriter(Writer):
    @property
    def obj_to_write(self) -> Union[NDFrame, Mapping[IndexLabel, Any]]: ...

class FrameWriter(Writer):
    @property
    def obj_to_write(self) -> Union[NDFrame, Mapping[IndexLabel, Any]]: ...

class JSONTableWriter(FrameWriter):
    schema: Incomplete
    obj: Incomplete
    date_format: str
    orient: str
    index: Incomplete
    def __init__(self, obj, orient: Union[str, None], date_format: str, double_precision: int, ensure_ascii: bool, date_unit: str, index: bool, default_handler: Union[Callable[[Any], JSONSerializable], None] = ..., indent: int = ...) -> None: ...
    @property
    def obj_to_write(self) -> Union[NDFrame, Mapping[IndexLabel, Any]]: ...


@overload
def read_json(path_or_buf: Union[FilePath, ReadBuffer[str], ReadBuffer[bytes]], *, orient: Union[str, None] = ..., typ: Literal['frame'] = ..., dtype: Union[DtypeArg, None] = ..., convert_axes=..., convert_dates: Union[bool, list[str]] = ..., keep_default_dates: bool = ..., numpy: bool = ..., precise_float: bool = ..., date_unit: Union[str, None] = ..., encoding: Union[str, None] = ..., encoding_errors: Union[str, None] = ..., lines: bool = ..., chunksize: int, compression: CompressionOptions = ..., nrows: Union[int, None] = ..., storage_options: StorageOptions = ...) -> JsonReader[Literal['frame']]: ...
@overload
def read_json(path_or_buf: Union[FilePath, ReadBuffer[str], ReadBuffer[bytes]], *, orient: Union[str, None] = ..., typ: Literal['series'], dtype: Union[DtypeArg, None] = ..., convert_axes=..., convert_dates: Union[bool, list[str]] = ..., keep_default_dates: bool = ..., numpy: bool = ..., precise_float: bool = ..., date_unit: Union[str, None] = ..., encoding: Union[str, None] = ..., encoding_errors: Union[str, None] = ..., lines: bool = ..., chunksize: int, compression: CompressionOptions = ..., nrows: Union[int, None] = ..., storage_options: StorageOptions = ...) -> JsonReader[Literal['series']]: ...
@overload
def read_json(path_or_buf: Union[FilePath, ReadBuffer[str], ReadBuffer[bytes]], *, orient: Union[str, None] = ..., typ: Literal['series'], dtype: Union[DtypeArg, None] = ..., convert_axes=..., convert_dates: Union[bool, list[str]] = ..., keep_default_dates: bool = ..., numpy: bool = ..., precise_float: bool = ..., date_unit: Union[str, None] = ..., encoding: Union[str, None] = ..., encoding_errors: Union[str, None] = ..., lines: bool = ..., chunksize: None = ..., compression: CompressionOptions = ..., nrows: Union[int, None] = ..., storage_options: StorageOptions = ...) -> Series: ...
@overload
def read_json(path_or_buf: Union[FilePath, ReadBuffer[str], ReadBuffer[bytes]], orient: Union[str, None] = ..., typ: Literal['frame'] = ..., dtype: Union[DtypeArg, None] = ..., convert_axes=..., convert_dates: Union[bool, list[str]] = ..., keep_default_dates: bool = ..., numpy: bool = ..., precise_float: bool = ..., date_unit: Union[str, None] = ..., encoding: Union[str, None] = ..., encoding_errors: Union[str, None] = ..., lines: bool = ..., chunksize: None = ..., compression: CompressionOptions = ..., nrows: Union[int, None] = ..., storage_options: StorageOptions = ...) -> DataFrame: ...

class JsonReader(abc.Iterator):
    orient: Incomplete
    typ: Incomplete
    dtype: Incomplete
    convert_axes: Incomplete
    convert_dates: Incomplete
    keep_default_dates: Incomplete
    numpy: Incomplete
    precise_float: Incomplete
    date_unit: Incomplete
    encoding: Incomplete
    compression: Incomplete
    storage_options: Incomplete
    lines: Incomplete
    chunksize: Incomplete
    nrows_seen: int
    nrows: Incomplete
    encoding_errors: Incomplete
    handles: Incomplete
    data: Incomplete
    def __init__(self, filepath_or_buffer, orient, typ: FrameSeriesStrT, dtype, convert_axes, convert_dates, keep_default_dates: bool, numpy: bool, precise_float: bool, date_unit, encoding, lines: bool, chunksize: Union[int, None], compression: CompressionOptions, nrows: Union[int, None], storage_options: StorageOptions = ..., encoding_errors: Union[str, None] = ...) -> None: ...
    @overload
    def read(self) -> DataFrame: ...
    @overload
    def read(self) -> Series: ...
    @overload
    def read(self) -> Union[DataFrame, Series]: ...
    def close(self) -> None: ...
    def __iter__(self) -> JsonReader[FrameSeriesStrT]: ...
    @overload
    def __next__(self) -> DataFrame: ...
    @overload
    def __next__(self) -> Series: ...
    @overload
    def __next__(self) -> Union[DataFrame, Series]: ...
    def __enter__(self) -> JsonReader[FrameSeriesStrT]: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class Parser:
    json: Incomplete
    orient: Incomplete
    dtype: Incomplete
    min_stamp: Incomplete
    numpy: Incomplete
    precise_float: Incomplete
    convert_axes: Incomplete
    convert_dates: Incomplete
    date_unit: Incomplete
    keep_default_dates: Incomplete
    obj: Incomplete
    def __init__(self, json, orient, dtype: Union[DtypeArg, None] = ..., convert_axes: bool = ..., convert_dates: Union[bool, list[str]] = ..., keep_default_dates: bool = ..., numpy: bool = ..., precise_float: bool = ..., date_unit: Incomplete | None = ...) -> None: ...
    def check_keys_split(self, decoded) -> None: ...
    def parse(self): ...

class SeriesParser(Parser): ...
class FrameParser(Parser): ...
