import numpy as np
from _typeshed import Incomplete
from collections import abc
from pandas import DataFrame as DataFrame, isna as isna
from pandas._typing import CompressionOptions as CompressionOptions, FilePath as FilePath, ReadBuffer as ReadBuffer
from pandas.errors import EmptyDataError as EmptyDataError, OutOfBoundsDatetime as OutOfBoundsDatetime
from pandas.io.common import get_handle as get_handle
from pandas.io.sas._sas import Parser as Parser
from pandas.io.sas.sasreader import ReaderBase as ReaderBase

class _SubheaderPointer:
    offset: int
    length: int
    compression: int
    ptype: int
    def __init__(self, offset: int, length: int, compression: int, ptype: int) -> None: ...

class _Column:
    col_id: int
    name: Union[str, bytes]
    label: Union[str, bytes]
    format: Union[str, bytes]
    ctype: bytes
    length: int
    def __init__(self, col_id: int, name: Union[str, bytes], label: Union[str, bytes], format: Union[str, bytes], ctype: bytes, length: int) -> None: ...

class SAS7BDATReader(ReaderBase, abc.Iterator):
    index: Incomplete
    convert_dates: Incomplete
    blank_missing: Incomplete
    chunksize: Incomplete
    encoding: Incomplete
    convert_text: Incomplete
    convert_header_text: Incomplete
    default_encoding: str
    compression: bytes
    column_names_raw: Incomplete
    column_names: Incomplete
    column_formats: Incomplete
    columns: Incomplete
    handles: Incomplete
    def __init__(self, path_or_buf: Union[FilePath, ReadBuffer[bytes]], index: Incomplete | None = ..., convert_dates: bool = ..., blank_missing: bool = ..., chunksize: Union[int, None] = ..., encoding: Union[str, None] = ..., convert_text: bool = ..., convert_header_text: bool = ..., compression: CompressionOptions = ...) -> None: ...
    def column_data_lengths(self) -> np.ndarray: ...
    def column_data_offsets(self) -> np.ndarray: ...
    def column_types(self) -> np.ndarray: ...
    def close(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def read(self, nrows: Union[int, None] = ...) -> DataFrame: ...
