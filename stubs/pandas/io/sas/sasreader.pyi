from abc import ABCMeta, abstractmethod
from pandas import DataFrame as DataFrame
from pandas._typing import CompressionOptions as CompressionOptions, FilePath as FilePath, ReadBuffer as ReadBuffer
from pandas.io.common import stringify_path as stringify_path
from pandas.util._decorators import deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments, doc as doc
from typing import Hashable, overload

class ReaderBase(metaclass=ABCMeta):
    @abstractmethod
    def read(self, nrows: Union[int, None] = ...) -> DataFrame: ...
    @abstractmethod
    def close(self) -> None: ...
    def __enter__(self) -> ReaderBase: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...


@overload
def read_sas(filepath_or_buffer: Union[FilePath, ReadBuffer[bytes]], format: Union[str, None] = ..., index: Union[Hashable, None] = ..., encoding: Union[str, None] = ..., chunksize: int = ..., iterator: bool = ..., compression: CompressionOptions = ...) -> ReaderBase: ...
@overload
def read_sas(filepath_or_buffer: Union[FilePath, ReadBuffer[bytes]], format: Union[str, None] = ..., index: Union[Hashable, None] = ..., encoding: Union[str, None] = ..., chunksize: None = ..., iterator: bool = ..., compression: CompressionOptions = ...) -> Union[DataFrame, ReaderBase]: ...
