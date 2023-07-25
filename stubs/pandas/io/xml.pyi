import io
from _typeshed import Incomplete
from pandas import DataFrame as DataFrame
from pandas._typing import CompressionOptions as CompressionOptions, ConvertersArg as ConvertersArg, DtypeArg as DtypeArg, FilePath as FilePath, ParseDatesArg as ParseDatesArg, ReadBuffer as ReadBuffer, StorageOptions as StorageOptions, TYPE_CHECKING as TYPE_CHECKING, XMLParsers as XMLParsers
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.errors import AbstractMethodError as AbstractMethodError, ParserError as ParserError
from pandas.io.common import file_exists as file_exists, get_handle as get_handle, infer_compression as infer_compression, is_fsspec_url as is_fsspec_url, is_url as is_url, stringify_path as stringify_path
from pandas.io.parsers import TextParser as TextParser
from pandas.util._decorators import deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments, doc as doc
from typing import Sequence

class _XMLFrameParser:
    path_or_buffer: Incomplete
    xpath: Incomplete
    namespaces: Incomplete
    elems_only: Incomplete
    attrs_only: Incomplete
    names: Incomplete
    dtype: Incomplete
    converters: Incomplete
    parse_dates: Incomplete
    encoding: Incomplete
    stylesheet: Incomplete
    iterparse: Incomplete
    is_style: Incomplete
    compression: Incomplete
    storage_options: Incomplete
    def __init__(self, path_or_buffer: Union[FilePath, ReadBuffer[bytes], ReadBuffer[str]], xpath: str, namespaces: Union[dict[str, str], None], elems_only: bool, attrs_only: bool, names: Union[Sequence[str], None], dtype: Union[DtypeArg, None], converters: Union[ConvertersArg, None], parse_dates: Union[ParseDatesArg, None], encoding: Union[str, None], stylesheet: Union[FilePath, ReadBuffer[bytes], ReadBuffer[str], None], iterparse: Union[dict[str, list[str]], None], compression: CompressionOptions, storage_options: StorageOptions) -> None: ...
    def parse_data(self) -> list[dict[str, Union[str, None]]]: ...

class _EtreeFrameParser(_XMLFrameParser):
    xml_doc: Incomplete
    def parse_data(self) -> list[dict[str, Union[str, None]]]: ...

class _LxmlFrameParser(_XMLFrameParser):
    xml_doc: Incomplete
    xsl_doc: Incomplete
    def parse_data(self) -> list[dict[str, Union[str, None]]]: ...

def get_data_from_filepath(filepath_or_buffer: Union[FilePath, bytes, ReadBuffer[bytes], ReadBuffer[str]], encoding: Union[str, None], compression: CompressionOptions, storage_options: StorageOptions) -> Union[str, bytes, ReadBuffer[bytes], ReadBuffer[str]]: ...
def preprocess_data(data) -> Union[io.StringIO, io.BytesIO]: ...
def read_xml(path_or_buffer: Union[FilePath, ReadBuffer[bytes], ReadBuffer[str]], xpath: str = ..., namespaces: Union[dict[str, str], None] = ..., elems_only: bool = ..., attrs_only: bool = ..., names: Union[Sequence[str], None] = ..., dtype: Union[DtypeArg, None] = ..., converters: Union[ConvertersArg, None] = ..., parse_dates: Union[ParseDatesArg, None] = ..., encoding: Union[str, None] = ..., parser: XMLParsers = ..., stylesheet: Union[FilePath, ReadBuffer[bytes], ReadBuffer[str], None] = ..., iterparse: Union[dict[str, list[str]], None] = ..., compression: CompressionOptions = ..., storage_options: StorageOptions = ...) -> DataFrame: ...
