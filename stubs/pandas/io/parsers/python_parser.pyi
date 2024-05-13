from _typeshed import Incomplete
from collections import abc
from pandas import Index as Index, MultiIndex as MultiIndex
from pandas._typing import ArrayLike as ArrayLike, ReadCsvBuffer as ReadCsvBuffer, Scalar as Scalar
from pandas.core.dtypes.common import is_integer as is_integer
from pandas.core.dtypes.inference import is_dict_like as is_dict_like
from pandas.errors import EmptyDataError as EmptyDataError, ParserError as ParserError
from pandas.io.parsers.base_parser import ParserBase as ParserBase, parser_defaults as parser_defaults
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Hashable, IO, Literal, Mapping, Sequence

class PythonParser(ParserBase):
    data: Incomplete
    buf: Incomplete
    pos: int
    line_pos: int
    skiprows: Incomplete
    skipfunc: Incomplete
    skipfooter: Incomplete
    delimiter: Incomplete
    quotechar: Incomplete
    escapechar: Incomplete
    doublequote: Incomplete
    skipinitialspace: Incomplete
    lineterminator: Incomplete
    quoting: Incomplete
    skip_blank_lines: Incomplete
    names_passed: Incomplete
    has_index_names: bool
    verbose: Incomplete
    thousands: Incomplete
    decimal: Incomplete
    comment: Incomplete
    columns: Incomplete
    orig_names: Incomplete
    index_names: Incomplete
    num: Incomplete
    def __init__(self, f: Union[ReadCsvBuffer[str], list], **kwds) -> None: ...
    def read(self, rows: Union[int, None] = ...) -> tuple[Union[Index, None], Union[Sequence[Hashable], MultiIndex], Mapping[Hashable, ArrayLike]]: ...
    def get_chunk(self, size: Union[int, None] = ...) -> tuple[Union[Index, None], Union[Sequence[Hashable], MultiIndex], Mapping[Hashable, ArrayLike]]: ...

class FixedWidthReader(abc.Iterator):
    f: Incomplete
    buffer: Incomplete
    delimiter: Incomplete
    comment: Incomplete
    colspecs: Incomplete
    def __init__(self, f: Union[IO[str], ReadCsvBuffer[str]], colspecs: Union[list[tuple[int, int]], Literal['infer']], delimiter: Union[str, None], comment: Union[str, None], skiprows: Union[set[int], None] = ..., infer_nrows: int = ...) -> None: ...
    def get_rows(self, infer_nrows: int, skiprows: Union[set[int], None] = ...) -> list[str]: ...
    def detect_colspecs(self, infer_nrows: int = ..., skiprows: Union[set[int], None] = ...) -> list[tuple[int, int]]: ...
    def __next__(self) -> list[str]: ...

class FixedWidthFieldParser(PythonParser):
    colspecs: Incomplete
    infer_nrows: Incomplete
    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None: ...

def count_empty_vals(vals) -> int: ...
