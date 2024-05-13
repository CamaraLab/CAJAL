import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from pandas.core.dtypes.generic import ABCMultiIndex as ABCMultiIndex
from pandas.io.formats.format import DataFrameFormatter as DataFrameFormatter
from typing import Iterator

class RowStringConverter(ABC):
    fmt: Incomplete
    frame: Incomplete
    multicolumn: Incomplete
    multicolumn_format: Incomplete
    multirow: Incomplete
    clinebuf: Incomplete
    strcols: Incomplete
    strrows: Incomplete
    def __init__(self, formatter: DataFrameFormatter, multicolumn: bool = ..., multicolumn_format: Union[str, None] = ..., multirow: bool = ...) -> None: ...
    def get_strrow(self, row_num: int) -> str: ...
    @property
    def index_levels(self) -> int: ...
    @property
    def column_levels(self) -> int: ...
    @property
    def header_levels(self) -> int: ...

class RowStringIterator(RowStringConverter, metaclass=abc.ABCMeta):
    @abstractmethod
    def __iter__(self) -> Iterator[str]: ...

class RowHeaderIterator(RowStringIterator):
    def __iter__(self) -> Iterator[str]: ...

class RowBodyIterator(RowStringIterator):
    def __iter__(self) -> Iterator[str]: ...

class TableBuilderAbstract(ABC, metaclass=abc.ABCMeta):
    fmt: Incomplete
    column_format: Incomplete
    multicolumn: Incomplete
    multicolumn_format: Incomplete
    multirow: Incomplete
    caption: Incomplete
    short_caption: Incomplete
    label: Incomplete
    position: Incomplete
    def __init__(self, formatter: DataFrameFormatter, column_format: Union[str, None] = ..., multicolumn: bool = ..., multicolumn_format: Union[str, None] = ..., multirow: bool = ..., caption: Union[str, None] = ..., short_caption: Union[str, None] = ..., label: Union[str, None] = ..., position: Union[str, None] = ...) -> None: ...
    def get_result(self) -> str: ...
    @property
    @abstractmethod
    def env_begin(self) -> str: ...
    @property
    @abstractmethod
    def top_separator(self) -> str: ...
    @property
    @abstractmethod
    def header(self) -> str: ...
    @property
    @abstractmethod
    def middle_separator(self) -> str: ...
    @property
    @abstractmethod
    def env_body(self) -> str: ...
    @property
    @abstractmethod
    def bottom_separator(self) -> str: ...
    @property
    @abstractmethod
    def env_end(self) -> str: ...

class GenericTableBuilder(TableBuilderAbstract, metaclass=abc.ABCMeta):
    @property
    def header(self) -> str: ...
    @property
    def top_separator(self) -> str: ...
    @property
    def middle_separator(self) -> str: ...
    @property
    def env_body(self) -> str: ...

class LongTableBuilder(GenericTableBuilder):
    @property
    def env_begin(self) -> str: ...
    @property
    def middle_separator(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class RegularTableBuilder(GenericTableBuilder):
    @property
    def env_begin(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class TabularBuilder(GenericTableBuilder):
    @property
    def env_begin(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class LatexFormatter:
    fmt: Incomplete
    frame: Incomplete
    longtable: Incomplete
    multicolumn: Incomplete
    multicolumn_format: Incomplete
    multirow: Incomplete
    label: Incomplete
    position: Incomplete
    def __init__(self, formatter: DataFrameFormatter, longtable: bool = ..., column_format: Union[str, None] = ..., multicolumn: bool = ..., multicolumn_format: Union[str, None] = ..., multirow: bool = ..., caption: Union[str, tuple[str, str], None] = ..., label: Union[str, None] = ..., position: Union[str, None] = ...) -> None: ...
    def to_string(self) -> str: ...
    @property
    def builder(self) -> TableBuilderAbstract: ...
    @property
    def column_format(self) -> Union[str, None]: ...
    @column_format.setter
    def column_format(self, input_column_format: Union[str, None]) -> None: ...
