from _typeshed import Incomplete
from pandas._typing import FilePath as FilePath, StorageOptions as StorageOptions, WriteExcelBuffer as WriteExcelBuffer
from pandas.io.excel._base import ExcelWriter as ExcelWriter
from pandas.io.excel._util import combine_kwargs as combine_kwargs, validate_freeze_panes as validate_freeze_panes
from typing import Any
from xlsxwriter import Workbook

class _XlsxStyler:
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]]
    @classmethod
    def convert(cls, style_dict, num_format_str: Incomplete | None = ...): ...

class XlsxWriter(ExcelWriter):
    def __init__(self, path: Union[FilePath, WriteExcelBuffer, ExcelWriter], engine: Union[str, None] = ..., date_format: Union[str, None] = ..., datetime_format: Union[str, None] = ..., mode: str = ..., storage_options: StorageOptions = ..., if_sheet_exists: Union[str, None] = ..., engine_kwargs: Union[dict[str, Any], None] = ..., **kwargs) -> None: ...
    @property
    def book(self): ...
    @book.setter
    def book(self, other: Workbook) -> None: ...
    @property
    def sheets(self) -> dict[str, Any]: ...
