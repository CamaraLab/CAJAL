from _typeshed import Incomplete
from pandas import DataFrame as DataFrame, Index as Index, MultiIndex as MultiIndex, PeriodIndex as PeriodIndex
from pandas._libs.lib import is_list_like as is_list_like
from pandas._typing import IndexLabel as IndexLabel, StorageOptions as StorageOptions
from pandas.core.dtypes import missing as missing
from pandas.core.dtypes.common import is_float as is_float, is_scalar as is_scalar
from pandas.io.formats._color_data import CSS4_COLORS as CSS4_COLORS
from pandas.io.formats.css import CSSResolver as CSSResolver, CSSWarning as CSSWarning
from pandas.io.formats.format import get_level_lengths as get_level_lengths
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

class ExcelCell:
    __fields__: Incomplete
    row: Incomplete
    col: Incomplete
    val: Incomplete
    style: Incomplete
    mergestart: Incomplete
    mergeend: Incomplete
    def __init__(self, row: int, col: int, val, style: Incomplete | None = ..., mergestart: Union[int, None] = ..., mergeend: Union[int, None] = ...) -> None: ...

class CssExcelCell(ExcelCell):
    def __init__(self, row: int, col: int, val, style: Union[dict, None], css_styles: Union[dict[tuple[int, int], list[tuple[str, Any]]], None], css_row: int, css_col: int, css_converter: Union[Callable, None], **kwargs) -> None: ...

class CSSToExcelConverter:
    NAMED_COLORS: Incomplete
    VERTICAL_MAP: Incomplete
    BOLD_MAP: Incomplete
    ITALIC_MAP: Incomplete
    FAMILY_MAP: Incomplete
    inherited: Union[dict[str, str], None]
    def __init__(self, inherited: Union[str, None] = ...) -> None: ...
    compute_css: Incomplete
    def __call__(self, declarations: Union[str, frozenset[tuple[str, str]]]) -> dict[str, dict[str, str]]: ...
    def build_xlstyle(self, props: Mapping[str, str]) -> dict[str, dict[str, str]]: ...
    def build_alignment(self, props: Mapping[str, str]) -> dict[str, Union[bool, str, None]]: ...
    def build_border(self, props: Mapping[str, str]) -> dict[str, dict[str, Union[str, None]]]: ...
    def build_fill(self, props: Mapping[str, str]): ...
    def build_number_format(self, props: Mapping[str, str]) -> dict[str, Union[str, None]]: ...
    def build_font(self, props: Mapping[str, str]) -> dict[str, Union[bool, float, str, None]]: ...
    def color_to_excel(self, val: Union[str, None]) -> Union[str, None]: ...

class ExcelFormatter:
    max_rows: Incomplete
    max_cols: Incomplete
    rowcounter: int
    na_rep: Incomplete
    styler: Incomplete
    style_converter: Incomplete
    df: Incomplete
    columns: Incomplete
    float_format: Incomplete
    index: Incomplete
    index_label: Incomplete
    header: Incomplete
    merge_cells: Incomplete
    inf_rep: Incomplete
    def __init__(self, df, na_rep: str = ..., float_format: Union[str, None] = ..., cols: Union[Sequence[Hashable], None] = ..., header: Union[Sequence[Hashable], bool] = ..., index: bool = ..., index_label: Union[IndexLabel, None] = ..., merge_cells: bool = ..., inf_rep: str = ..., style_converter: Union[Callable, None] = ...) -> None: ...
    @property
    def header_style(self) -> dict[str, dict[str, Union[str, bool]]]: ...
    def get_formatted_cells(self) -> Iterable[ExcelCell]: ...
    def write(self, writer, sheet_name: str = ..., startrow: int = ..., startcol: int = ..., freeze_panes: Union[tuple[int, int], None] = ..., engine: Union[str, None] = ..., storage_options: StorageOptions = ...) -> None: ...
