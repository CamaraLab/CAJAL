from _typeshed import Incomplete
from pandas import DataFrame as DataFrame, Index as Index, IndexSlice as IndexSlice, MultiIndex as MultiIndex, Series as Series, isna as isna
from pandas._config import get_option as get_option
from pandas._libs import lib as lib
from pandas._typing import Axis as Axis, Level as Level
from pandas.api.types import is_list_like as is_list_like
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import is_complex as is_complex, is_float as is_float, is_integer as is_integer
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

jinja2: Incomplete
BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, Dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties
CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence, Index]

class StylerRenderer:
    loader: Incomplete
    env: Incomplete
    template_html: Incomplete
    template_html_table: Incomplete
    template_html_style: Incomplete
    template_latex: Incomplete
    template_string: Incomplete
    data: Incomplete
    index: Incomplete
    columns: Incomplete
    uuid: Incomplete
    uuid_len: Incomplete
    table_styles: Incomplete
    table_attributes: Incomplete
    caption: Incomplete
    cell_ids: Incomplete
    css: Incomplete
    concatenated: Incomplete
    hide_index_names: bool
    hide_column_names: bool
    hide_index_: Incomplete
    hide_columns_: Incomplete
    hidden_rows: Incomplete
    hidden_columns: Incomplete
    ctx: Incomplete
    ctx_index: Incomplete
    ctx_columns: Incomplete
    cell_context: Incomplete
    tooltips: Incomplete
    def __init__(self, data: Union[DataFrame, Series], uuid: Union[str, None] = ..., uuid_len: int = ..., table_styles: Union[CSSStyles, None] = ..., table_attributes: Union[str, None] = ..., caption: Union[str, tuple, None] = ..., cell_ids: bool = ..., precision: Union[int, None] = ...) -> None: ...
    def format(self, formatter: Union[ExtFormatter, None] = ..., subset: Union[Subset, None] = ..., na_rep: Union[str, None] = ..., precision: Union[int, None] = ..., decimal: str = ..., thousands: Union[str, None] = ..., escape: Union[str, None] = ..., hyperlinks: Union[str, None] = ...) -> StylerRenderer: ...
    def format_index(self, formatter: Union[ExtFormatter, None] = ..., axis: Union[int, str] = ..., level: Union[Level, list[Level], None] = ..., na_rep: Union[str, None] = ..., precision: Union[int, None] = ..., decimal: str = ..., thousands: Union[str, None] = ..., escape: Union[str, None] = ..., hyperlinks: Union[str, None] = ...) -> StylerRenderer: ...
    def relabel_index(self, labels: Union[Sequence, Index], axis: Axis = ..., level: Union[Level, list[Level], None] = ...) -> StylerRenderer: ...

def format_table_styles(styles: CSSStyles) -> CSSStyles: ...
def non_reducing_slice(slice_: Subset): ...
def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList: ...
def refactor_levels(level: Union[Level, list[Level], None], obj: Index) -> list[int]: ...

class Tooltips:
    class_name: Incomplete
    class_properties: Incomplete
    tt_data: Incomplete
    table_styles: Incomplete
    def __init__(self, css_props: CSSProperties = ..., css_name: str = ..., tooltips: DataFrame = ...) -> None: ...
