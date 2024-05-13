from _typeshed import Incomplete
from pandas.errors import CSSWarning as CSSWarning
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Generator, Iterable, Iterator

class CSSResolver:
    UNIT_RATIOS: Incomplete
    FONT_SIZE_RATIOS: Incomplete
    MARGIN_RATIOS: Incomplete
    BORDER_WIDTH_RATIOS: Incomplete
    BORDER_STYLES: Incomplete
    SIDE_SHORTHANDS: Incomplete
    SIDES: Incomplete
    CSS_EXPANSIONS: Incomplete
    def __call__(self, declarations: Union[str, Iterable[tuple[str, str]]], inherited: Union[dict[str, str], None] = ...) -> dict[str, str]: ...
    def size_to_pt(self, in_val, em_pt: Incomplete | None = ..., conversions=...): ...
    def atomize(self, declarations: Iterable) -> Generator[tuple[str, str], None, None]: ...
    def parse(self, declarations_str: str) -> Iterator[tuple[str, str]]: ...
