from _typeshed import Incomplete
from pandas.io.formats.format import DataFrameFormatter as DataFrameFormatter
from pandas.io.formats.printing import pprint_thing as pprint_thing

class StringFormatter:
    fmt: Incomplete
    adj: Incomplete
    frame: Incomplete
    line_width: Incomplete
    def __init__(self, fmt: DataFrameFormatter, line_width: Union[int, None] = ...) -> None: ...
    def to_string(self) -> str: ...
