from _typeshed import Incomplete
from typing import Iterable

class TablePlotter:
    cell_width: Incomplete
    cell_height: Incomplete
    font_size: Incomplete
    def __init__(self, cell_width: float = ..., cell_height: float = ..., font_size: float = ...) -> None: ...
    def plot(self, left, right, labels: Iterable[str] = ..., vertical: bool = ...): ...
