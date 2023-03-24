from _typeshed import Incomplete
from matplotlib.axes import Axes as Axes
from matplotlib.lines import Line2D as Line2D
from pandas.core.dtypes.common import is_dict_like as is_dict_like
from pandas.core.dtypes.missing import remove_na_arraylike as remove_na_arraylike
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.plotting._matplotlib.core import LinePlot as LinePlot, MPLPlot as MPLPlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by as create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors as get_standard_colors
from pandas.plotting._matplotlib.tools import create_subplots as create_subplots, flatten_axes as flatten_axes, maybe_adjust_figure as maybe_adjust_figure
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import NamedTuple

class BoxPlot(LinePlot):
    class BP(NamedTuple):
        ax: Axes
        lines: dict[str, list[Line2D]]
    return_type: Incomplete
    def __init__(self, data, return_type: str = ..., **kwargs) -> None: ...
    def maybe_color_bp(self, bp) -> None: ...
    @property
    def orientation(self): ...
    @property
    def result(self): ...

def boxplot(data, column: Incomplete | None = ..., by: Incomplete | None = ..., ax: Incomplete | None = ..., fontsize: Incomplete | None = ..., rot: int = ..., grid: bool = ..., figsize: Incomplete | None = ..., layout: Incomplete | None = ..., return_type: Incomplete | None = ..., **kwds): ...
def boxplot_frame(self, column: Incomplete | None = ..., by: Incomplete | None = ..., ax: Incomplete | None = ..., fontsize: Incomplete | None = ..., rot: int = ..., grid: bool = ..., figsize: Incomplete | None = ..., layout: Incomplete | None = ..., return_type: Incomplete | None = ..., **kwds): ...
def boxplot_frame_groupby(grouped, subplots: bool = ..., column: Incomplete | None = ..., fontsize: Incomplete | None = ..., rot: int = ..., grid: bool = ..., ax: Incomplete | None = ..., figsize: Incomplete | None = ..., layout: Incomplete | None = ..., sharex: bool = ..., sharey: bool = ..., **kwds): ...
