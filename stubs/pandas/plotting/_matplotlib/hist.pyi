import numpy as np
from _typeshed import Incomplete
from matplotlib.axes import Axes as Axes
from pandas import DataFrame as DataFrame
from pandas._typing import PlottingOrientation as PlottingOrientation
from pandas.core.dtypes.common import is_integer as is_integer, is_list_like as is_list_like
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex
from pandas.core.dtypes.missing import isna as isna, remove_na_arraylike as remove_na_arraylike
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.plotting._matplotlib.core import LinePlot as LinePlot, MPLPlot as MPLPlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by as create_iter_data_given_by, reformat_hist_y_given_by as reformat_hist_y_given_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list as unpack_single_str_list
from pandas.plotting._matplotlib.tools import create_subplots as create_subplots, flatten_axes as flatten_axes, maybe_adjust_figure as maybe_adjust_figure, set_ticks_props as set_ticks_props
from typing import Literal

class HistPlot(LinePlot):
    bins: Incomplete
    bottom: Incomplete
    def __init__(self, data, bins: Union[int, np.ndarray, list[np.ndarray]] = ..., bottom: Union[int, np.ndarray] = ..., **kwargs) -> None: ...
    @property
    def orientation(self) -> PlottingOrientation: ...

class KdePlot(HistPlot):
    @property
    def orientation(self) -> Literal['vertical']: ...
    bw_method: Incomplete
    ind: Incomplete
    def __init__(self, data, bw_method: Incomplete | None = ..., ind: Incomplete | None = ..., **kwargs) -> None: ...

def hist_series(self, by: Incomplete | None = ..., ax: Incomplete | None = ..., grid: bool = ..., xlabelsize: Incomplete | None = ..., xrot: Incomplete | None = ..., ylabelsize: Incomplete | None = ..., yrot: Incomplete | None = ..., figsize: Incomplete | None = ..., bins: int = ..., legend: bool = ..., **kwds): ...
def hist_frame(data, column: Incomplete | None = ..., by: Incomplete | None = ..., grid: bool = ..., xlabelsize: Incomplete | None = ..., xrot: Incomplete | None = ..., ylabelsize: Incomplete | None = ..., yrot: Incomplete | None = ..., ax: Incomplete | None = ..., sharex: bool = ..., sharey: bool = ..., figsize: Incomplete | None = ..., layout: Incomplete | None = ..., bins: int = ..., legend: bool = ..., **kwds): ...
