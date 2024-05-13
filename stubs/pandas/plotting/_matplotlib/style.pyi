from matplotlib.colors import Colormap as Colormap
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.plotting._matplotlib.compat import mpl_ge_3_6_0 as mpl_ge_3_6_0
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Collection, Sequence, Union

Color = Union[str, Sequence[float]]

def get_standard_colors(num_colors: int, colormap: Union[Colormap, None] = ..., color_type: str = ..., color: Union[dict[str, Color], Color, Collection[Color], None] = ...): ...
