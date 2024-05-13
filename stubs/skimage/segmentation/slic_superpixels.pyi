from .._shared import utils as utils
from .._shared.filters import gaussian as gaussian
from ..color import rgb2lab as rgb2lab
from ..util import img_as_float as img_as_float, regular_grid as regular_grid
from _typeshed import Incomplete

def slic(image, n_segments: int = ..., compactness: float = ..., max_num_iter: int = ..., sigma: int = ..., spacing: Incomplete | None = ..., multichannel: bool = ..., convert2lab: Incomplete | None = ..., enforce_connectivity: bool = ..., min_size_factor: float = ..., max_size_factor: int = ..., slic_zero: bool = ..., start_label: int = ..., mask: Incomplete | None = ..., *, channel_axis: int = ...): ...
