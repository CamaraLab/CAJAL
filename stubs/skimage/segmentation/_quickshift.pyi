from .._shared.filters import gaussian as gaussian
from ..color import rgb2lab as rgb2lab
from ..util import img_as_float as img_as_float

def quickshift(image, ratio: float = ..., kernel_size: int = ..., max_dist: int = ..., return_tree: bool = ..., sigma: int = ..., convert2lab: bool = ..., random_seed: int = ..., *, channel_axis: int = ...): ...
