from .._shared.utils import check_nD as check_nD
from ..color import gray2rgb as gray2rgb
from ..util import img_as_float as img_as_float
from _typeshed import Incomplete

def graycomatrix(image, distances, angles, levels: Incomplete | None = ..., symmetric: bool = ..., normed: bool = ...): ...
def graycoprops(P, prop: str = ...): ...
def local_binary_pattern(image, P, R, method: str = ...): ...
def multiblock_lbp(int_image, r, c, width, height): ...
def draw_multiblock_lbp(image, r, c, width, height, lbp_code: int = ..., color_greater_block=..., color_less_block=..., alpha: float = ...): ...
