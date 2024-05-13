from .. import draw as draw
from .._shared.filters import gaussian as gaussian
from .._shared.utils import check_nD as check_nD
from ..color import gray2rgb as gray2rgb
from ..util.dtype import img_as_float as img_as_float
from _typeshed import Incomplete

def daisy(image, step: int = ..., radius: int = ..., rings: int = ..., histograms: int = ..., orientations: int = ..., normalization: str = ..., sigmas: Incomplete | None = ..., ring_radii: Incomplete | None = ..., visualize: bool = ...): ...
