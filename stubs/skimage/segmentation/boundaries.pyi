from ..color import gray2rgb as gray2rgb
from ..morphology import dilation as dilation, erosion as erosion, square as square
from ..util import img_as_float as img_as_float, view_as_windows as view_as_windows
from _typeshed import Incomplete

def find_boundaries(label_img, connectivity: int = ..., mode: str = ..., background: int = ...): ...
def mark_boundaries(image, label_img, color=..., outline_color: Incomplete | None = ..., mode: str = ..., background_label: int = ...): ...
