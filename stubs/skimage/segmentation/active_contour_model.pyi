from .._shared.utils import deprecate_kwarg as deprecate_kwarg
from ..filters import sobel as sobel
from ..util import img_as_float as img_as_float

def active_contour(image, snake, alpha: float = ..., beta: float = ..., w_line: int = ..., w_edge: int = ..., gamma: float = ..., max_px_move: float = ..., max_num_iter: int = ..., convergence: float = ..., *, boundary_condition: str = ..., coordinates: str = ...): ...
