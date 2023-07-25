from .._shared.utils import check_nD as check_nD
from ..filters._gaussian import gaussian as gaussian
from ..util.dtype import dtype_limits as dtype_limits
from _typeshed import Incomplete

def canny(image, sigma: float = ..., low_threshold: Incomplete | None = ..., high_threshold: Incomplete | None = ..., mask: Incomplete | None = ..., use_quantiles: bool = ..., *, mode: str = ..., cval: float = ...): ...
