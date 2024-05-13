from .._shared.filters import gaussian as gaussian
from .._shared.utils import check_nD as check_nD
from ..transform import integral_image as integral_image
from ..util import img_as_float as img_as_float
from .peak import peak_local_max as peak_local_max
from _typeshed import Incomplete

def blob_dog(image, min_sigma: int = ..., max_sigma: int = ..., sigma_ratio: float = ..., threshold: float = ..., overlap: float = ..., *, threshold_rel: Incomplete | None = ..., exclude_border: bool = ...): ...
def blob_log(image, min_sigma: int = ..., max_sigma: int = ..., num_sigma: int = ..., threshold: float = ..., overlap: float = ..., log_scale: bool = ..., *, threshold_rel: Incomplete | None = ..., exclude_border: bool = ...): ...
def blob_doh(image, min_sigma: int = ..., max_sigma: int = ..., num_sigma: int = ..., threshold: float = ..., overlap: float = ..., log_scale: bool = ..., *, threshold_rel: Incomplete | None = ...): ...
