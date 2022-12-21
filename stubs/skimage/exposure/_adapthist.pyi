from ..color.adapt_rgb import adapt_rgb as adapt_rgb, hsv_value as hsv_value
from ..exposure import rescale_intensity as rescale_intensity
from ..util import img_as_uint as img_as_uint
from _typeshed import Incomplete

NR_OF_GRAY: Incomplete

def equalize_adapthist(image, kernel_size: Incomplete | None = ..., clip_limit: float = ..., nbins: int = ...): ...
def clip_histogram(hist, clip_limit): ...
def map_histogram(hist, min_val, max_val, n_pixels): ...
