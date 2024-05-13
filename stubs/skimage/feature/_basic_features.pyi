from _typeshed import Incomplete
from skimage import feature as feature, filters as filters
from skimage._shared import utils as utils
from skimage.util.dtype import img_as_float32 as img_as_float32

def multiscale_basic_features(image, multichannel: bool = ..., intensity: bool = ..., edges: bool = ..., texture: bool = ..., sigma_min: float = ..., sigma_max: int = ..., num_sigma: Incomplete | None = ..., num_workers: Incomplete | None = ..., *, channel_axis: Incomplete | None = ...): ...
