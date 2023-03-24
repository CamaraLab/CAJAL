from ..color import gray2rgb as gray2rgb
from ..draw import rectangle as rectangle
from ..util import img_as_float as img_as_float
from ._haar import haar_like_feature_coord_wrapper as haar_like_feature_coord_wrapper, haar_like_feature_wrapper as haar_like_feature_wrapper
from _typeshed import Incomplete

FEATURE_TYPE: Incomplete

def haar_like_feature_coord(width, height, feature_type: Incomplete | None = ...): ...
def haar_like_feature(int_image, r, c, width, height, feature_type: Incomplete | None = ..., feature_coord: Incomplete | None = ...): ...
def draw_haar_like_feature(image, r, c, width, height, feature_coord, color_positive_block=..., color_negative_block=..., alpha: float = ..., max_n_features: Incomplete | None = ..., random_state: Incomplete | None = ...): ...
