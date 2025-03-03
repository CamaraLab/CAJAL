from .._shared.utils import channel_as_last_axis as channel_as_last_axis, convert_to_float as convert_to_float, deprecate_multichannel_kwarg as deprecate_multichannel_kwarg, get_bound_method_class as get_bound_method_class, safe_as_int as safe_as_int, warn as warn
from ..measure import block_reduce as block_reduce
from ._geometric import AffineTransform as AffineTransform, ProjectiveTransform as ProjectiveTransform, SimilarityTransform as SimilarityTransform
from _typeshed import Incomplete

HOMOGRAPHY_TRANSFORMS: Incomplete

def resize(image, output_shape, order: Incomplete | None = ..., mode: str = ..., cval: int = ..., clip: bool = ..., preserve_range: bool = ..., anti_aliasing: Incomplete | None = ..., anti_aliasing_sigma: Incomplete | None = ...): ...
def rescale(image, scale, order: Incomplete | None = ..., mode: str = ..., cval: int = ..., clip: bool = ..., preserve_range: bool = ..., multichannel: bool = ..., anti_aliasing: Incomplete | None = ..., anti_aliasing_sigma: Incomplete | None = ..., *, channel_axis: Incomplete | None = ...): ...
def rotate(image, angle, resize: bool = ..., center: Incomplete | None = ..., order: Incomplete | None = ..., mode: str = ..., cval: int = ..., clip: bool = ..., preserve_range: bool = ...): ...
def downscale_local_mean(image, factors, cval: int = ..., clip: bool = ...): ...
def swirl(image, center: Incomplete | None = ..., strength: int = ..., radius: int = ..., rotation: int = ..., output_shape: Incomplete | None = ..., order: Incomplete | None = ..., mode: str = ..., cval: int = ..., clip: bool = ..., preserve_range: bool = ...): ...
def warp_coords(coord_map, shape, dtype=...): ...
def warp(image, inverse_map, map_args=..., output_shape: Incomplete | None = ..., order: Incomplete | None = ..., mode: str = ..., cval: float = ..., clip: bool = ..., preserve_range: bool = ...): ...
def warp_polar(image, center: Incomplete | None = ..., *, radius: Incomplete | None = ..., output_shape: Incomplete | None = ..., scaling: str = ..., multichannel: bool = ..., channel_axis: Incomplete | None = ..., **kwargs): ...
def resize_local_mean(image, output_shape, grid_mode: bool = ..., preserve_range: bool = ..., *, channel_axis: Incomplete | None = ...): ...
