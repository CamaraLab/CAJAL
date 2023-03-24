from .._shared.utils import deprecate_multichannel_kwarg as deprecate_multichannel_kwarg, warn as warn
from _typeshed import Incomplete

SHAPE_GENERATORS: Incomplete
SHAPE_CHOICES: Incomplete

def random_shapes(image_shape, max_shapes, min_shapes: int = ..., min_size: int = ..., max_size: Incomplete | None = ..., multichannel: bool = ..., num_channels: int = ..., shape: Incomplete | None = ..., intensity_range: Incomplete | None = ..., allow_overlap: bool = ..., num_trials: int = ..., random_seed: Incomplete | None = ..., *, channel_axis: int = ...): ...
