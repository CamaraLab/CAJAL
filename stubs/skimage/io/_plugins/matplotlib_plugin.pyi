from ..._shared.utils import warn as warn
from ...exposure import is_low_contrast as is_low_contrast
from _typeshed import Incomplete
from typing import NamedTuple

class ImageProperties(NamedTuple):
    signed: Incomplete
    out_of_range_float: Incomplete
    low_data_range: Incomplete
    unsupported_dtype: Incomplete

def imshow(image, ax: Incomplete | None = ..., show_cbar: Incomplete | None = ..., **kwargs): ...
def imshow_collection(ic, *args, **kwargs): ...

imread: Incomplete
