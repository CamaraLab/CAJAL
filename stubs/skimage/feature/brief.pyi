from .._shared.filters import gaussian as gaussian
from .._shared.utils import check_nD as check_nD
from .util import DescriptorExtractor as DescriptorExtractor
from _typeshed import Incomplete

class BRIEF(DescriptorExtractor):
    descriptor_size: Incomplete
    patch_size: Incomplete
    mode: Incomplete
    sigma: Incomplete
    sample_seed: Incomplete
    descriptors: Incomplete
    mask: Incomplete
    def __init__(self, descriptor_size: int = ..., patch_size: int = ..., mode: str = ..., sigma: int = ..., sample_seed: int = ...) -> None: ...
    def extract(self, image, keypoints) -> None: ...
