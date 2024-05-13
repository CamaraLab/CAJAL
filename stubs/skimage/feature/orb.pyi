from .._shared.utils import check_nD as check_nD
from ..feature import corner_fast as corner_fast, corner_harris as corner_harris, corner_orientations as corner_orientations, corner_peaks as corner_peaks
from ..feature.util import DescriptorExtractor as DescriptorExtractor, FeatureDetector as FeatureDetector
from ..transform import pyramid_gaussian as pyramid_gaussian
from _typeshed import Incomplete

OFAST_MASK: Incomplete
OFAST_UMAX: Incomplete

class ORB(FeatureDetector, DescriptorExtractor):
    downscale: Incomplete
    n_scales: Incomplete
    n_keypoints: Incomplete
    fast_n: Incomplete
    fast_threshold: Incomplete
    harris_k: Incomplete
    keypoints: Incomplete
    scales: Incomplete
    responses: Incomplete
    orientations: Incomplete
    descriptors: Incomplete
    def __init__(self, downscale: float = ..., n_scales: int = ..., n_keypoints: int = ..., fast_n: int = ..., fast_threshold: float = ..., harris_k: float = ...) -> None: ...
    def detect(self, image) -> None: ...
    mask_: Incomplete
    def extract(self, image, keypoints, scales, orientations) -> None: ...
    def detect_and_extract(self, image) -> None: ...
