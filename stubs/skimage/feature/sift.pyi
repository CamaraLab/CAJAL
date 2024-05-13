from .._shared.filters import gaussian as gaussian
from .._shared.utils import check_nD as check_nD
from ..feature.util import DescriptorExtractor as DescriptorExtractor, FeatureDetector as FeatureDetector
from ..transform import rescale as rescale
from ..util import img_as_float as img_as_float
from _typeshed import Incomplete

class SIFT(FeatureDetector, DescriptorExtractor):
    upsampling: Incomplete
    n_octaves: Incomplete
    n_scales: Incomplete
    sigma_min: Incomplete
    sigma_in: Incomplete
    c_dog: Incomplete
    c_edge: Incomplete
    n_bins: Incomplete
    lambda_ori: Incomplete
    c_max: Incomplete
    lambda_descr: Incomplete
    n_hist: Incomplete
    n_ori: Incomplete
    delta_min: Incomplete
    float_dtype: Incomplete
    scalespace_sigmas: Incomplete
    keypoints: Incomplete
    positions: Incomplete
    sigmas: Incomplete
    scales: Incomplete
    orientations: Incomplete
    octaves: Incomplete
    descriptors: Incomplete
    def __init__(self, upsampling: int = ..., n_octaves: int = ..., n_scales: int = ..., sigma_min: float = ..., sigma_in: float = ..., c_dog=..., c_edge: int = ..., n_bins: int = ..., lambda_ori: float = ..., c_max: float = ..., lambda_descr: int = ..., n_hist: int = ..., n_ori: int = ...) -> None: ...
    @property
    def deltas(self): ...
    def detect(self, image) -> None: ...
    def extract(self, image) -> None: ...
    def detect_and_extract(self, image) -> None: ...
