from .._shared.utils import check_nD as check_nD
from ..feature import structure_tensor as structure_tensor
from ..feature.util import FeatureDetector as FeatureDetector
from ..morphology import octagon as octagon, star as star
from ..transform import integral_image as integral_image
from _typeshed import Incomplete

OCTAGON_OUTER_SHAPE: Incomplete
OCTAGON_INNER_SHAPE: Incomplete
STAR_SHAPE: Incomplete
STAR_FILTER_SHAPE: Incomplete

class CENSURE(FeatureDetector):
    min_scale: Incomplete
    max_scale: Incomplete
    mode: Incomplete
    non_max_threshold: Incomplete
    line_threshold: Incomplete
    keypoints: Incomplete
    scales: Incomplete
    def __init__(self, min_scale: int = ..., max_scale: int = ..., mode: str = ..., non_max_threshold: float = ..., line_threshold: int = ...) -> None: ...
    def detect(self, image) -> None: ...
