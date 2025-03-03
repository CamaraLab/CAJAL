from .._shared.utils import check_nD as check_nD
from ..util import img_as_float as img_as_float
from _typeshed import Incomplete

class FeatureDetector:
    keypoints_: Incomplete
    def __init__(self) -> None: ...
    def detect(self, image) -> None: ...

class DescriptorExtractor:
    descriptors_: Incomplete
    def __init__(self) -> None: ...
    def extract(self, image, keypoints) -> None: ...

def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches, keypoints_color: str = ..., matches_color: Incomplete | None = ..., only_matches: bool = ..., alignment: str = ...) -> None: ...
