from .._shared.version_requirements import require as require
from ..draw import polygon as polygon

LEFT_CLICK: int
RIGHT_CLICK: int

def manual_polygon_segmentation(image, alpha: float = ..., return_all: bool = ...): ...
def manual_lasso_segmentation(image, alpha: float = ..., return_all: bool = ...): ...
