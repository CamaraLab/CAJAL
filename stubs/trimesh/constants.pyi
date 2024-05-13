from .util import PY3 as PY3, log as log
from _typeshed import Incomplete

class ToleranceMesh:
    zero: Incomplete
    merge: float
    planar: float
    facet_threshold: int
    strict: bool
    def __init__(self, **kwargs) -> None: ...

class TolerancePath:
    zero: float
    merge: float
    planar: float
    buffer: float
    seg_frac: float
    seg_angle: Incomplete
    seg_angle_min: Incomplete
    seg_angle_frac: float
    aspect_frac: float
    radius_frac: float
    radius_min: float
    radius_max: int
    tangent: Incomplete
    strict: bool
    def __init__(self, **kwargs) -> None: ...

class ResolutionPath:
    seg_frac: float
    seg_angle: float
    max_sections: int
    min_sections: int
    export: str
    def __init__(self, **kwargs) -> None: ...

tol: Incomplete
tol_path: Incomplete
res_path: Incomplete

def log_time(method): ...
