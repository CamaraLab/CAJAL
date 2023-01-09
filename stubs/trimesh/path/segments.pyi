from .. import geometry as geometry, grouping as grouping, interval as interval, transformations as transformations, util as util
from ..constants import tol as tol
from _typeshed import Incomplete

def segments_to_parameters(segments): ...
def parameters_to_segments(origins, vectors, parameters): ...
def colinear_pairs(segments, radius: float = ..., angle: float = ..., length: Incomplete | None = ...): ...
def split(segments, points, atol: float = ...): ...
def unique(segments, digits: int = ...): ...
def overlap(origins, vectors, params): ...
def extrude(segments, height, double_sided: bool = ...): ...
def length(segments, summed: bool = ...): ...
def resample(segments, maxlen, return_index: bool = ..., return_count: bool = ...): ...
def to_svg(segments, digits: int = ..., matrix: Incomplete | None = ..., merge: bool = ...): ...