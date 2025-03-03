from .. import constants as constants, grouping as grouping, util as util
from ..exceptions import ExceptionModule as ExceptionModule
from .util import is_ccw as is_ccw
from _typeshed import Incomplete

def vertex_graph(entities): ...
def vertex_to_entity_path(vertex_path, graph, entities, vertices: Incomplete | None = ...): ...
def closed_paths(entities, vertices): ...
def discretize_path(entities, vertices, path, scale: float = ...): ...

class PathSample:
    length: Incomplete
    def __init__(self, points) -> None: ...
    def sample(self, distances): ...
    def truncate(self, distance): ...

def resample_path(points, count: Incomplete | None = ..., step: Incomplete | None = ..., step_round: bool = ...): ...
def split(path): ...
