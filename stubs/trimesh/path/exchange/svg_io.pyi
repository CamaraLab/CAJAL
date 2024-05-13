from ... import exceptions as exceptions, grouping as grouping, resources as resources, util as util
from ...constants import log as log, tol as tol
from ...transformations import planar_matrix as planar_matrix, transform_points as transform_points
from ...util import jsonify as jsonify
from ..arc import arc_center as arc_center
from ..entities import Arc as Arc, Bezier as Bezier, Line as Line
from _typeshed import Incomplete

def svg_to_path(file_obj: Incomplete | None = ..., file_type: Incomplete | None = ..., path_string: Incomplete | None = ...): ...
def transform_to_matrices(transform): ...
def export_svg(drawing, return_path: bool = ..., only_layers: Incomplete | None = ..., digits: Incomplete | None = ..., **kwargs): ...
