from ... import grouping as grouping, resources as resources, util as util
from ...constants import log as log
from ..arc import to_threepoint as to_threepoint
from ..entities import Arc as Arc, BSpline as BSpline, Line as Line, Text as Text
from _typeshed import Incomplete

XRECORD_METADATA: int
XRECORD_SENTINEL: str
XRECORD_MAX_LINE: int
XRECORD_MAX_INDEX: int

def load_dxf(file_obj, **kwargs): ...
def convert_entities(blob, blob_raw: Incomplete | None = ..., blocks: Incomplete | None = ..., return_name: bool = ...): ...
def export_dxf(path, only_layers: Incomplete | None = ...): ...
def bulge_to_arcs(lines, bulge, bulge_idx, is_closed: bool = ..., metadata: Incomplete | None = ...): ...
def get_key(blob, field, code): ...
