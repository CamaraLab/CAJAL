from .. import resources as resources, util as util
from ..constants import log as log
from .generic import MeshScript as MeshScript
from _typeshed import Incomplete

pf: Incomplete
exists: Incomplete

def boolean(meshes, operation: str = ..., debug: bool = ...): ...
def unwrap(mesh, angle_limit: int = ..., island_margin: float = ..., debug: bool = ...): ...
