from ..constants import log as log
from ..util import which as which
from .generic import MeshScript as MeshScript
from _typeshed import Incomplete

exists: Incomplete

def interface_scad(meshes, script, debug: bool = ..., **kwargs): ...
def boolean(meshes, operation: str = ..., debug: bool = ..., **kwargs): ...
