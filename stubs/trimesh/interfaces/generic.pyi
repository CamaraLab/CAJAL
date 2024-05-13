from .. import exchange as exchange
from ..util import log as log
from _typeshed import Incomplete

class MeshScript:
    debug: Incomplete
    kwargs: Incomplete
    meshes: Incomplete
    script: Incomplete
    exchange: Incomplete
    def __init__(self, meshes, script, exchange: str = ..., debug: bool = ..., **kwargs) -> None: ...
    mesh_pre: Incomplete
    mesh_post: Incomplete
    script_out: Incomplete
    replacement: Incomplete
    def __enter__(self): ...
    def run(self, command): ...
    def __exit__(self, *args, **kwargs) -> None: ...
