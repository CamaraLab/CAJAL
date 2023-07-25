from .. import util as util
from ..constants import log as log, tol as tol
from ..exceptions import ExceptionModule as ExceptionModule
from ..visual.color import to_float as to_float
from ..visual.material import SimpleMaterial as SimpleMaterial
from ..visual.texture import TextureVisuals as TextureVisuals, unmerge_faces as unmerge_faces
from _typeshed import Incomplete

def load_obj(file_obj, resolver: Incomplete | None = ..., split_object: bool = ..., group_material: bool = ..., skip_materials: bool = ..., maintain_order: bool = ..., **kwargs): ...
def parse_mtl(mtl, resolver: Incomplete | None = ...): ...
def export_obj(mesh, include_normals: bool = ..., include_color: bool = ..., include_texture: bool = ..., return_texture: bool = ..., write_texture: bool = ..., resolver: Incomplete | None = ..., digits: int = ..., header: str = ...): ...
