from . import color as color
from .. import caching as caching, grouping as grouping, util as util
from .base import Visuals as Visuals
from .material import PBRMaterial as PBRMaterial, SimpleMaterial as SimpleMaterial, empty_material as empty_material
from _typeshed import Incomplete

class TextureVisuals(Visuals):
    vertex_attributes: Incomplete
    material: Incomplete
    face_materials: Incomplete
    def __init__(self, uv: Incomplete | None = ..., material: Incomplete | None = ..., image: Incomplete | None = ..., face_materials: Incomplete | None = ...) -> None: ...
    @property
    def kind(self): ...
    @property
    def defined(self): ...
    def __hash__(self): ...
    @property
    def uv(self): ...
    @uv.setter
    def uv(self, values) -> None: ...
    def copy(self, uv: Incomplete | None = ...): ...
    def to_color(self): ...
    def face_subset(self, face_index): ...
    def update_vertices(self, mask) -> None: ...
    def update_faces(self, mask) -> None: ...
    def concatenate(self, others): ...

def unmerge_faces(faces, *args, **kwargs): ...
def power_resize(image, resample: int = ..., square: bool = ...): ...