from .. import rendering as rendering, resources as resources, transformations as transformations, util as util, visual as visual
from ..caching import hash_fast as hash_fast
from ..constants import log as log, tol as tol
from ..util import unique_name as unique_name
from _typeshed import Incomplete

float32: Incomplete
uint32: Incomplete
uint8: Incomplete

def export_gltf(scene, include_normals: Incomplete | None = ..., merge_buffers: bool = ..., unitize_normals: bool = ..., tree_postprocessor: Incomplete | None = ...): ...
def export_glb(scene, include_normals: Incomplete | None = ..., unitize_normals: bool = ..., tree_postprocessor: Incomplete | None = ...): ...
def load_gltf(file_obj: Incomplete | None = ..., resolver: Incomplete | None = ..., ignore_broken: bool = ..., merge_primitives: bool = ..., **mesh_kwargs): ...
def load_glb(file_obj, resolver: Incomplete | None = ..., ignore_broken: bool = ..., merge_primitives: bool = ..., **mesh_kwargs): ...
def specular_to_pbr(specularFactor: Incomplete | None = ..., glossinessFactor: Incomplete | None = ..., specularGlossinessTexture: Incomplete | None = ..., diffuseTexture: Incomplete | None = ..., diffuseFactor: Incomplete | None = ..., **kwargs): ...
def validate(header): ...
def get_schema(): ...