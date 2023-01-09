from . import boolean as boolean, caching as caching, comparison as comparison, convex as convex, curvature as curvature, decomposition as decomposition, geometry as geometry, graph as graph, grouping as grouping, inertia as inertia, intersections as intersections, permutate as permutate, poses as poses, proximity as proximity, ray as ray, registration as registration, remesh as remesh, repair as repair, sample as sample, smoothing as smoothing, transformations as transformations, triangles as triangles, units as units, util as util
from .constants import log as log, log_time as log_time, tol as tol
from .exchange.export import export_mesh as export_mesh
from .parent import Geometry3D as Geometry3D
from .scene import Scene as Scene
from .visual import TextureVisuals as TextureVisuals, create_visual as create_visual
from _typeshed import Incomplete

class Trimesh(Geometry3D):
    vertex_normals: Incomplete
    ray: Incomplete
    permutate: Incomplete
    nearest: Incomplete
    metadata: Incomplete
    face_attributes: Incomplete
    vertex_attributes: Incomplete
    def __init__(self, vertices: Incomplete | None = ..., faces: Incomplete | None = ..., face_normals: Incomplete | None = ..., vertex_normals: Incomplete | None = ..., face_colors: Incomplete | None = ..., vertex_colors: Incomplete | None = ..., face_attributes: Incomplete | None = ..., vertex_attributes: Incomplete | None = ..., metadata: Incomplete | None = ..., process: bool = ..., validate: bool = ..., merge_tex: Incomplete | None = ..., merge_norm: Incomplete | None = ..., use_embree: bool = ..., initial_cache: Incomplete | None = ..., visual: Incomplete | None = ..., **kwargs) -> None: ...
    def process(self, validate: bool = ..., merge_tex: Incomplete | None = ..., merge_norm: Incomplete | None = ...): ...
    @property
    def faces(self): ...
    @faces.setter
    def faces(self, values): ...
    def faces_sparse(self): ...
    @property
    def face_normals(self): ...
    @face_normals.setter
    def face_normals(self, values) -> None: ...
    @property
    def vertices(self): ...
    @vertices.setter
    def vertices(self, values) -> None: ...
    def vertex_faces(self): ...
    def bounds(self): ...
    def extents(self): ...
    def scale(self): ...
    def centroid(self): ...
    @property
    def center_mass(self): ...
    @center_mass.setter
    def center_mass(self, cm) -> None: ...
    @property
    def density(self): ...
    @density.setter
    def density(self, value) -> None: ...
    @property
    def volume(self): ...
    @property
    def mass(self): ...
    @property
    def moment_inertia(self): ...
    def principal_inertia_components(self): ...
    @property
    def principal_inertia_vectors(self): ...
    def principal_inertia_transform(self): ...
    def symmetry(self): ...
    @property
    def symmetry_axis(self): ...
    @property
    def symmetry_section(self): ...
    def triangles(self): ...
    def triangles_tree(self): ...
    def triangles_center(self): ...
    def triangles_cross(self): ...
    def edges(self): ...
    def edges_face(self): ...
    def edges_unique(self): ...
    def edges_unique_length(self): ...
    def edges_unique_inverse(self): ...
    def edges_sorted(self): ...
    def edges_sorted_tree(self): ...
    def edges_sparse(self): ...
    def body_count(self): ...
    def faces_unique_edges(self): ...
    def euler_number(self): ...
    def referenced_vertices(self): ...
    @property
    def units(self): ...
    @units.setter
    def units(self, value) -> None: ...
    def convert_units(self, desired, guess: bool = ...): ...
    def merge_vertices(self, merge_tex: Incomplete | None = ..., merge_norm: Incomplete | None = ..., digits_vertex: Incomplete | None = ..., digits_norm: Incomplete | None = ..., digits_uv: Incomplete | None = ...) -> None: ...
    def update_vertices(self, mask, inverse: Incomplete | None = ...) -> None: ...
    def update_faces(self, mask) -> None: ...
    def remove_infinite_values(self) -> None: ...
    def remove_duplicate_faces(self) -> None: ...
    def rezero(self) -> None: ...
    def split(self, **kwargs): ...
    def face_adjacency(self): ...
    def face_neighborhood(self): ...
    def face_adjacency_edges(self): ...
    def face_adjacency_edges_tree(self): ...
    def face_adjacency_angles(self): ...
    def face_adjacency_projections(self): ...
    def face_adjacency_convex(self): ...
    def face_adjacency_unshared(self): ...
    def face_adjacency_radius(self): ...
    def face_adjacency_span(self): ...
    def integral_mean_curvature(self): ...
    def vertex_adjacency_graph(self): ...
    def vertex_neighbors(self): ...
    def is_winding_consistent(self): ...
    def is_watertight(self): ...
    def is_volume(self): ...
    @property
    def is_empty(self): ...
    def is_convex(self): ...
    def kdtree(self): ...
    def remove_degenerate_faces(self, height=...): ...
    def facets(self): ...
    def facets_area(self): ...
    def facets_normal(self): ...
    def facets_origin(self): ...
    def facets_boundary(self): ...
    def facets_on_hull(self): ...
    def fix_normals(self, multibody: Incomplete | None = ...) -> None: ...
    def fill_holes(self): ...
    def register(self, other, **kwargs): ...
    def compute_stable_poses(self, center_mass: Incomplete | None = ..., sigma: float = ..., n_samples: int = ..., threshold: float = ...): ...
    def subdivide(self, face_index: Incomplete | None = ...): ...
    def subdivide_to_size(self, max_edge, max_iter: int = ..., return_index: bool = ...): ...
    def smoothed(self, **kwargs): ...
    @property
    def visual(self): ...
    @visual.setter
    def visual(self, value) -> None: ...
    def section(self, plane_normal, plane_origin, **kwargs): ...
    def section_multiplane(self, plane_origin, plane_normal, heights): ...
    def slice_plane(self, plane_origin, plane_normal, cap: bool = ..., face_index: Incomplete | None = ..., cached_dots: Incomplete | None = ..., **kwargs): ...
    def unwrap(self, image: Incomplete | None = ...): ...
    def convex_hull(self): ...
    def sample(self, count, return_index: bool = ..., face_weight: Incomplete | None = ...): ...
    def remove_unreferenced_vertices(self) -> None: ...
    def unmerge_vertices(self) -> None: ...
    def apply_transform(self, matrix): ...
    def voxelized(self, pitch, method: str = ..., **kwargs): ...
    def as_open3d(self): ...
    def simplify_quadratic_decimation(self, face_count): ...
    def outline(self, face_ids: Incomplete | None = ..., **kwargs): ...
    def projected(self, normal, **kwargs): ...
    def area(self): ...
    def area_faces(self): ...
    def mass_properties(self): ...
    def invert(self) -> None: ...
    def scene(self, **kwargs): ...
    def show(self, **kwargs): ...
    def submesh(self, faces_sequence, **kwargs): ...
    def identifier(self): ...
    def identifier_hash(self): ...
    @property
    def identifier_md5(self): ...
    def export(self, file_obj: Incomplete | None = ..., file_type: Incomplete | None = ..., **kwargs): ...
    def to_dict(self): ...
    def convex_decomposition(self, maxhulls: int = ..., **kwargs): ...
    def union(self, other, engine: Incomplete | None = ..., **kwargs): ...
    def difference(self, other, engine: Incomplete | None = ..., **kwargs): ...
    def intersection(self, other, engine: Incomplete | None = ..., **kwargs): ...
    def contains(self, points): ...
    def face_angles(self): ...
    def face_angles_sparse(self): ...
    def vertex_defects(self): ...
    def vertex_degree(self): ...
    def face_adjacency_tree(self): ...
    def copy(self, include_cache: bool = ...): ...
    def __deepcopy__(self, *args): ...
    def __copy__(self, *args): ...
    def eval_cached(self, statement, *args): ...
    def __add__(self, other): ...