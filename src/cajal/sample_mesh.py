"""
Functions for sampling points from a triangular mesh
"""

from __future__ import annotations
import os
import sys
import csv
import numpy as np
import numpy.typing as npt
import potpourri3d as pp3d
import networkx as nx
import networkx.algorithms.shortest_paths as nxsp
from scipy.spatial.distance import squareform, cdist, euclidean, pdist
import trimesh
import itertools as it
import warnings
from typing import (
    Tuple,
    List,
    Set,
    Dict,
    Optional,
    Iterator,
    Callable,
    Literal,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias

    VertexArray: TypeAlias = npt.NDArray[np.float64]
    FaceArray: TypeAlias = npt.NDArray[np.int_]

from pathos.pools import ProcessPool

from .utilities import write_csv_block

# We represent a mesh as a pair (vertices, faces) : Tuple[VertexArray,FaceArray].
# A VertexArray is a numpy array of shape (n, 3), where n is the number of vertices in the mesh.
# Each row of a VertexArray is an XYZ coordinate triple for a point in the mesh.

# A FaceArray is a numpy array of shape (m, 3) where m is the number of faces in the mesh.
# Each row of a FaceArray is a list of three natural numbers, corresponding to indices
# in the corresponding VertexArray,
# representing triangular faces joining those three points.


def read_obj(file_path: str) -> Tuple[VertexArray, FaceArray]:
    """
    Reads in the vertices and triangular faces of a .obj file.

    :param file_path: Path to .obj file

    :return: Ordered pair `(vertices, faces)`, where:

        * `vertices` is an array of 3D floating-point coordinates of shape `(n,3)`, \
             where `n` is the number of vertices in the mesh
        * `faces` is an array of shape `(m,3)`, where `m` is the number of \
           faces; the `k`-th row gives the indices for the vertices in the `k`-th face.
    """
    obj_file = open(file_path, "r")
    obj_split = csv.reader(obj_file, delimiter=" ")
    vertices = []
    faces = []
    for line in obj_split:
        if line[0] == "v":
            vertices.append([float(x) for x in line[1:]])
        elif line[0] == "f":
            faces.append([float(x) for x in line[1:]])
        # Skipping over any vertex textures or normals
    obj_file.close()
    return np.array(vertices), (np.array(faces) - 1).astype("int64")


def connect_mesh(vertices: VertexArray, faces: FaceArray) -> FaceArray:
    """
    Args:
        * vertices : VertexArray, the vertices of a mesh
        * faces : FaceArray, the faces of a mesh

    Returns:
        numpy int array "new_faces" of shape (m+k,3) which extends the \
        original faces array; the mesh represented by (vertices, new_faces) is \
        connected.
    """

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # The nodes of "graph" are natural numbers which are array indices for the array "vertices".
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    # The elements of connected_components[k] are nodes from "graph", i.e.,
    # natural number array indices.
    connected_components: List[Set[int]] = [
        i[1] for i in enumerate(nx.connected_components(graph))
    ]
    if len(connected_components) == 1:
        return faces

    # We build a graph spt_graph whose nodes are connected components,
    # represented as integer indices for the list connected_components.
    spt_graph = nx.Graph()

    # spt_graph is a weighted graph; the weight between components i and j is the
    # minimum distance in 3D space from a node in connected_components[i] to a node
    # in connected_components[j].

    # Then, we form a minimum spanning tree T for spt_graph. If T contains an
    # edge between i and j, we add a new face to the mesh connecting components
    # i and j.  If there are k connected components of the mesh to begin with,
    # then a total of k-1 new faces will be added to the mesh

    # We build a a minimum spanning tree of edges I can add between components

    # connections will associate to each pair of components i, j a new face
    # straddling the two components.  Not all new faces in "connections" will
    # be added to the mesh - only the ones that lie along an edge in the
    # minimal spanning tree.
    connections: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    # Add components as nodes in graph
    for i in range(len(connected_components)):
        spt_graph.add_node(i)
    # For every pair of components, find nearest points
    for i in range(len(connected_components)):
        for j in range(i + 1, len(connected_components)):
            # Set bigger component as reference so we can make KDTree of that and loop over smaller
            ref, query = (
                (i, j)
                if len(connected_components[i]) > len(connected_components[j])
                else (j, i)
            )
            ref_ids = list(connected_components[ref])
            query_ids = list(connected_components[query])
            # dist_ij is a rectangular matrix of floats.
            # dist_ij[i, j] is the distance between the i-th node of component ref_ids
            # and the j-th node of query_ids.
            dist_ij = cdist(vertices[ref_ids], vertices[query_ids])
            nearest: Tuple[int, int]
            nearest = np.unravel_index(dist_ij.argmin(), dist_ij.shape)  # type: ignore[assignment]
            next_nearest = np.argpartition(dist_ij[nearest[0]], 1)[1]
            face: Tuple[int, int, int] = (
                ref_ids[nearest[0]],
                query_ids[nearest[1]],
                query_ids[next_nearest],
            )
            # Save face so I know to add it if this edge is in SPT
            connections[(i, j)] = face
            spt_graph.add_edge(i, j, weight=dist_ij[nearest])
    new_faces: List[Tuple[int, int, int]] = []
    for u, v, _ in nx.minimum_spanning_edges(spt_graph):
        new_faces.append(connections[(min(u, v), max(u, v))])
    return np.vstack([faces, np.array(new_faces)])


def disconnect_mesh(
    vertices: VertexArray, faces: FaceArray
) -> List[Tuple[VertexArray, FaceArray]]:
    """
    Returns the list of connected submeshes of the given mesh, as\
     ordered pairs (vertices_i, faces_i).

    Args:
        * vertices (VertexArray) : vertices of the mesh
        * faces (FaceArray): faces of the mesh

    Returns:
        * List of ordered pairs (vertices_i, faces_i) corresponding to \
          the connected components of the original mesh
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    connected_components = [i[1] for i in enumerate(nx.connected_components(graph))]
    if len(connected_components) == 1:
        return [(vertices, faces)]

    disconn_meshes = []
    for component in connected_components:
        v_ids = list(component)
        new_vertices = vertices[v_ids]
        index = np.min(faces)
        v_ids = [v_id + index for v_id in v_ids]
        keep_faces = faces[np.sum(np.isin(faces, v_ids), axis=1) == 3]
        convert_ids = dict(zip(v_ids, range(index, len(v_ids) + index)))
        new_faces = np.array([convert_ids[x] for x in keep_faces.flatten()]).reshape(
            keep_faces.shape
        )
        disconn_meshes.append((new_vertices, new_faces))

    return disconn_meshes


def cell_generator(
    directory_name: str, segment: bool
) -> Iterator[Tuple[str, VertexArray, FaceArray]]:
    r"""
:param directory_name: The directory where the *.obj files are stored
:param segment: if segment is True, each cell will be segmented into its \
    set of connected components before being returned. If segment \
    is False, the contents of the \*.obj file will be returned as-is.
:return: An iterator over all cells in the directory, where a "cell"\
    is a triple (cell_name, vertices, faces).
    """

    file_names = [
        file_name
        for file_name in os.listdir(directory_name)
        if os.path.splitext(file_name)[1] in [".obj", ".OBJ"]
    ]
    if segment:
        for file_name in file_names:
            vertices, faces = read_obj(os.path.join(directory_name, file_name))
            mesh_list = disconnect_mesh(vertices, faces)
            i = 0
            for mesh in mesh_list:
                yield (os.path.splitext(file_name)[0] + "_" + str(i), mesh[0], mesh[1])
                i = i + 1
    else:
        for file_name in file_names:
            vertices, faces = read_obj(os.path.join(directory_name, file_name))
            yield (os.path.splitext(file_name)[0], vertices, faces)


def sample_vertices(vertices: VertexArray, n_sample: int) -> Optional[VertexArray]:
    """
Evenly samples n vertices. Most .obj vertices are ordered \
counter-clockwise, so evenly sampling from vertex matrix \
can roughly approximate even sampling across the mesh

:param vertices: 3D coordinates for vertices
:param n_sample: number of vertices to sample
:return: None, if there are fewer vertices than points to sample. \
    Otherwise, a numpy array of sampled vertices, of shape (n_sample, 3).
    """

    if vertices.shape[0] < n_sample:
        warnings.warn("Fewer vertices than points to sample, skipping")
        return None
    return vertices[np.linspace(0, vertices.shape[0] - 1, n_sample).astype("uint32"), :]


def get_geodesic_heat_one_mesh(
    vertices: VertexArray, faces: FaceArray, n_sample: int
) -> Optional[npt.NDArray[np.float64]]:
    r"""
Given a mesh, randomly sample n_sample points from the mesh, \
compute the pairwise geodesic distances between the sampled points using the heat method, \
and return the square matrix of pairwise geodesic distances, linearized into a vector, \
or "None" if there are fewer than n_sample vertices in the mesh. \
For more on the heat method, see: \

https://github.com/nmwsharp/potpourri3d/blob/master/README.md#mesh-distance

https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/

:param vertices: 3D coordinates for vertices
:param faces: row of vertices contained in each face
:param n_sample: number of vertices to sample

:return: heat geodesic distance in vector form, of shape \
    (n_sample \* (n_sample - 1)/2, 1)
    """
    if vertices.shape[0] < n_sample:
        warnings.warn("Fewer vertices than points to sample, skipping")
        return None
    even_sample = np.linspace(0, vertices.shape[0] - 1, n_sample).astype("uint32")
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    dist_mat = solver.compute_distance(even_sample[0])[even_sample]
    for i in range(1, len(even_sample)):
        dist_mat = np.c_[dist_mat, solver.compute_distance(even_sample[i])[even_sample]]
    dist_mat = np.maximum(dist_mat, dist_mat.T)  # symmetrize
    dist_vec = squareform(dist_mat)
    return dist_vec


def get_geodesic_networkx_one_mesh(
    vertices: VertexArray, faces: FaceArray, n_sample: int
) -> Optional[npt.NDArray[np.float64]]:
    """
Given a mesh, randomly sample n_sample points from the \
mesh, computes the pairwise geodesic distances between the sampled points \
along the (distance-weighted) underlying graph of the mesh, and returns the \
square matrix of pairwise geodesic distances, linearized into a vector, or \
"None" if there are fewer than n_sample vertices in the mesh.

:param vertices: 3D coordinates for vertices
:param faces: row of vertices contained in each face
:param n_sample: number of vertices to sample
:return: graph geodesic distance in vector form
    """
    if vertices.shape[0] < n_sample:
        warnings.warn("Fewer vertices than points to sample, skipping")
        return None
    graph = nx.Graph()
    for i in range(vertices.shape[0]):
        graph.add_node(i)
    for i in range(faces.shape[0]):
        for edge_id in [(0, 1), (1, 2), (2, 0)]:
            v1 = faces[i, edge_id[0]]
            v2 = faces[i, edge_id[1]]
            graph.add_edge(v1, v2, weight=euclidean(vertices[v1], vertices[v2]))
    even_sample = np.linspace(0, vertices.shape[0] - 1, n_sample).astype("uint32")
    arguments = list(it.combinations(even_sample, 2))
    dist_vec = np.zeros(len(arguments))
    for i in range(len(arguments)):
        pair = arguments[i]
        dist_vec[i] = nxsp.generic.shortest_path_length(
            graph, source=pair[0], target=pair[1], weight="weight"
        )
    return dist_vec


def get_geodesic(
    vertices: VertexArray, faces: FaceArray, n_sample: int, method: str
) -> Optional[npt.NDArray[np.float64]]:
    """
    Sample `n_sample` many points and compute an intracell distance matrix of pairwise \
    geodesic distances between points.
    """

    if method == "networkx":
        return get_geodesic_networkx_one_mesh(vertices, faces, n_sample)
    elif method == "heat":
        return get_geodesic_heat_one_mesh(vertices, faces, n_sample)

    raise Exception("Invalid method, must be one of 'networkx' or 'heat'")


def _connect_helper(
    t: Tuple[str, VertexArray, FaceArray]
) -> Tuple[str, VertexArray, FaceArray]:
    name, vertices, faces = t
    return (t[0], t[1], connect_mesh(t[1], t[2]))


def compute_intracell_all(
    infolder: str,
    n_sample: int,
    metric: Literal["euclidean"] | Literal["geodesic"],
    pool: ProcessPool,
    segment: bool = True,
    method: Literal["networkx"] | Literal["heat"] = "networkx",
) -> Iterator[Tuple[str, Optional[npt.NDArray[np.float64]]]]:
    if metric == "geodesic" and not segment:
        cell_gen = pool.imap(
            _connect_helper, cell_generator(infolder, segment), chunksize=1
        )
    else:
        cell_gen = cell_generator(infolder, segment)

    if metric == "geodesic":
        chunksize = 1 if method == "networkx" else 20
        compute_geodesic: Callable[
            [Tuple[str, VertexArray, FaceArray]],
            Tuple[str, Optional[npt.NDArray[np.float64]]],
        ]

        def compute_geodesic(
            t: tuple[str, VertexArray, FaceArray]
        ) -> tuple[str, Optional[npt.NDArray[np.float64]]]:
            return t[0], get_geodesic(t[1], t[2], n_sample, method)

        # compute_geodesic = lambda t: (t[0], get_geodesic(t[1], t[2], n_sample, method))
        return pool.imap(compute_geodesic, cell_gen, chunksize=chunksize)

    # metric is not "geodesic".
    if metric == "euclidean":
        pt_clouds = pool.imap(
            lambda t: (t[0], sample_vertices(t[1], n_sample)), cell_gen, chunksize=1000
        )
        return pool.imap(
            lambda t: (t[0], None if t[1] is None else pdist(t[1])),
            pt_clouds,
            chunksize=1000,
        )
    raise Exception("Metric should be either 'geodesic' or 'euclidean'")


def compute_icdm_all(
    infolder: str,
    out_csv: str,
    metric: Literal["euclidean"] | Literal["geodesic"],
    n_sample: int = 50,
    num_processes: int = 8,
    segment: bool = True,
    method: Literal["networkx"] | Literal["heat"] = "heat",
) -> List[str]:
    r"""
    Go through every Wavefront \*.obj file in the given input directory `infolder`
    and compute intracell distances according to the given metric. Write the results
    to output \*.csv file named `out_csv`.

    :param infolder: Folder full of \*.obj files.
    :param out_csv: Output will be written to a \*.csv file
        titled `out_csv`.
    :param metric: How to compute the distance between points.
    :param n_sample: How many points to sample from each cell.
    :param num_processes: Number of independent processes which will be created.
        Recommended to set this equal to the number of cores on your machine.
    :param method: How to compute geodesic distance.
        The "networkx" method is more precise, and takes between 5 - 15 seconds for
        a cell with 50 sample points. The "heat" method is a faster but rougher
        approximation, and takes between 0.05 - 0.15 seconds for a cell with
        50 sample points. This flag is not relevant if the user is sampling
        Euclidean distances.
    :param segment: If `segment` is True, each \*.obj file will be segmented into its
        set of connected components before being returned, so an \*.obj file with multiple
        connected components will be understood to contain multiple distinct cells.
        If `segment` is False, each \*.obj file will be understood to contain a single
        cell, and points will be sampled accordingly. If `segment` is False and the user
        chooses "geodesic", in the event that an \*.obj file contains multiple connected components,
        the function will attempt to "repair" the \*.obj file by adjoining new faces to the complex
        so that a sensible notion of geodesic distance can be computed between two points. The user
        is warned that this imputing of data carries the same consequences with regard
        to scientific interpretation of the results as any other kind of data imputation
        for incomplete data sets.
    :return: Names of cells for which sampling failed because the cells have
        fewer than `n_sample` points.
    """

    pool = ProcessPool(nodes=num_processes)
    dist_mats = compute_intracell_all(infolder, n_sample, metric, pool, segment, method)
    batch_size = 1000
    failed_cells = write_csv_block(out_csv, n_sample, dist_mats, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells
