# Functions for sampling points from a triangular mesh
from __future__ import annotations
import os
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
from typing import Tuple, List, Set, Dict, Optional, Iterator, Iterable, TypeAlias
from multiprocessing import Pool
from pathos.pools import ProcessPool
from CAJAL.lib.utilities import pj

# We represent a mesh as a pair (vertices, faces) : Tuple[VertexArray,FaceArray].
# A VertexArray is a numpy array of shape (n, 3), where n is the number of vertices in the mesh.
# Each row of a VertexArray is an XYZ coordinate triple for a point in the mesh.
VertexArray : TypeAlias = npt.NDArray[np.float_]
# A FaceArray is a numpy array of shape (m, 3) where m is the number of faces in the mesh.
# Each row of a FaceArray is a list of three natural numbers, corresponding to indices
# in the corresponding VertexArray,
# representing triangular faces joining those three points.
FaceArray : TypeAlias = npt.NDArray[np.int_]

def read_obj(file_path: str) -> Tuple[VertexArray,FaceArray]:
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
    return np.array(vertices), (np.array(faces)-1).astype("int64")

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
    connected_components : List[Set[int]] = [i[1] for i in enumerate(nx.connected_components(graph))]
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
    connections : Dict[Tuple[int,int],Tuple[int,int,int]]= {}


    # Add components as nodes in graph
    for i in range(len(connected_components)): spt_graph.add_node(i)
    # For every pair of components, find nearest points
    for i in range(len(connected_components)):
        for j in range(i+1, len(connected_components)):
            # Set bigger component as reference so we can make KDTree of that and loop over smaller
            ref, query = (i, j) if len(connected_components[i]) > len(connected_components[j]) \
                 else (j, i)
            ref_ids = list(connected_components[ref])
            query_ids = list(connected_components[query])
            # dist_ij is a rectangular matrix of floats.
            # dist_ij[i, j] is the distance between the i-th node of component ref_ids
            # and the j-th node of query_ids.
            dist_ij = cdist(vertices[ref_ids], vertices[query_ids])
            nearest : Tuple[int,int] = np.unravel_index(dist_ij.argmin(), dist_ij.shape) # type: ignore[assignment]
            next_nearest = np.argpartition(dist_ij[nearest[0]],1)[1]
            face : Tuple[int,int,int] = \
                (ref_ids[nearest[0]], query_ids[nearest[1]], query_ids[next_nearest])
            # Save face so I know to add it if this edge is in SPT
            connections[(i, j)] = face
            spt_graph.add_edge(i, j, weight=dist_ij[nearest])
    new_faces : List[Tuple[int,int,int]] = []
    for u, v, w in nx.minimum_spanning_edges(spt_graph):
        new_faces.append(connections[(min(u, v), max(u, v))])
    return np.vstack([faces, np.array(new_faces)])


def disconnect_mesh(vertices: VertexArray, faces: FaceArray) -> List[Tuple[VertexArray,FaceArray]]:
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
        convert_ids = dict(zip(v_ids, range(index, len(v_ids)+index)))
        new_faces = np.array(
               [convert_ids[x] for x in keep_faces.flatten()]
            ).reshape(keep_faces.shape)
        disconn_meshes.append((new_vertices, new_faces))

    return disconn_meshes

def cell_generator(
        directory_name : str,
        segment: bool
        ) -> Iterator[Tuple[str,VertexArray,FaceArray]]:
    """
    Arguments:
        * directory_name: The directory where the *.obj files are stored
        * segment: if segment is True, each cell will be segmented into its \
             set of connected components before being returned. If segment \
             is False, the contents of the \*.obj file will be returned as-is.
    
    Returns:
        * An iterator over all cells in the directory, where a "cell"\
            is a triple (cell_name, vertices, faces). 
    
    """

    file_names = [file_name for file_name in os.listdir(directory_name)
                  if os.path.splitext(file_name)[1] in [".obj", ".OBJ"]]
    if segment:
        for file_name in file_names:
            vertices, faces = read_obj(os.path.join(directory_name,file_name))
            mesh_list = disconnect_mesh(vertices, faces)
            i=0
            for mesh in mesh_list:
                yield (os.path.splitext(file_name)[0] + "_" + str(i),
                       mesh[0], mesh[1])
                i = i+1
    else:
        for file_name in file_names:
            vertices, faces = read_obj(os.path.join(directory_name,file_name))
            yield (os.path.splitext(file_name)[0], vertices, faces)

def sample_vertices(vertices: VertexArray, n_sample : int) -> Optional[VertexArray]:
    """
    Evenly samples n vertices. Most .obj vertices are ordered \
    counter-clockwise, so evenly sampling from vertex matrix \
    can roughly approximate even sampling across the mesh

    Args:
        * vertices (numpy array): 3D coordinates for vertices
        * n_sample (integer): number of vertices to sample

    Returns:
        None, if there are fewer vertices than points to sample. \
        Otherwise, a numpy array of sampled vertices, of shape (n_sample, 3).
    """
    
    if vertices.shape[0] < n_sample:
        warnings.warn("Fewer vertices than points to sample, skipping")
        return None
    return vertices[np.linspace(0, vertices.shape[0]-1, n_sample).astype("uint32"), :]

# def return_sampled_vertices(
#        vertices: VertexArray, faces : FaceArray,
#        n_sample: int, disconnect : bool =True) -> List[Optional[VertexArray]]:
#     """
#     Returns list of sampled vertices from each component of mesh (i.e. multiple cells in an .obj file)

#     Args:
#        * vertices (VertexArray): vertices of a mesh
#        * faces (FaceArray): faces of a mesh
#        * n_sample (integer): number of vertices to sample
#        * disconnect (boolean): If disconnect is True, the mesh will \
#             be decomposed into a list of connected meshes, \
#             and the function will return a list of VertexArrays \
#             of shape (n_sample,3), one for each component of the \
#             mesh, or "None" if the component has fewer than \
#             n_sample vertices.\
#             If disconnect is False, we sample n_sample vertices \
#             throughout the mesh, without regard to \
#             whether it is connected, and return a list of VertexArrays of length 1.
            
#     Returns:
#         list of VertexArrays of sampled vertices of size n_sample. \
#         A list entry will be "None" if there are fewer \
#          vertices than n_sample in that component. \
         
#     """
    
#     if not disconnect:
#         new_vertices = sample_vertices(vertices, n_sample)
#         return [new_vertices]
#     else:
#         disconn_meshes = disconnect_mesh(vertices, faces)
#         sample_list = []
#         for mesh in disconn_meshes:
#             new_vertices = sample_vertices(mesh[0], n_sample)
#             sample_list.append(new_vertices)
#         return sample_list


# def sample_vertices_and_save(vertices : VertexArray, faces: FaceArray,
#                          n_sample: int, outfile: str,
#                          disconnect: bool =True) -> None:
#     """
#     Evenly sample n vertices from a mesh and write the samples to csv. \
#       The output file(s) will have n_sample rows,\
#       where each row contains the xyz coordinates of a vertex.

#     Args:
#         * vertices : VertexArray, vertices of the mesh
#         * faces : FaceArray, faces of the mesh
#         * n_sample (integer): number of vertices to sample
#         * outfile (string): file path to write vertices. If disconnect is \
#             True and there are multiple disconnected components to the input \
#             mesh, multiple files are created with different index numbers. \
#             If outfile is given with string formatting {} characters, index \
#             is inserted there. Otherwise it is inserted before extension.
#         * disconnect (boolean): If disconnect is True, the mesh will \
#             be decomposed into a list of connected meshes, \
#             and the function will create one file for each connected \
#             component of the mesh with more than n_sample vertices.
#             If disconnect is false, one file is created.

#     Returns:
#         None (writes to file)
#     """
#     if not disconnect:
#         new_vertices = sample_vertices(vertices, n_sample)
#         if new_vertices is not None:
#             np.savetxt(outfile, new_vertices, delimiter=",", fmt="%.16f")
#     else:
#         disconn_meshes = disconnect_mesh(vertices, faces)
#         for i in range(len(disconn_meshes)):
#             mesh = disconn_meshes[i]
#             new_vertices = sample_vertices(mesh[0], n_sample)
#             if new_vertices is not None:
#                 if "{" in outfile:
#                     np.savetxt(outfile.format(i + 1), new_vertices, delimiter=",", fmt="%.16f")
#                 else:
#                     file_name_split = outfile.split(".")
#                     file_name = ".".join(file_name_split[:-1]) + "_" + str(i + 1)
#                     extension = file_name_split[-1]
#                     np.savetxt(file_name + "." + extension, new_vertices, delimiter=",", fmt="%.16f")

                    
# def _save_sample_from_obj(file_name, infolder, outfolder, n_sample, disconnect=True):
#     """
#     Read a mesh from a given \*.obj file, sample n_sample vertices from
#     each connected component, and write the results to \*.csv files.

#     This function is a wrapper for "sample_vertices_and_save" that
#     reads from an input file before sampling. Consult the documentation for that function.

#     Args:
#         * file_name (string): .obj file name
#         * infolder (string): folder containing .obj file
#         * outfolder (string): folder to save sampled vertices csv
#         * n_sample (integer): number of vertices to sample
#         * disconnect (boolean): Whether to sample vertices from whole mesh, \
#               or separate into connected components

#     Returns:
#         None (writes to file)
#     """
    
#     vertices, faces = read_obj(pj(infolder, file_name))
#     sample_vertices_and_save(vertices, faces, n_sample, pj(outfolder, file_name.replace(".obj", ".csv")), disconnect)


# def obj_sample_parallel(infolder: str, outfolder: str,
#                         n_sample: int, disconnect: bool =True,
#                         num_cores : int =8) -> None:
#     """
#     Read all \*.obj files in the given directory "infolder" into memory.
#     Decomposes each mesh into its connected components.
#     For each component, samples n_sample points from each component (in parallel),
#     and writes the resulting samples to a \*.csv file in directory "outfolder."
        
#     Args:
#         * infolder(string): path to directory containing .obj files
#         * outfolder (string): path to directory to write distance matrices
#         * n_sample (integer): number of vertices to sample from each mesh
#         * disconnect (boolean): Whether to sample vertices from whole mesh, or separate into connected components
#         * num_cores (integer): number of processes to use for parallelization
    
#     Returns:
#         None (writes files to outfolder)
#     """
    
#     if not os.path.exists(outfolder):
#         os.mkdir(outfolder)
#     arguments = [(file_name, infolder, outfolder, n_sample, disconnect)
#                  for file_name in os.listdir(infolder)]
#     with Pool(processes=num_cores) as pool:
#         pool.starmap(_save_sample_from_obj, arguments)


def get_geodesic_heat_one_mesh(vertices : VertexArray,
                               faces : FaceArray,
                               n_sample: int
                               ) -> Optional[npt.NDArray[np.float_]]:
    """
    Given a mesh, this function randomly samples n_sample points from the mesh,
    computes the pairwise geodesic distances between the sampled points using the heat method,
    and returns the square matrix of pairwise geodesic distances, linearized into a vector,
    or "None" if there are fewer than n_sample vertices in the mesh.    
    
    For more on the heat method, see:

    https://github.com/nmwsharp/potpourri3d/blob/master/README.md#mesh-distance

    https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/

    Args:
        * vertices (numpy array): 3D coordinates for vertices
        * faces (numpy array): row of vertices contained in each face
        * n_sample (integer): number of vertices to sample

    Returns:
        heat geodesic distance in vector form, of shape
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
        vertices : VertexArray,
        faces : FaceArray,
        n_sample : int) -> Optional[npt.NDArray[np.float_]]:
    """
    Given a mesh, this function randomly samples n_sample points from the
    mesh, computes the pairwise geodesic distances between the sampled points
    along the (distance-weighted) underlying graph of the mesh, and returns the
    square matrix of pairwise geodesic distances, linearized into a vector, or
    "None" if there are fewer than n_sample vertices in the mesh.

    Args:
        * vertices (numpy array): 3D coordinates for vertices
        * faces (numpy array): row of vertices contained in each face
        * n_sample (integer): number of vertices to sample

    Result:
        graph geodesic distance in vector form

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
            graph, source=pair[0], target=pair[1], weight="weight")
    return dist_vec

def get_geodesic(
        vertices : VertexArray,
        faces : FaceArray,
        n_sample : int,
        method : str) -> Optional[npt.NDArray[np.float_]]:

    match method:
        case "networkx":
            return get_geodesic_networkx_one_mesh(vertices, faces, n_sample)
        case "heat":
            return get_geodesic_heat_one_mesh(vertices, faces, n_sample)
        case _:
            raise Exception("Invalid method, must be one of 'networkx' or 'heat'")
    

# def return_geodesic(vertices : VertexArray,
#                     faces: FaceArray,
#                     n_sample : int,
#                     method: str = "networkx" ,
#                     connect: bool = False) -> List[Optional[npt.NDArray[np.float_]]]:
#     """
#     Returns a list of intracell geodesic distance matrices, one for each component of the given mesh
#     (i.e. multiple cells in an .obj file)

#     Args:
#         * vertices (numpy array): 3D coordinates for vertices
#         * faces (numpy array): row of vertices contained in each face
#         * n_sample (integer): number of vertices to sample
#         * method (string): one of 'networxk' or 'heat', how to compute geodesic distance \
#             networkx is slower but more exact for non-watertight \
#             methods, heat is a faster approximation.
#         * connect (boolean): If connect is True, then new faces will \
#             be added to the mesh until it is connected, and we \
#             compute a single geodesic intracell distance matrix for \
#             this connected mesh. In this case, the output list is of length one. \
#             If connect is False, then we compute a geodesic intracell \
#             distance matrix for each component of the mesh.

#     Result:
#         List of geodesic distance matrices, each linearized \
#         into vector form. If connect is True, this list \
#         has one element.

#     """
#     if connect:
#         new_faces = connect_mesh(vertices, faces)
#         if method == "heat":
#             dist_vec = get_geodesic_heat_one_mesh(vertices, new_faces, n_sample)
#         elif method == "networkx":
#             dist_vec = get_geodesic_networkx_one_mesh(vertices, new_faces, n_sample)
#         else:
#             raise Exception("Invalid method, must be one of 'networkx' or 'heat'")
#         return [dist_vec]
#     else:
#         disconn_meshes = disconnect_mesh(vertices, faces)
#         geo_list = []
#         for mesh in disconn_meshes:
#             if method == "heat":
#                 dist_vec = get_geodesic_heat_one_mesh(mesh[0], mesh[1], n_sample)
#             elif method == "networkx":
#                 dist_vec = get_geodesic_networkx_one_mesh(mesh[0], mesh[1], n_sample)
#             else:
#                 raise Exception("Invalid method, must be one of 'networkx' or 'heat'")
#             geo_list.append(dist_vec)
#         return geo_list


# def compute_and_save_geodesic(vertices: VertexArray,
#                               faces: FaceArray,
#                               n_sample: int,
#                               outfile: str,
#                               method:str ="networkx",
#                               connect: bool =False) -> None:
#     """
#     Compute and save the geodesic distance vector for each component of mesh \
#     (i.e. multple cells in an .obj file)

#     Args:
#         * vertices (numpy array): 3D coordinates for vertices
#         * faces (numpy array): row of vertices contained in each face
#         * n_sample (integer): number of vertices to sample
#         * outfile (string): file path to write to. If connect is \
#             False and there are multiple connected components \
#             to the input mesh, multiple files are created with \
#             different index numbers. If outfile is given with \
#             string formatting {} characters, index is inserted \
#             there. Otherwise it is inserted before extension.
#         * method (string): one of 'networxk' or 'heat', how to \
#           compute geodesic distance. 'networkx' is slower, but \
#           more exact for non-watertight methods, heat is a faster \
#           approximation.
#         * connect (boolean): If connect is True, then new faces will \
#             be added to the mesh until it is connected, and we \
#             compute a single geodesic intracell distance matrix for \
#             this connected mesh. In this case, only one file is generated.  \
#             If connect is False, then we compute a geodesic intracell \
#             distance matrix for each component of the mesh.

#     Result:
#         None (writes to file)
#     """
#     if connect:
#         new_faces = connect_mesh(vertices, faces)
#         if method == "heat":
#             dist_vec = get_geodesic_heat_one_mesh(vertices, new_faces, n_sample)
#         elif method == "networkx":
#             dist_vec = get_geodesic_networkx_one_mesh(vertices, new_faces, n_sample)
#         else:
#             raise Exception("Invalid method, must be one of 'networkx' or 'heat'")
#         if dist_vec is not None:
#             np.savetxt(outfile, dist_vec, fmt='%.8f')
#     else:
#         disconn_meshes = disconnect_mesh(vertices, faces)
#         for i in range(len(disconn_meshes)):
#             mesh = disconn_meshes[i]
#             if method == "heat":
#                 dist_vec = get_geodesic_heat_one_mesh(mesh[0], mesh[1], n_sample)
#             elif method == "networkx":
#                 dist_vec = get_geodesic_networkx_one_mesh(mesh[0], mesh[1], n_sample)
#             else:
#                 raise Exception("Invalid method, must be one of 'networkx' or 'heat'")
#             if dist_vec is not None:
#                 if "{" in outfile:
#                     np.savetxt(outfile.format(i + 1), dist_vec, fmt='%.8f')
#                 else:
#                     file_name_split = outfile.split(".")
#                     file_name = ".".join(file_name_split[:-1]) + "_" + str(i + 1)
#                     extension = file_name_split[-1]
#                     np.savetxt(file_name + "." + extension, dist_vec, fmt='%.8f')

# def save_geodesic_from_obj(file_name: str,
#                            infolder: str,
#                            outfolder: str,
#                            n_sample: int,
#                            method: str="networxk",
#                            connect: bool=False) -> None:
#     """
#     Computes geodesic distance matrices for mesh from an .obj file.

#     Args:
#         * file_name (string): name of single .obj file
#         * infolder (string): path to directory containing .obj files
#         * outfolder (string): path to directory to write distance matrices
#         * n_sample (integer): number of vertices to sample from each mesh
#         * method (string): one of 'networxk' or 'heat', how to compute geodesic distance\
#             networkx is slower but more exact for non-watertight methods, \
#             heat is a faster approximation
#         * connect (boolean): whether to check for disconnected meshes and connect them \
#             simply by adding faces

#     Returns:
#         None (writes files to outfolder)
#     """
#     vertices, faces = read_obj(pj(infolder, file_name))
#     if connect:
#         outfile = pj(outfolder, file_name.replace(".obj", "_dist.txt"))
#     else:
#         outfile = pj(outfolder, file_name.replace(".obj", "{}_dist.txt")) # string formatting for indices
#     compute_and_save_geodesic(vertices, faces, n_sample, outfile, method, connect)
   
    
# def compute_and_save_geodesic_from_obj_parallel(infolder: str,
#                                     outfolder: str,
#                                     n_sample: int,
#                                     method:str ="heat",
#                                     connect:bool =False,
#                                     num_cores: int =8)-> None:
#     """
#     Computes geodesic distance in parallel processes for all meshes in .obj files in a directory

#     Args:
#         * infolder(string): path to directory containing .obj files
#         * outfolder (string): path to directory to write distance matrices
#         * n_sample (integer): number of vertices to sample from each mesh
#         * method (string): one of 'networxk' or 'heat', how to compute geodesic distance.
#               The "networkx" method is more precise, and takes between 5 - 15 seconds for\
#               a cell with 50 sample points. The "heat" method is a faster but rougher \
#               approximation, and takes between 0.05 - 0.15 seconds for a cell with\
#               50 sample points.
#         * connect (boolean): whether to check for disconnected meshes and\
#               connect them simply by adding faces
#         * num_cores (integer): number of processes to use for parallelization

#     Returns:
#         None (writes files to outfolder)
#     """
#     if not os.path.exists(outfolder):
#         os.mkdir(outfolder)
#     arguments = [(file_name, infolder, outfolder, n_sample, method, connect)
#                  for file_name in os.listdir(infolder)]
#     with Pool(processes=num_cores) as pool:
#         pool.starmap(save_geodesic_from_obj, arguments)

def _connect_helper(t : Tuple[str,VertexArray,FaceArray]
                    ) -> Tuple[str,VertexArray,FaceArray]:

    name, vertices, faces = t
    return (t[0], t[1], connect_mesh(t[1],t[2]))

def compute_and_save_intracell_all(
        infolder: str,
        outfolder: str,
        n_sample : int,
        metric: str,
        segment : bool = True,
        method: str = "networkx",
        # force_connect: bool = "False"
        num_cores: int = 8) -> List[str]:

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    pool = ProcessPool(nodes=num_cores)
    dist_mats : Iterable[Tuple[str,Optional[npt.NDArray[np.float_]]]]

    if metric == "geodesic" and not segment:
        cell_gen = pool.imap(
            _connect_helper,
            cell_generator(infolder, segment),
            chunksize=1)
    else:
        cell_gen = cell_generator(infolder, segment)

    if metric == "geodesic":
        chunksize = 1 if method == "networkx" else 20
        dist_mats = \
              pool.imap( 
                lambda t : (t[0],get_geodesic(t[1],t[2],n_sample,method)),
                cell_gen,
                chunksize=chunksize)
    elif metric == "euclidean":
        pt_clouds = \
            pool.imap(
                lambda t : (t[0], sample_vertices(t[1],n_sample)),
                cell_gen,
                chunksize=1000)
        dist_mats =\
            pool.imap(
                lambda t : (t[0], None if t[1] is None else pdist(t[1])),
                pt_clouds,
                chunksize=1000)
    else:
        raise Exception("Metric should be either 'geodesic' or 'euclidean'")


    def _write_output(t : Tuple[str,Optional[npt.NDArray[np.float_]]]) -> Optional[str]:
        name, arr = t
        if arr is None:
            return name
        else:
            output_name = os.path.join(outfolder, name + ".txt")
            np.savetxt(output_name,arr,fmt='%.8f')
            return None
        
    failed_cell_names = pool.map(_write_output, dist_mats,chunksize=100)
    pool.close()
    pool.join()
    failed_cells = [name for name in failed_cell_names if name is not None]
    pool.clear()
    return failed_cells

    
