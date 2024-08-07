"""
Functionality to process cell images with polygonal boundaries and
multiple intensity channels (corresponding to different proteins)
"""

import warnings
import itertools as it
from typing import Any, Optional, Literal
from multiprocessing import Pool, cpu_count
from math import ceil

import numpy as np
import numpy.typing as npt
import skimage.transform
import networkx as nx
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.neighbors import kneighbors_graph
from matplotlib.path import Path
import ot
from tqdm import tqdm


def image_coords(shape: tuple[int, int]):
    """
    Given a shape tuple (n,m), returns a numpy array of shape (n * m, 2)
    which enumerates the x and y indices of the array in row-major order.
    """
    n, m = shape
    xs = np.repeat(np.arange(stop=n, step=1, dtype=int), m)
    ys = np.tile(np.arange(stop=m, step=1, dtype=int), reps=(n,))
    return np.stack((xs, ys), axis=1)


def polygon_to_bitmap(
    vertex_coords: npt.NDArray, shape: tuple[int, int]
) -> npt.NDArray[np.bool]:
    """
    Given an array `vertex_coords`, return a boolean mask of shape
    `shape` where points lying inside the polygon are `true` and
    points outside the polygon are `false`.

    :param vertex_coords: An array of shape (z,2), where `z` is the
        number of vertices in the polygon, and the i-th row of the array
        is a pair of points (x_i, y_i). The x and y values can be
        integers or floats; their value should lie within the
        rectangle bounded by (0,0) and `shape`.
    :param shape: The desired shape of the output boolean bitmap.
    """

    if len(vertex_coords.shape) != 2 or vertex_coords.shape[1] != 2:
        raise ValueError("Vertex_coords should be of shape (z,2)")

    return Path(vertex_coords).contains_points(image_coords(shape)).reshape(shape)


def polygon_to_points(
    vertex_coords: npt.NDArray, shape: tuple[int, int]
) -> npt.NDArray[int]:
    """
    Given an array `vertex_coords`, return an array of shape (k, 2),
    where k is the number of pixels that are contained in the polygon
    vertex_coords, and the i-th row is the pair (x_i, y_i) of integer
    coordinates of the pixel.

    :param vertex_coords: An array of shape (z,2), where `z` is the
        number of vertices in the polygon, and the i-th row of the array
        is a pair of points (x_i, y_i). The x and y values can be
        integers or floats; their value should lie within the
        rectangle bounded by (0,0) and `shape`.
    :param shape: The shape of a bounding box for the polygon, i.e.,
        the dimensions of the image that the cell is drawn from.
    """
    if len(vertex_coords.shape) != 2 or vertex_coords.shape[1] != 2:
        raise ValueError("Vertex_coords should be of shape (z,2)")
    im_coords = image_coords(shape)
    bitmap = Path(vertex_coords).contains_points(im_coords)
    return im_coords[bitmap, :]


def compute_geodesic_dmat(
    coords: npt.NDArray[int], n_neigh=4
) -> npt.NDArray[np.float64]:
    """
    Compute a nearest-neighbors graph through an array of integer pixel
    coordinates, and return the matrix of pairwise distances through
    the body of the graph.

    :param coords: Of shape (n, 2), where n is the number of pixels in
        the region, and the i-th row is the pair (x_i, y_i) of integer
        coordinates of the i-th pixel.
    :param n_neigh: How many nearest neighbors to consider when
        computing the nearest neighbors graph.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError("coords should be of shape (z,2)")

    knn = kneighbors_graph(coords, n_neigh, mode="connectivity", include_self=False)
    # compute pairwise geodesic distances
    graph = nx.from_numpy_array(knn.toarray())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        p = dict(nx.shortest_path(graph))
    # construct distance matrix
    gastr_dmat = np.zeros((coords.shape[0], coords.shape[0]))
    index_pairs = it.combinations(iter(range(coords.shape[0])), 2)
    for i, j in index_pairs:
        gastr_dmat[i, j] = len(p[i][j]) - 1
        gastr_dmat[j, i] = len(p[i][j]) - 1
    return gastr_dmat


class CellImage:
    """
    A CellImage consists of the following data:

    - one or more two-dimensional raster images of a
      cell, each of the same dimensions, but expressing different
      intensities corresponding to different image channels or
      different protein levels.

    - a polygonal boundary for the cell, expressed as a list of
      ordered pairs (the vertices for the polygon). The boundary
      should separate the cell from the background and other cells.
      CAJAL does not include functionality to segment cells, you will
      need image segmentation software.
    """

    def __init__(
        self,
        image_intensity_levels: npt.NDArray[np.int64],
        polygonal_boundary: npt.NDArray,
        downsample: int = 1,
        rescale_threshold: float = 0.99,
        distance_metric: Literal["euclidean"] | Literal["geodesic"] = "geodesic",
        n_neigh: int = 4,
    ):
        """
        N. b. At the moment, we instantiate the cell with the uniform
        probability distribution on points, as the number of points that
        will lie in the masked region is not known ahead of time, so it is
        not sensible to let the user supply a probability distribution in
        the form of an array. If more flexibility is desired, contact the
        authors.

        :param image_intensity_levels: A floating-point array of shape (k,
            n, m), where k is the number of image channels for the cell, n
            is the number of pixel rows in the cell, and m is the number
            of pixel columns in the cell. Code is tested with integer
            pixel intensities.
        :param polygonal_boundary: A floating-point array of shape (z,2),
            where z is the number of vertices in the polygonal boundary,
            and the i-th row is a pair (x_i,y_i) of coordinates of the
            i-th vertex. The values can be integers or floats but they
            should lie in the box bounded by (0,0) and
            image_intensity_levels.shape.
        :param downsample: Using the resize function from scikit-image, we
            rescale the picture by a factor of 1/downsample, lowering the
            resolution of the cell and increasing the speed of the fused
            Gromov-Wasserstein computation (at the cost of degraded
            accuracy). You should experiment with this parameter to find
            an acceptable tradeoff between time cost and accuracy.
        :param rescale_threshold: Image artifacts of an extremely high
            intensity may distort the results, so it may be desirable to
            set an upper threshold for the pixel intensity and cap pixel
            intensities at that threshold. This parameter sets
            caps at the `rescale_threshold` quantile of the distribution
            of pixel intensities.
        :param distance_metric: How to reduce the cell image to a distance
            matrix. If the cell is pill-shaped, then the two distance
            matrices are likely to be substantially identical. If the cell
            has significant protuberances, or is a neuron, then they will
            differ substantially.
        :param n_neigh: (for geodesic distance) How many nearest neighbors
            to consider when computing the nearest neighbors graph through
            the interior of the cell. As n grows large, the matrix
            converges to the Euclidean distance matrix.
        :param n_neigh: (for geodesic distance) How many nearest neighbors
            to consider when computing the nearest neighbors graph through
            the interior of the cell. As n grows large, the matrix
            converges to the Euclidean distance matrix.
        """

        if distance_metric not in ("euclidean", "geodesic"):
            raise ValueError(
                "`distance_metric` should be 'euclidean' or \
            'geodesic'."
            )
        if len(image_intensity_levels.shape) == 2:
            warnings.warn(
                "image_intensity_levels should be a \
                three-dimensional array representing a stack of images. \
                Assuming that the array represents a single image."
            )
            image_intensity_levels = np.array(
                image_intensity_levels, copy=False, ndmin=3
            )
        elif len(image_intensity_levels.shape) != 3:
            raise ValueError("`image_intensity_levels` should be three-dimensional.")

        # original_shape = image_intensity_levels.shape
        # resized_shape = (
        #     original_shape[0],
        #     original_shape[1] // downsample,
        #     original_shape[2] // downsample,
        # )
        # Resize the image.
        image_intensity_levels = skimage.transform.rescale(
            image_intensity_levels, 1 / downsample, channel_axis=0, anti_aliasing=True
        )
        # Cap all pixel intensities at rescale_threshold.
        image_intensity_levels = np.maximum(
            image_intensity_levels,
            np.quantile(image_intensity_levels, rescale_threshold, axis=(1, 2))[
                :, np.newaxis, np.newaxis
            ],
        )

        polygonal_boundary /= downsample
        x_min, y_min = np.min(polygonal_boundary, axis=0)
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = np.max(polygonal_boundary, axis=0)
        x_max, y_max = int(ceil(x_max)), int(ceil(y_max))
        image_intensity_levels = image_intensity_levels[
            :, x_min : (x_max + 1), y_min : (y_max + 1)
        ]
        polygonal_boundary -= np.array((x_min, y_min))[np.newaxis, :]

        nchannels, nrow, ncol = image_intensity_levels.shape
        pixel_indices: npt.NDArray[int] = polygon_to_points(
            polygonal_boundary, (nrow, ncol)
        )
        x_min, y_min = np.min(pixel_indices, axis=0)
        x_max, y_max = np.max(pixel_indices, axis=0)

        pixel_indices[:, 0] -= x_min
        pixel_indices[:, 1] -= y_min

        if distance_metric == "euclidean":
            distance_matrix = squareform(pdist(pixel_indices) * downsample)
        else:
            assert distance_metric == "geodesic"
            distance_matrix = compute_geodesic_dmat(pixel_indices, n_neigh) * downsample
        self.image_intensities = image_intensity_levels
        self.distance_matrix = distance_matrix
        self.pixel_indices = pixel_indices
        n = distance_matrix.shape[0]
        self.distribution = np.ones((n,)) / n

    def feature_matrix(self, channels: tuple[int, ...]):
        return self.image_intensities[
            :, self.pixel_indices[:, 0], self.pixel_indices[:, 1]
        ][np.array(channels), :].transpose()


def normalize_across_cells(cells: list[CellImage]):
    """
    Given a list of CellImages, each with the same channels in the
    same order, normalize each channel across _all_ cell images -
    that is, compute the global standard deviation of the
    pixel intensities for the given channel across all cells, and then
    update the cells in-place so that the average pixel intensity for
    a given channel across all cells has standard deviation 1. (The
    average pixel intensity is not relevant to the Fused GW distance.)
    """

    all_intensities = np.concatenate(
        [
            cell.image_intensities.reshape((cell.image_intensities.shape[0], -1))
            for cell in cells
        ],
        axis=1,
    )
    assert len(all_intensities.shape) == 2
    global_stdev = np.std(all_intensities, axis=1)
    assert global_stdev.shape[0] == cells[0].image_intensities.shape[0]

    for cell in cells:
        cell.image_intensities /= global_stdev[:, np.newaxis, np.newaxis]


def fused_gromov_wasserstein(
    cell1: CellImage,
    cell2: CellImage,
    channels: tuple[int, ...] | None = None,
    **kwargs,
):
    """
    Compute the Fused Gromov-Wasserstein distance between two
    cells equipped with image channel data. This wraps the Python
    Optimal Transport library implementation of the Fused GW algorithm
    [here](https://pythonot.github.io/all.html#ot.fused_gromov_wasserstein),
    which should be consulted for documentation of relevant keyword
    arguments. The return type of the function varies depending on the
    boolean keyword argument `log`.

    For details on the Fused Gromov-Wasserstein distance, see the
    original paper: Vayer Titouan, Chapel Laetitia, Flamary Rémi,
    Tavenard Romain and Courty Nicolas, “Optimal Transport for
    structured data with application on graphs”, International
    Conference on Machine Learning (ICML). 2019.

    :param channels: A tuple of integers indicating the image
        channels we are interested in computing the Fused GW distance.
        If `channels` is `None`, then all channels are taken into
        consideration. The selected channels are concatenated to give a
        feature vector of pixel intensities in Euclidean space,
        and the distance between feature vectors determines the
        transport cost between those two pixels.
    """

    if isinstance(channels, int):
        warnings.warn(
            "`channels` should be a tuple of ints; \
        interpreting as a tuple with one element."
        )
        channels = (channels,)
    if channels is None:
        channels = list(range(cell1.image_intensities.shape[0]))

    linear_cost_matrix = cdist(
        cell1.feature_matrix(channels), cell2.feature_matrix(channels)
    )

    return ot.fused_gromov_wasserstein(
        linear_cost_matrix,
        cell1.distance_matrix,
        cell2.distance_matrix,
        p=cell1.distribution,
        q=cell2.distribution,
        **kwargs,
    )


def _init_fgw_pool(
    cells: list[CellImage], channels: Optional[tuple[int]], kwargs: dict[str, Any]
):
    """
    Set a global variable _CELLS so that the parallel process pool can
    access it.
    """
    global _CELLS
    _CELLS = cells
    global _CHANNELS
    _CHANNELS = channels
    global _KWARGS
    _KWARGS = kwargs


def _fgw_index(p: tuple[int, int]):
    """
    Compute the Fused GW distance between cells i and j
    in the master cell list.
    """
    i, j = p
    _, log = fused_gromov_wasserstein(_CELLS[i], _CELLS[j], _CHANNELS, **_KWARGS)
    return (i, j, log["fgw_dist"])


def fused_gromov_wasserstein_parallel(
    cells: list[CellImage],
    channels: Optional[tuple[int]] = None,
    num_processes: Optional[int] = None,
    chunksize=1,
    **kwargs,
) -> npt.NDArray[np.float64]:
    """
    Compute the fused Gromov-Wasserstein distance between all pairs of
    cells in the given list, incorporating information from the
    channels specified in `channels`.

    :param cells: The cells to compute the pairwise FGW distance.
    :param channels: The channels that will be incorporated into the
        FGW distance computation.
    :param num_processes: How many Python processes should be launched
        in parallel to compute the GW distance. Defaults to
        `multiprocessing.cpu_count()`.
    :return: A symmetric distance matrix of pairwise GW distances
    """
    num_cells = len(cells)
    if num_processes is None:
        num_processes = cpu_count()
    # compute pairwise fGW distances between all objects
    index_pairs = it.combinations(
        iter(range(num_cells)), 2
    )  # object pairs to compute fGW / OT for
    total_num_pairs = int(
        (num_cells * (num_cells - 1)) / 2
    )  # total number of object pairs to compute (for progress bar)
    kwargs["log"] = True

    with Pool(
        initializer=_init_fgw_pool,
        initargs=(cells, channels, kwargs),
        processes=num_processes,
    ) as pool:
        res = pool.imap_unordered(_fgw_index, index_pairs, chunksize=chunksize)
        # store GW distances
        fgw_dmat = np.zeros((num_cells, num_cells))
        for i, j, fgw_dist in tqdm(res, total=total_num_pairs, position=0, leave=True):
            fgw_dmat[i, j] = fgw_dist
            fgw_dmat[j, i] = fgw_dist
    return fgw_dmat
