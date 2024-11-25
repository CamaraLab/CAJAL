# Functions for sampling points from a 2D segmented image
import os
import warnings
from typing import List, Iterator, Tuple, Literal, Optional, Any
import itertools as it
from math import ceil

import numpy as np
import ot
import numpy.typing as npt
from skimage import measure
from sklearn.neighbors import kneighbors_graph
import tifffile
from scipy.spatial.distance import pdist, cdist, squareform
from pathos.pools import ProcessPool
from matplotlib.path import Path
import multiprocessing
import networkx as nx
import scipy.sparse
from tqdm import tqdm
import skimage

from .utilities import write_csv_block


def _filter_to_cells(segmask: npt.NDArray[np.int_], background: int) -> list[int]:
    """
    Return a list of identifiers for cells in the interior of the image.
    """
    cell_ids = set(np.unique(segmask))
    remove_cells = set()
    remove_cells.add(background)
    remove_cells.update(np.unique(segmask[0, :]))
    remove_cells.update(np.unique(segmask[-1, :]))
    remove_cells.update(np.unique(segmask[:, 0]))
    remove_cells.update(np.unique(segmask[:, -1]))
    return list(cell_ids.difference(remove_cells))


def cell_boundaries(
    imarray: npt.NDArray[np.int_],
    n_sample: int,
    background: int = 0,
    discard_cells_with_holes: bool = False,
    only_longest: bool = False,
) -> List[Tuple[int, npt.NDArray[np.float64]]]:
    """
    Sample n coordinates from the boundary of each cell in a segmented image,
    skipping cells that touch the border of the image

    :param imarray: 2D segmented image where the pixels belonging to\
          different cells have different values
    :param n_sample: number of pixel coordinates to sample from boundary of each cell
    :param background: value of background pixels, this will not be saved as a boundary
    :param discard_cells_with_holes: \
          if discard_cells_with_holes is true, we discard any cells \
          with more than one boundary (e.g., an annulus) with a \
          warning. Else, the behavior is determined by only_longest.
    :param only_longest: if discard_cells_with_holes is true, \
          only_longest is irrelevant. Otherwise, this determines whether \
          we sample points from only the longest boundary (presumably \
          the exterior) or from all boundaries, exterior and interior.
    :return:
       list of float numpy arrays of shape (n_sample, 2) \
       containing points sampled from the contours.
    """

    cell_id_list = _filter_to_cells(imarray, background)
    outlist: List[Tuple[int, npt.NDArray[np.float64]]] = []
    for cell in cell_id_list:
        cell_imarray = (imarray == cell) * 1
        boundary_pts_list = measure.find_contours(
            cell_imarray, 0.5, fully_connected="high"
        )
        if discard_cells_with_holes and len(boundary_pts_list) > 1:
            warnings.warn("More than one boundary for cell " + str(cell))
            continue
        boundary_pts: npt.NDArray[np.float64]
        if only_longest:
            boundary_pts_list.sort(key=lambda ell: ell.shape[0])
            boundary_pts = boundary_pts_list[0]
        else:
            boundary_pts = np.concatenate(boundary_pts_list)
        if boundary_pts.shape[0] < n_sample:
            warnings.warn(
                "Fewer than "
                + str(n_sample)
                + " pixels around boundary of cell "
                + str(cell)
            )
        indices = np.linspace(0, boundary_pts.shape[0] - 1, n_sample)
        outlist.append((cell, boundary_pts[indices.astype("uint32")]))
    return list(outlist)


def _compute_intracell_all(
    infolder: str,
    n_sample: int,
    pool: ProcessPool,
    background: int,
    discard_cells_with_holes: bool,
    only_longest: bool,
) -> Iterator[Tuple[str, npt.NDArray[np.float64]]]:
    file_names = [
        file_name
        for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] in [".tif", ".tiff", ".TIF", ".TIFF"]
    ]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]

    # compute_cell_boundaries: Callable[[str], List[Tuple[int, npt.NDArray[np.float64]]]]
    def compute_cell_boundaries(file_name: str):
        return cell_boundaries(
            tifffile.imread(os.path.join(infolder, file_name)),  # type: ignore
            n_sample,
            background,
            discard_cells_with_holes,
            only_longest,
        )

    cell_names_repeat: Iterator[Iterator[str]]
    cell_names_repeat = map(it.repeat, cell_names)
    cell_bdary_lists: Iterator[
        Tuple[Iterator[str], Iterator[Tuple[int, npt.NDArray[np.float64]]]]
    ]
    cell_bdary_lists = zip(
        cell_names_repeat, pool.imap(compute_cell_boundaries, file_names, chunksize=100)
    )
    cell_bdary_list_iters: Iterator[
        Iterator[Tuple[str, Tuple[int, npt.NDArray[np.float64]]]]
    ]
    cell_bdary_list_iters = map(lambda tup: zip(tup[0], tup[1]), cell_bdary_lists)
    cell_bdary_list_flattened: Iterator[Tuple[str, Tuple[int, npt.NDArray[np.float64]]]]
    cell_bdary_list_flattened = it.chain.from_iterable(cell_bdary_list_iters)

    def restructure_and_get_pdist(
        tup: tuple[str, tuple[int, npt.NDArray[np.float64]]]
    ) -> tuple[str, npt.NDArray[np.float64]]:
        name = tup[0] + "_" + str(tup[1][0])
        pd = pdist(tup[1][1])
        return name, pd

    return pool.imap(
        restructure_and_get_pdist, cell_bdary_list_flattened, chunksize=1000
    )


def compute_icdm_all(
    infolder: str,
    out_csv: str,
    n_sample: int,
    num_processes: int = 8,
    background: int = 0,
    discard_cells_with_holes: bool = False,
    only_longest: bool = False,
) -> None:
    """
    Read in each segmented image in a folder (assumed to be .tif), \
    save n pixel coordinates sampled from the boundary
    of each cell in the segmented image, \
    skipping cells that touch the border of the image.

    :param infolder: path to folder containing .tif files.
    :param out_csv: path to csv file to save cell boundaries.
    :param n_sample: number of pixel coordinates \
           to sample from boundary of each cell
    :param discard_cells_with_holes: \
        if discard_cells_with_holes is true, we discard any cells \
        with more than one boundary (e.g., an annulus) with a \
        warning. Else, the behavior is determined by only_longest.
    :param background: value which characterizes the color of the background pixels, \
         this will not be saved as a boundary
    :param only_longest: if discard_cells_with_holes is true, \
         only_longest is irrelevant. Otherwise, this determines whether \
         we sample points from only the longest boundary (presumably \
         the exterior) or from all boundaries, exterior and interior.

    :param num_processes: How many threads to run while sampling.
    :return: None (writes to file)
    """

    pool = ProcessPool(nodes=num_processes)
    name_dist_mat_pairs = _compute_intracell_all(
        infolder, n_sample, pool, background, discard_cells_with_holes, only_longest
    )
    batch_size: int = 1000
    write_csv_block(out_csv, n_sample, name_dist_mat_pairs, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return None


def image_coords(shape: tuple[int, int]) -> npt.NDArray[np.int64]:  # type: ignore
    """
    Given a shape tuple (n,m), returns a numpy array of shape (n * m, 2)
    which enumerates the x and y indices of the array in row-major order.
    """
    n, m = shape
    xs = np.repeat(np.arange(start=0, stop=n, step=1, dtype=np.int64), m)
    ys = np.tile(np.arange(start=0, stop=m, step=1, dtype=np.int64), reps=(n,))
    return np.stack((xs, ys), axis=1)


def polygon_to_bitmap(
    vertex_coords: npt.NDArray, shape: tuple[int, int]
) -> npt.NDArray[bool]:
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
) -> npt.NDArray[np.int64]:
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
    im_coords: npt.NDArray[np.int64] = image_coords(shape)
    bitmap = Path(vertex_coords).contains_points(im_coords)
    return im_coords[bitmap, :]


def compute_geodesic_dmat(
    coords: npt.NDArray[np.int64], n_neigh=4
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

    knn: scipy.sparse.csr_matrix = kneighbors_graph(
        coords, n_neigh, mode="connectivity", include_self=False
    )
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

    @staticmethod
    def _validate_metric(distance_metric):
        if distance_metric not in ("euclidean", "geodesic"):
            raise ValueError(
                "`distance_metric` should be 'euclidean' or \
            'geodesic'."
            )

    @staticmethod
    def _validate_channels(image_intensity_levels):
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
        return image_intensity_levels

    @staticmethod
    def _rescale_channels(
        distance_metric: Literal["euclidean"] | Literal["geodesic"],
        image_intensity_levels: npt.NDArray,
        downsample: int,
        intensity_threshold: float,
    ):
        """Validate inputs, downsample, and cap intensity thresholds"""

        CellImage._validate_metric(distance_metric)
        image_intensity_levels = CellImage._validate_channels(image_intensity_levels)
        s = image_intensity_levels.shape
        # Resize the image.
        if downsample > 1:
            image_intensity_levels = skimage.transform.resize(
                image_intensity_levels,
                (s[0], s[1] // downsample, s[2] // downsample),
                anti_aliasing=True,
            )
        # Cap all pixel intensities at intensity_threshold.
        if intensity_threshold < 1:
            image_intensity_levels = np.minimum(
                image_intensity_levels,
                np.quantile(image_intensity_levels, intensity_threshold, axis=(1, 2))[
                    :, np.newaxis, np.newaxis
                ],
            )
        return image_intensity_levels

    @staticmethod
    def _restrict_to_polygon(
        image_intensity_levels: npt.NDArray[np.uint8], polygonal_boundary: npt.NDArray
    ):
        """
        Given image intensity levels and an associated polygonal boundary,
        crop the image horizontally and vertically so that it is a tight bound
        for the polygonal boundary, and return the indices for the pixels
        internal to the polygon.
        """
        x_min, y_min = np.min(polygonal_boundary, axis=0)
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = np.max(polygonal_boundary, axis=0)
        x_max, y_max = int(ceil(x_max)), int(ceil(y_max))

        # image_intensity_levels = image_intensity_levels[
        #     :, x_min : (x_max + 1), y_min : (y_max + 1)
        # ]
        # polygonal_boundary -= np.array((x_min, y_min))[np.newaxis, :]

        _, nrow, ncol = image_intensity_levels.shape
        pixel_indices: npt.NDArray[np.int64] = polygon_to_points(
            polygonal_boundary, (nrow, ncol)
        )
        # I'm fairly confident that pixel_indices == mask_pts at this line.
        x_min, y_min = np.min(pixel_indices, axis=0)
        x_max, y_max = np.max(pixel_indices, axis=0)

        image_intensity_levels = image_intensity_levels[
            :, x_min : (x_max + 1), y_min : (y_max + 1)
        ]

        pixel_indices[:, 0] -= x_min
        pixel_indices[:, 1] -= y_min
        return (image_intensity_levels, pixel_indices)

    def __init__(
        self,
        image_intensity_levels: npt.NDArray[np.uint8],
        # polygonal_boundary: npt.NDArray,
        region: npt.NDArray,
        downsample: int = 1,
        intensity_threshold: float = 0.99,
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
            of pixel columns in the cell.
            (That is, the image intensities are stored in row-major order.)
            Code is tested with integer pixel intensities.
        :param region: region can be either a boolean mask of the same
            shape as the image (where 'true' indicates pixels in the cell)
            or a floating-point array of shape (z,2),
            coding a polygonal boundary for the cell,
            where z is the number of vertices in the polygonal boundary,
            and the i-th row is a pair (x_i,y_i) of coordinates of the
            i-th vertex (x_i the *column* index, position on the x-axis,
            and y_i the *row* index, position on the y-axis,
            in convention with the standard Cartesian coordinate system.
            Note that this is the opposite convention of
            image_intensity_levels.)
            The values in the polygonal boundary
            can be either integers or floats, but (y_i, x_i)
            should lie in the box bounded by (0,0) and
            image_intensity_levels.shape.
        :param downsample: Using the resize function from scikit-image, we
            rescale the picture by a factor of 1/downsample, lowering the
            resolution of the cell and increasing the speed of the fused
            Gromov-Wasserstein computation (at the cost of degraded
            accuracy). You should experiment with this parameter to find
            an acceptable tradeoff between time cost and accuracy.
        :param intensity_threshold: Image artifacts of an extremely high
            intensity may distort the results, so it may be desirable to
            set an upper threshold for the pixel intensity and cap pixel
            intensities at that threshold. This parameter sets
            caps at the `intensity_threshold` quantile of the distribution
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

        # Validate inputs, downsample, cap the image intensity levels.
        image_intensity_levels = CellImage._rescale_channels(
            distance_metric, image_intensity_levels, downsample, intensity_threshold
        )
        k, n, m = image_intensity_levels.shape
        if region.dtype == bool and region.shape == (n, m):
            segmentation_mask = region
            if downsample > 1:
                s = segmentation_mask.shape
                segmentation_mask = skimage.transform.resize(
                    segmentation_mask,
                    (s[0] // downsample, s[1] // downsample),
                    anti_aliasing=False,
                    order=0,
                )
            assert segmentation_mask.shape == image_intensity_levels.shape
            pixel_indices = np.argwhere(segmentation_mask)
        elif len(region.shape) == 2 and region.shape[1] == 2:
            polygonal_boundary = region[:, ::-1]
            # Crop the image and return the pixel indices for the interior of the image.
            image_intensity_levels, pixel_indices = CellImage._restrict_to_polygon(
                image_intensity_levels, polygonal_boundary / downsample
            )
        else:
            raise Exception(
                "region argument could not be interpreted as a bitmask or polygon."
            )

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

    @staticmethod
    def from_segmented_image(
        segmentation_mask: npt.NDArray,
        image_intensity_levels: npt.NDArray[np.uint8],
        background: int = 0,
        downsample: int = 1,
        intensity_threshold: float = 0.99,
        distance_metric: Literal["euclidean"] | Literal["geodesic"] = "geodesic",
        n_neigh: int = 4,
    ) -> List["CellImage"]:
        """
        :warning: The list of objects returned by this function all share
            (alias) the same field image_intensities.
            This cuts down on copying, but it may introduce spooky action at a distance.

        :param segmentation_mask: An array of shape (n, m), coding different cells in an image.
        :param image_intensity_levels: An array of image intensities of shape (k, n, m),
            where there are k different channels in the image, and each image is the same shape
            as the segmentation mask.
        :param background: The value in the segmentation mask associated with the background.
        :param downsample: Using the resize function from scikit-image, we
            rescale the picture by a factor of 1/downsample, lowering the
            resolution of the cell and increasing the speed of the fused
            Gromov-Wasserstein computation (at the cost of degraded
            accuracy). You should experiment with this parameter to find
            an acceptable tradeoff between time cost and accuracy.
        :param n_neigh: The geodesic distance through the interior of the cell is implemented
            in terms of the distance through a nearest-neighbor graph on the pixels, this
            argument determines how many nearest neighbors in the cell to consider when
            forming the nearest-neighbor graph.
        """

        # CellImage._validate_metric(distance_metric)
        # image_intensity_levels = CellImage._validate_channels(image_intensity_levels)
        image_intensity_levels = CellImage._rescale_channels(
            distance_metric, image_intensity_levels, downsample, intensity_threshold
        )
        # Note that anti_aliasing = False and order=0, as it is inappropriate here
        # to linearly interpolate between segmentation mask identifiers.
        s = segmentation_mask.shape
        segmentation_mask = skimage.transform.resize(
            segmentation_mask,
            (s[0] // downsample, s[1] // downsample),
            anti_aliasing=False,
            order=0,
        )
        assert segmentation_mask.shape == image_intensity_levels.shape
        cell_ids = _filter_to_cells(segmentation_mask, background)
        retv = [
            CellImage(
                image_intensity_levels,
                np.equal(id, segmentation_mask),
                downsample=1,
                intensity_threshold=1,
                distance_metric=distance_metric,
                n_neigh=n_neigh,
            )
            for id in cell_ids
        ]
        for i in range(len(retv) - 1):
            assert retv[i].image_intensities == retv[i + 1].image_intensities
        return retv


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
        channels = tuple(range(cell1.image_intensities.shape[0]))

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
    cells: list[CellImage], channels: Optional[tuple[int, ...]], kwargs: dict[str, Any]
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
    channels: Optional[tuple[int, ...]] = None,
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
        num_processes = multiprocessing.cpu_count()
    # compute pairwise fGW distances between all objects
    index_pairs = it.combinations(
        iter(range(num_cells)), 2
    )  # object pairs to compute fGW / OT for
    total_num_pairs = int(
        (num_cells * (num_cells - 1)) / 2
    )  # total number of object pairs to compute (for progress bar)
    kwargs["log"] = True

    with multiprocessing.Pool(
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
