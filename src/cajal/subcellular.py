import numpy as np
import pandas as pd
import os
import skimage as ski
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import copy
from scipy.spatial.distance import cdist, pdist, squareform
from multiprocessing import Pool, cpu_count
import itertools as it
import ot
import warnings
import copy

from .sample_seg import cell_boundaries
from .gw_cython import gw_cython_core


def make_cell_image(gw_ot_cell, channels):
    """Create a multi-channel image array from a GW_OT_Cell object.

    This function generates a 3D image array where the first channel contains 
    the cell segmentation mask and subsequent channels contain normalized 
    intensity values for the specified channels.

    :param gw_ot_cell: Either a GW_OT_Cell object or a path to a pickled GW_OT_Cell object.
    :type gw_ot_cell: GW_OT_Cell or str
    :param channels: List of channel names to include in the image. Each channel should 
        correspond to a key in the GW_OT_Cell's intensities dictionary or 
        be 'nucleus' for nuclear segmentation.
    :type channels: list[str]
    :return: 3D array of shape (height, width, len(channels)+1) where the first 
        channel (index 0) contains the segmentation mask and subsequent 
        channels contain normalized intensity values (0-1 range).
    :rtype: numpy.ndarray
    """
    # Load GW_OT_Cell object if path specified
    if isinstance(gw_ot_cell, str):
        with open(gw_ot_cell, 'rb') as file:
            gw_ot_cell = pickle.load(file)
    coords = gw_ot_cell.coords
    coords[:,0] = coords[:,0] - coords[:,0].min()
    coords[:,1] = coords[:,1] - coords[:,1].min()
    # make new (n_channel + 1) x cell_width x cell_len image array 
    cell_image = np.zeros((coords[:,0].max()+1, coords[:,1].max()+1, len(channels)+1))
    for coord_i in range(len(coords)):
        i,j = coords[coord_i]
        cell_image[i,j,0] = 1 # store segmentation mask
        for channel_i in range(len(channels)): # store channel pixel intensities
            channel = channels[channel_i]
            if channel == 'nucleus':
                cell_image[i,j,channel_i+1] = gw_ot_cell.nucleus[coord_i] 
            else:
                cell_image[i,j,channel_i+1] = gw_ot_cell.intensities[channel][coord_i] 
    for channel_i in range(len(channels)):
        cell_image[:,:,channel_i+1] -= cell_image[:,:,channel_i+1].min()
        cell_image[:,:,channel_i+1] /= cell_image[:,:,channel_i+1].max()
    return(cell_image)


def to_shape(a, shape):
    """Pad a 3D array to match a target shape.

    Centers the input array within the target shape by adding symmetric 
    padding with zeros. This is useful for standardizing array dimensions 
    for visualization or analysis.

    :param a: numpy.ndarray
        Input 3D array to be padded.
    :param shape: tuple of int
        Target shape as (z, y, x) dimensions.
    :return: numpy.ndarray
        Padded array with the specified target shape, centered with 
        zero-padding.
    :rtype: numpy.ndarray
    """
    z_, y_, x_ = shape
    z, y, x = a.shape
    z_pad = (z_-z)
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((z_pad//2, z_pad//2 + z_pad%2),
                     (y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')


def make_cell_image_for_plot(image, mask_alpha=0.2):
    """Prepare a cell image for visualization by creating an RGB composite.

    Converts a multi-channel cell image into an RGB format suitable for 
    plotting. Adds a transparent cell mask overlay and reorders channels 
    to blue-red-green color scheme.

    :param image: Multi-channel cell image with shape (height, width, channels).
        Channel 0 should contain the cell segmentation mask.
    :type image: numpy.ndarray
    :param mask_alpha: Transparency level for the cell mask overlay, by default 0.2.
    :type mask_alpha: float
    :return: RGB image array with shape (height, width, 3) suitable for 
        visualization with channel ordering as blue, red, green.
    :rtype: numpy.ndarray
    """
    im = np.zeros((image.shape[0], image.shape[1], 3))
    mask = image[:,:,0].copy()
    for channel_i in range(1, image.shape[2]):
        # add transparent cell mask to channel
        im[:,:,channel_i-1] = image[:,:,channel_i] + (mask * mask_alpha)
        # rescale channel
        im[:,:,channel_i-1] = im[:,:,channel_i-1] / im[:,:,channel_i-1].max()
    # add mask to empty channels if any
    for channel_i in range(image.shape[2], 4):
        im[:,:,channel_i-1] = im[:,:,channel_i-1] + (mask * mask_alpha)
    # reorder channel color ordering to blue, red, green
    im = im[:,:,[1,2,0]]
    return(im)


def plot_cell_image(gw_ot_cell, channels, make_square=True, ax=None, mask_alpha=0.2):
    """Plot a cell image with the specified channels as an RGB composite.

    Creates a visualization of cell image data by combining multiple channels 
    into an RGB representation with an optional transparent mask overlay.

    :param image: 3D numpy array of shape (H, W, C) representing the multi-channel image.
    :type image: numpy.ndarray
    :param channels: List of channel names corresponding to the last dimension of the image.
    :type channels: list[str]
    :param cell_mask_image: 2D numpy array of shape (H, W) with integer labels for each cell (0 for background).
    :type cell_mask_image: numpy.ndarray
    :param nucleus_mask_image: 2D numpy array of shape (H, W) with integer labels for nuclei (0 for background). Default is None.
    :type nucleus_mask_image: numpy.ndarray or None
    :param ds_factor: Downsampling factor. If provided, downsample by this factor. Default is None.
    :type ds_factor: int or None
    :param ds_target_size: Target number of pixels per cell after downsampling. If provided, downsample to achieve this pixel count. Default is None.
    :type ds_target_size: int or None
    :param filter_border_cells: If True, exclude cells touching the image border. Default is True.
    :type filter_border_cells: bool
    :param n_boundary_points: Number of points to sample from the cell boundary. If None, boundary sampling is skipped. Default is 100.
    :type n_boundary_points: int or None
    :param save_path: Directory path to save the processed cell objects as pickle files. If None, objects are not saved. Default is None.
    :type save_path: str or None
    :param return_objects: If True, return the list of GW_OT_Cell objects. Default is True.
    :type return_objects: bool

    :return:
        If return_objects is True, returns a list of GW_OT_Cell objects; otherwise returns None.
    :rtype: list[GW_OT_Cell] or None
    """
    if len(channels) > 3:
        raise ValueError("Only up to 3 channels can be plotted.")
    image = make_cell_image(gw_ot_cell, channels)
    image = make_cell_image_for_plot(image, mask_alpha=mask_alpha)
    if make_square:
        max_dim = max(image.shape[0], image.shape[1])
        image = to_shape(image, (max_dim, max_dim, image.shape[2]))
    if ax:
        return(ax.imshow(image))
    else:
        plt.imshow(image)


def rescale_mask_to_pixel_count(mask, target_pixels, max_iter=20, tolerance=0.01):
    """Rescale a binary mask to achieve a target number of non-zero pixels.

    Iteratively adjusts the scale factor to resize a binary mask until the 
    number of non-zero pixels is close to the target value. Uses skimage's 
    resize function with nearest neighbor interpolation to maintain binary 
    nature of the mask.

    :param mask : numpy.ndarray
        2D binary numpy array containing 0s and 1s.
    :param target_pixels : int
        Desired number of non-zero pixels in the rescaled mask.
    :param max_iter : int, optional
        Maximum number of scaling iterations, by default 20.
    :param tolerance : float, optional
        Acceptable relative error as a fraction, by default 0.01 (1%).
    :return: numpy.ndarray
        Rescaled binary mask with pixel count close to target, maintaining 
        binary values (0s and 1s).
    :rtype: numpy.ndarray
    """
    current_pixels = np.count_nonzero(mask)
    
    # If mask is already close enough, return it
    if abs(current_pixels - target_pixels) / target_pixels < tolerance:
        return mask
    
    # Initial scale factor estimate (area scales with square of linear dimensions)
    scale_factor = np.sqrt(target_pixels / current_pixels)
    
    for _ in range(max_iter):
        # Calculate new dimensions (maintaining aspect ratio)
        h, w = mask.shape
        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))
        
        # Resize the mask using skimage.transform.resize
        resized_mask = ski.transform.resize(mask.astype(float), 
                                            (new_h, new_w),
                                            order=0,  # nearest neighbor interpolation
                                            preserve_range=True,
                                            anti_aliasing=False)
        
        # Binarize the result (threshold at 0.5 to maintain binary nature)
        resized_mask = (resized_mask > 0.5).astype(np.uint8)
        
        # Count current non-zero pixels
        current_pixels = np.count_nonzero(resized_mask)
        
        # Check if we're within tolerance
        if abs(current_pixels - target_pixels) / target_pixels < tolerance:
            return resized_mask
        
        # Update scale factor based on current error
        scale_factor *= np.sqrt(target_pixels / current_pixels)
    
    # Return the best result if max iterations reached
    return resized_mask


def compute_geodesic_dmat(mask_coords):
    """Compute geodesic distance matrix for given coordinates within a binary mask.

    Calculates the shortest path distances between all pairs of coordinates 
    that lie within a connected binary mask region. Uses the MCP_Geometric 
    algorithm from scikit-image to compute geodesic distances.

    :param mask_coords : numpy.ndarray
        2D array of shape (N, 2) containing (x, y) coordinates of pixels 
        within the binary mask.
    :return: numpy.ndarray
        Symmetric distance matrix of shape (N, N) where element (i, j) 
        contains the geodesic distance between coordinates i and j.
    :rtype: numpy.ndarray
    """
    mask_coords[:,0] = mask_coords[:,0] - mask_coords[:,0].min()
    mask_coords[:,1] = mask_coords[:,1] - mask_coords[:,1].min()
    cell_mask = np.zeros((mask_coords[:,0].max()+1, mask_coords[:,1].max()+1))
    cell_mask[mask_coords[:,0], mask_coords[:,1]] = 1
    # Initialize MCP_Geometric with the mask (cost=1 for foreground, inf for background)
    cost_array = np.where(cell_mask > 0, 1, np.inf)
    mcp = ski.graph.MCP_Geometric(cost_array)
    # Compute geodesic distances from each pixel to all others
    N = len(mask_coords)
    geodesic_dmat = np.zeros((N, N))
    for i, start in enumerate(mask_coords):
        costs, traceback = mcp.find_costs([tuple(start)])
        geodesic_dmat[i] = costs[mask_coords[:,0], mask_coords[:,1]]
    return(geodesic_dmat)



class GW_OT_Cell:
    """A cell representation for Gromov-Wasserstein Optimal Transport analysis.

    This class encapsulates cell morphology and intensity information needed 
    for Gromov-Wasserstein distance computations and optimal transport analysis 
    between cells.

    :param coords: Array of shape (N, 2) containing (x, y) coordinates for each pixel in the cell.
    :type coords: numpy.ndarray
    :param boundary_coords: Array of shape (M, 2) containing (x, y) coordinates sampled from the cell boundary. Default is None.
    :type boundary_coords: numpy.ndarray or None
    :param intensities: Dictionary mapping channel names (str) to intensity arrays (numpy.ndarray) 
        of length N, where N is the number of cell pixels. Default is None.
    :type intensities: dict or None
    :param nucleus: Array of length N indicating nuclear identity (0 or 1) for each cell pixel. Default is None.
    :type nucleus: numpy.ndarray or None
    :param metric: Distance metric for computing coordinate distance matrices. Options are 'euclidean', 'geodesic', or None. Default is 'euclidean'.
    :type metric: str or None
    """
    def __init__(self, coords, boundary_coords=None, intensities=None, nucleus=None, metric='euclidean'):
        self.coords = coords
        self.boundary_coords = boundary_coords
        if metric is None:
            self.coord_dmat = None
            self.boundary_coord_dmat = None
        elif metric == 'geodesic':
            self.coord_dmat = compute_geodesic_dmat(self.coords)
            self.boundary_coord_dmat = squareform(pdist(boundary_coords, metric=metric)) if boundary_coords is not None else None
            # if boundary_coords is not None:
            #     warnings.warn("Geodesic distance matrix cannot be computed for cell boundary coordinates, ignoring.")
        else:
            self.coord_dmat = squareform(pdist(coords, metric=metric))
            self.boundary_coord_dmat = squareform(pdist(boundary_coords, metric=metric)) if boundary_coords is not None else None
        self.intensities = intensities if intensities is not None else {}
        self.nucleus = nucleus
        self.size = len(coords)

    def copy(self):
        """Return a deep copy of the GW_OT_Cell instance.

        Returns
        -------
        GW_OT_Cell
            A new GW_OT_Cell instance with copied data from the current object.
            All arrays and dictionaries are deep-copied to avoid shared references.
        """
        obj_copy = GW_OT_Cell(
            coords=self.coords.copy(),
            boundary_coords=self.boundary_coords.copy() if self.boundary_coords is not None else None,
            intensities=copy.deepcopy(self.intensities),
            nucleus=self.nucleus.copy() if self.nucleus is not None else None
        )
        obj_copy.coord_dmat = self.coord_dmat.copy() if self.coord_dmat is not None else None
        obj_copy.boundary_coord_dmat = self.boundary_coord_dmat.copy() if self.boundary_coord_dmat is not None else None
        obj_copy.size = self.size
        return obj_copy


def process_image(image, channels, cell_mask_image, nucleus_mask_image=None, ds_factor=None, ds_target_size=None, 
    filter_border_cells=True, n_boundary_points=100, save_path=None, return_objects=True):
    """Create GW_OT_Cell objects from segmented microscopy images.

    Processes a multi-channel microscopy image with cell and nuclear segmentation 
    masks to create a list of GW_OT_Cell objects suitable for Gromov-Wasserstein 
    optimal transport analysis.

    :param image: numpy.ndarray
        3D numpy array of shape (H, W, C) representing the multi-channel image.
    :param channels: list of str
        List of channel names corresponding to the last dimension of the image.
    :param cell_mask_image: numpy.ndarray
        2D numpy array of shape (H, W) with integer labels for each cell 
        (0 for background).
    :param nucleus_mask_image: numpy.ndarray, optional
        2D numpy array of shape (H, W) with integer labels for nuclei 
        (0 for background). Default is None.
    :param ds_factor: int, optional
        Downsampling factor. If provided, downsample by this factor. 
        Default is None.
    :param ds_target_size: int, optional
        Target number of pixels per cell after downsampling. If provided, 
        downsample to achieve this pixel count. Default is None.
    :param filter_border_cells: bool, optional
        If True, exclude cells touching the image border. Default is True.
    :param n_boundary_points: int, optional
        Number of points to sample from the cell boundary. If None, boundary 
        sampling is skipped. Default is 100.
    :param save_path: str, optional
        Directory path to save the processed cell objects as pickle files. 
        If None, objects are not saved. Default is None.
    :param return_objects: bool, optional
        If True, return the list of GW_OT_Cell objects. Default is True.
    :return: If return_objects is True, returns a list of GW_OT_Cell objects; otherwise returns None.
    :rtype: list[GW_OT_Cell] or None
    """
    cell_inds = np.unique(cell_mask_image)
    cell_inds = cell_inds[cell_inds > 0]  # Remove background (0)

    gw_ot_cells = []
    for cell_ind in cell_inds:
        cell_mask = (cell_mask_image == cell_ind).astype(np.uint8)
        nuc_mask = (nucleus_mask_image == cell_ind).astype(np.uint8)  if nucleus_mask_image is not None else None
        # Filter out cells touching the border
        if filter_border_cells:
            if np.any(cell_mask[0, :]) or np.any(cell_mask[-1, :]) or np.any(cell_mask[:, 0]) or np.any(cell_mask[:, -1]):
                continue
        # Downsample image and masks if necessary
        if ds_factor is not None: # downsample by a factor
            image_ds = ski.transform.resize(image, (image.shape[0] // ds_factor, image.shape[1] // ds_factor, image.shape[2]), order=1, preserve_range=True).astype(np.uint8)
            cell_mask_ds = ski.transform.resize(cell_mask, (cell_mask.shape[0] // ds_factor, cell_mask.shape[1] // ds_factor), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
            nuc_mask_ds = ski.transform.resize(nuc_mask, (nuc_mask.shape[0] // ds_factor, nuc_mask.shape[1] // ds_factor), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8) if nuc_mask is not None else None
        elif ds_target_size is not None: # downsample to a target size
            cell_mask_ds = rescale_mask_to_pixel_count(cell_mask, target_pixels=ds_target_size)
            nuc_mask_ds = ski.transform.resize(nuc_mask, (cell_mask_ds.shape[0], cell_mask_ds.shape[1]), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8) if nuc_mask is not None else None
            image_ds = ski.transform.resize(image, (cell_mask_ds.shape[0], cell_mask_ds.shape[1], image.shape[2]), order=1, preserve_range=True).astype(np.uint8)
        else:
            image_ds = image
            cell_mask_ds = cell_mask
            nuc_mask_ds = nuc_mask
        # Create a cell object
        # Sample points from cell boundary (if specified)
        cell_boundary_pts = None
        if n_boundary_points is not None:
            _, cell_boundary_pts = cell_boundaries(np.pad(cell_mask, 1), n_sample=n_boundary_points)[0] # pad to avoid border issues
            cell_boundary_pts = cell_boundary_pts - 1 # remove padding
        gw_ot_cell = GW_OT_Cell(coords=np.array(np.where(cell_mask_ds)).T, boundary_coords=cell_boundary_pts)
        if nucleus_mask_image is not None:
            gw_ot_cell.nucleus = nuc_mask_ds[np.where(cell_mask_ds)]
            if gw_ot_cell.nucleus.sum() == 0: # filter cells without segmented nuclei
                continue
        for channel in channels:
            gw_ot_cell.intensities[channel] = image_ds[np.where(cell_mask_ds)][:,channels.index(channel)]
        # Normalize the channels (to sum to 1)
        for channel in channels:
            gw_ot_cell.intensities[channel] = gw_ot_cell.intensities[channel] / np.sum(gw_ot_cell.intensities[channel])
        if return_objects:
            gw_ot_cells.append(gw_ot_cell)
        if save_path is not None:
            if not os.path.isdir(save_path): # Create directory if it doesn't exist
                os.makedirs(save_path)
            with open(os.path.join(save_path, 'cell_'+str(cell_ind).zfill(4)+'.pickle'), 'wb') as file:
                pickle.dump(gw_ot_cell, file)
    if return_objects:
        return gw_ot_cells
    else:
        return None


def _init_gw_pool(cell_objects: list, points: str):
    """Initialize global variables for parallel Gromov-Wasserstein computation.

    This function sets up shared state for multiprocessing workers computing 
    pairwise Gromov-Wasserstein distances.

    :param cell_objects: list
        List of GW_OT_Cell objects or paths to pickled GW_OT_Cell objects.
    :param points: str
        Type of points to use for distance computation. Either 'boundary' 
        for boundary coordinates or 'full' for all cell coordinates.
    """
    # list of GW_OT_Cell objects or list of paths to GW_OT_Cell objects
    global _CELL_OBJECTS
    _CELL_OBJECTS = cell_objects
    # set of points to use for distance computation ('boundary' or 'full')
    global _POINTS
    _POINTS = points


def _gw_index(p: tuple[int, int]):
    """Worker that computes the GW coupling and distance for a pair of cells.

    :param p: tuple (i, j) of cell indices
    :type p: tuple[int, int]
    :return: (i, j, coupling_mat, gw_dist)
    :rtype: tuple[int, int, numpy.ndarray, float]
    """
    i, j = p
    # load GW_OT_Cell objects if path specified
    if isinstance(_CELL_OBJECTS[i], str):
        _CELL_OBJECTS[i] = pickle.load(open(_CELL_OBJECTS[i], 'rb'))
    if isinstance(_CELL_OBJECTS[j], str):
        _CELL_OBJECTS[j] = pickle.load(open(_CELL_OBJECTS[j], 'rb'))
    if _POINTS == 'boundary':
        A = _CELL_OBJECTS[i].boundary_coord_dmat
        B = _CELL_OBJECTS[j].boundary_coord_dmat
    elif _POINTS == 'full':
        A = _CELL_OBJECTS[i].coord_dmat
        B = _CELL_OBJECTS[j].coord_dmat
    n_A = A.shape[0]
    n_B = B.shape[0]
    a = np.repeat(1/n_A, n_A)
    b = np.repeat(1/n_B, n_B)
    a_dot_dist = A@a
    b_dot_dist = B@b
    a_cell_constant = ((A * A)@a)@a
    b_cell_constant = ((B * B)@b)@b
    
    coupling_mat, gw_dist = gw_cython_core(
        A,
        a,
        a_dot_dist,
        a_cell_constant,
        B,
        b,
        b_dot_dist,
        b_cell_constant,
    )

    return (i, j, coupling_mat, gw_dist) 

def gw_pairwise_parallel(cell_objects, points='boundary', num_processes=4, chunksize=20, n_approx_anchors=None, initial_anchor=0):
    """Compute pairwise Gromov-Wasserstein distances (optionally in parallel).

    Calculates the Gromov-Wasserstein distance matrix for a colloection of cells 
    using either exact computation or traiangle inequality approximation with anchors.

    :param cell_objects: list of GW_OT_Cell objects or file paths
    :type cell_objects: list
    :param points: 'boundary' or 'full' (default: 'boundary')
    :type points: str
    :param num_processes: number of parallel processes to use
    :type num_processes: int
    :param chunksize: chunk size for parallel imap
    :type chunksize: int
    :param n_approx_anchors: number of anchors for approximation (None = exact)
    :type n_approx_anchors: int or None
    :param initial_anchor: initial anchor index for approximation
    :type initial_anchor: int
    :return: symmetric GW distance matrix of shape (N, N)
    :rtype: numpy.ndarray
    """
    N = len(cell_objects)
    # Compute all pairwise GW distances
    if n_approx_anchors is None:
        index_pairs = it.combinations(iter(range(N)), 2)
        total_num_pairs = int((N * (N - 1)) / 2) 
        with Pool(
            initializer=_init_gw_pool, initargs=(cell_objects,points,), processes=num_processes
        ) as pool:
            res = pool.imap_unordered(_gw_index, index_pairs, chunksize=chunksize)
            gw_dmat = np.zeros((N,N))
            for i, j, coupling_mat, gw_dist in tqdm(res, total=total_num_pairs, position=0, leave=True):
                gw_dmat[i,j] = gw_dist
                gw_dmat[j,i] = gw_dist
    # Approximate GW distances using triangle inequality
    else:
        anchor_ind = initial_anchor
        all_anchor_gw_dists = np.zeros((n_approx_anchors,N))
        for i_anchor in range(n_approx_anchors):
            anchor_gw_dists = np.zeros(N)
            index_pairs = it.product(iter(range(N)), [anchor_ind]) 
            total_num_pairs = N 
            with Pool(
                initializer=_init_gw_pool, initargs=(cell_objects,points,), processes=num_processes
            ) as pool:
                res = pool.imap_unordered(_gw_index, index_pairs, chunksize=chunksize)
                for i, j, coupling_mat, gw_dist in tqdm(res, total=total_num_pairs):
                    anchor_gw_dists[i] = gw_dist
                all_anchor_gw_dists[i_anchor,:] = anchor_gw_dists
                anchor_ind = np.argmax(all_anchor_gw_dists[:i_anchor+1,:].min(axis=0)) # next anchor
        gw_dmat = np.zeros((N,N))
        for i,j in it.combinations(range(N), 2):
            d = min(all_anchor_gw_dists[:,i] + all_anchor_gw_dists[:,j]) 
            gw_dmat[i,j] = d
            gw_dmat[j,i] = d
    return gw_dmat


def find_centroid(distance_matrix):
    """Return the index of the centroid point (minimizes sum of distances).

    :param distance_matrix: square symmetric pairwise distance matrix
    :type distance_matrix: numpy.ndarray
    :return: index of centroid point
    :rtype: int
    """
    sum_distances = np.sum(distance_matrix, axis=1)
    centroid_index = np.argmin(sum_distances)
    return centroid_index


def _init_fgw_map_pool(cell_objects: list, channels: list, compartment_specific: bool, method, 
                       fused_channel: str, fused_cost: float, fused_param: float, unbalanced_param: float, 
                       nuclear_fraction: float = 0.2):
    """Initialize global state for parallel fused GW mapping workers.

    :param cell_objects: list of GW_OT_Cell objects or paths
    :type cell_objects: list
    :param channels: channel names to compute OT for
    :type channels: list[str]
    :param compartment_specific: whether to perform compartment-specific mapping
    :type compartment_specific: bool
    :param method: mapping method ('fused' or 'fused_unbalanced')
    :type method: str
    :param fused_channel: channel name used for fused mapping
    :type fused_channel: str
    :param fused_cost: fused channel cost multiplier
    :type fused_cost: float
    :param fused_param: alpha parameter for fused GW
    :type fused_param: float
    :param unbalanced_param: regularization for unbalanced fused GW
    :type unbalanced_param: float
    :param nuclear_fraction: fraction considered nuclear during mapping
    :type nuclear_fraction: float
    """
    global _CELL_OBJECTS #
    _CELL_OBJECTS = cell_objects # list of GW_OT_Cell objects or list of paths to GW_OT_Cell objects
    global _CHANNELS
    _CHANNELS = channels # which channels to compute protein OT for
    global _COMPARTMENT_SPECIFIC
    _COMPARTMENT_SPECIFIC = compartment_specific # whether to do compartment-specific mapping (nuclear/cytoplasm)
    global _METHOD
    _METHOD = method # method for morphology mapping: 'fused' or 'fused_unbalanced'
    global _FUSED_CHANNEL
    _FUSED_CHANNEL = fused_channel # channel to use for fused GW morphology mapping
    global _FUSED_COST
    _FUSED_COST = fused_cost # cost for fused GW morphology mapping
    global _FUSED_PARAM
    _FUSED_PARAM = fused_param # parameter for fused/unbalanced GW morphology mapping
    global _UNBALANCED_PARAM
    _UNBALANCED_PARAM = unbalanced_param # parameter for fused unbalanced GW mapping
    global _NUC_FRAC
    _NUC_FRAC = nuclear_fraction # fraction of cell considered as nucleus for compartment-specific


def _fgw_map_index(p: tuple[int, int]):
    """Worker that computes fused GW mapping and maps protein distributions.

    :param p: tuple (i, j) of source and target cell indices
    :type p: tuple[int, int]
    :return: (i, j, gw_dist, mapped_distbs)
    :rtype: tuple[int, int, float, numpy.ndarray]
    """
    i, j = p
    # load GW_OT_Cell objects if path specified
    if isinstance(_CELL_OBJECTS[i], str):
        _CELL_OBJECTS[i] = pickle.load(open(_CELL_OBJECTS[i], 'rb'))
    if isinstance(_CELL_OBJECTS[j], str):
        _CELL_OBJECTS[j] = pickle.load(open(_CELL_OBJECTS[j], 'rb'))
    A = _CELL_OBJECTS[i].coord_dmat
    B = _CELL_OBJECTS[j].coord_dmat
    n_A = A.shape[0]
    n_B = B.shape[0]

    # dictionary to store fGW morphology and OT protein distances
    mapped_distbs = np.zeros((len(_CHANNELS),n_B))
    
    if _COMPARTMENT_SPECIFIC:
        # rescaling probabilities to allow fused GW to map nucleus to nucleus, cytoplasm to cytoplasm
        n_pixel_nuc_i = _CELL_OBJECTS[i].nucleus.sum()
        n_pixel_cyto_i = n_A - n_pixel_nuc_i
        n_pixel_nuc_j = _CELL_OBJECTS[j].nucleus.sum()
        n_pixel_cyto_j = n_B - n_pixel_nuc_j

        # rescale uniform distribution in cell i to have same nuclear/cytoplasm ration as cell j
        a = np.zeros(n_A)
        a[_CELL_OBJECTS[i].nucleus==1] = _NUC_FRAC / n_pixel_nuc_i
        a[_CELL_OBJECTS[i].nucleus==0] = (1 - _NUC_FRAC) / n_pixel_cyto_i
        b = np.zeros(n_B)
        b[_CELL_OBJECTS[j].nucleus==1] = _NUC_FRAC / n_pixel_nuc_j
        b[_CELL_OBJECTS[j].nucleus==0] = (1 - _NUC_FRAC) / n_pixel_cyto_j
    else:
        a = np.repeat(1/n_A, n_A)
        b = np.repeat(1/n_B, n_B)

    # fGW morphology mapping
    alpha = _FUSED_PARAM
    cost = _FUSED_COST
    rho = _UNBALANCED_PARAM
    if i == j:
        gw_dist = 0
        coupling_mat = np.eye(n_A)
    else:
        if _FUSED_CHANNEL == 'nucleus':
            cost_matrix = cdist(_CELL_OBJECTS[i].nucleus[:,np.newaxis], _CELL_OBJECTS[j].nucleus[:,np.newaxis],) * cost
        else:
            cost_matrix = cdist(_CELL_OBJECTS[i].intensities[_FUSED_CHANNEL][:,np.newaxis], _CELL_OBJECTS[j].intensities[_FUSED_CHANNEL][:,np.newaxis],) * cost
        if _METHOD == 'fused':
            coupling_mat, log = ot.gromov.fused_gromov_wasserstein(M=cost_matrix, C1=A, C2=B, p=a, q=b, alpha=alpha, log=True)
            gw_dist = log['fgw_dist']
        elif _METHOD == 'fused_unbalanced':
            coupling_mat, coupling_mat_2, log = ot.gromov.fused_unbalanced_gromov_wasserstein(M=cost_matrix, Cx=A, Cy=B, wx=a, wy=b, alpha=alpha, 
                                                                                            reg_marginals=rho, max_iter=20, log=True)
            gw_dist = log['fugw_cost']

    if _COMPARTMENT_SPECIFIC:
        # find nuclear pixels after mapping
        mapped_nucleus = _CELL_OBJECTS[i].nucleus.dot(coupling_mat * n_A)
        mapped_nucleus_thresh = ( np.quantile(mapped_nucleus, 0.9) + np.quantile(mapped_nucleus, 0.1) ) / 2
        mapped_is_nucleus = mapped_nucleus > mapped_nucleus_thresh
        n_pixel_nuc_mapped = mapped_is_nucleus.sum()
        if _METHOD == 'fused_unbalanced':
            mapped_cyto = np.repeat(1, n_A).dot(coupling_mat * n_A)
            mapped_cyto[mapped_is_nucleus] = 0
            mapped_cyto_thresh = ( np.quantile(mapped_cyto, 0.7) + np.quantile(mapped_cyto, 0.1) ) / 2
            mapped_is_cyto = mapped_cyto > mapped_cyto_thresh
            n_pixel_cyto_mapped = mapped_is_cyto.sum()
            mapping_cyto = np.repeat(1, n_B).dot(coupling_mat.T * n_B)
            mapping_cyto[_CELL_OBJECTS[i].nucleus==1] = 0
            mapping_cyto_thresh = ( np.quantile(mapping_cyto, 0.7) + np.quantile(mapping_cyto, 0.1) ) / 2
            mapping_is_cyto = mapping_cyto > mapping_cyto_thresh
            n_pixel_cyto_mapping = mapped_is_cyto.sum()
        else:
            n_pixel_cyto_mapped = n_B - n_pixel_nuc_mapped

    # mapping cell A's distribution on cell B
    for k in range(len(_CHANNELS)):
        channel = _CHANNELS[k]

        if channel == 'nucleus':
            a = _CELL_OBJECTS[i].nucleus.dot(coupling_mat * n_A)
        else:
            a = _CELL_OBJECTS[i].intensities[channel].dot(coupling_mat * n_A)

        if _COMPARTMENT_SPECIFIC:
            # rescale based on protein distribution in nucleus/cytoplasm
            if _METHOD == 'fused_unbalanced':
                if a[mapped_is_nucleus].sum() != 0:
                    a[mapped_is_nucleus] = a[mapped_is_nucleus] / a[mapped_is_nucleus].sum() * a[_CELL_OBJECTS[j].nucleus==1].sum()
                if a[~mapped_is_nucleus].sum() != 0:
                    a[~mapped_is_nucleus] = a[~mapped_is_nucleus] / n_pixel_cyto_mapped * n_pixel_cyto_mapping
            else:
                if a[mapped_is_nucleus].sum() != 0:
                    a[mapped_is_nucleus] = a[mapped_is_nucleus] / a[mapped_is_nucleus].sum() * _CELL_OBJECTS[i].intensities[channel][_CELL_OBJECTS[i].nucleus==1].sum()
                if a[~mapped_is_nucleus].sum() != 0:
                    a[~mapped_is_nucleus] = a[~mapped_is_nucleus] / a[~mapped_is_nucleus].sum() * _CELL_OBJECTS[i].intensities[channel][_CELL_OBJECTS[i].nucleus==0].sum()
            # then rescale based on number nuclear/cytoplasm pixels
            a[mapped_is_nucleus] *= n_pixel_nuc_j/n_B*n_A/n_pixel_nuc_i
            a[~mapped_is_nucleus] *= n_pixel_cyto_j/n_B*n_A/n_pixel_cyto_i 

        # final normalization
        a = a / a.sum()
        
        mapped_distbs[k,:] = a

    return (i, j, gw_dist, mapped_distbs) 


def map_to_cell(cell_object_from, cell_object_to, channels, compartment_specific=True, method='fused', 
                fused_channel='protein', fused_cost=10, fused_param=0.1, unbalanced_param=70, nuclear_fraction=0.2):
    """Map protein distributions from one cell onto another via Fused Gromov-Wasserstein.

    :param cell_object_from: source ``GW_OT_Cell``
    :type cell_object_from: GW_OT_Cell
    :param cell_object_to: target ``GW_OT_Cell``
    :type cell_object_to: GW_OT_Cell
    :param channels: list of channel names to map
    :type channels: list[str]
    :param compartment_specific: whether to use compartment-specific mapping
    :type compartment_specific: bool
    :param method: 'fused' or 'fused_unbalanced'
    :type method: str
    :param fused_channel: channel name to use for fused morphology cost
    :type fused_channel: str
    :param fused_cost: fused channel cost multiplier
    :type fused_cost: float
    :param fused_param: alpha parameter for fused GW
    :type fused_param: float
    :param unbalanced_param: regularization for unbalanced GW
    :type unbalanced_param: float
    :param nuclear_fraction: nuclear fraction for compartment scaling
    :type nuclear_fraction: float
    :return: mapped distributions with shape (len(channels), n_target_pixels)
    :rtype: numpy.ndarray
    """
    mapped_distbs = map_to_cell_parallel(
        cell_objects=[cell_object_from, cell_object_to],
        channels=channels,
        target_cell_ind=1,
        compartment_specific=compartment_specific,
        method=method,
        fused_channel=fused_channel,
        fused_cost=fused_cost,
        fused_param=fused_param,
        unbalanced_param=unbalanced_param,
        nuclear_fraction=nuclear_fraction,
        parallel=False
    )
    return mapped_distbs[:,0,:]


def map_to_cell_parallel(cell_objects, channels, target_cell_ind, compartment_specific=True, method='fused', 
                         fused_channel='protein', fused_cost=10, fused_param=0.1, unbalanced_param=70, parallel=True,
                         nuclear_fraction=0.2, num_processes=4, chunksize=20):
    """Map protein distributions from all cells to a single target cell.

    :param cell_objects: list of GW_OT_Cell objects or file paths
    :type cell_objects: list
    :param channels: channel names to map
    :type channels: list[str]
    :param target_cell_ind: index of the target cell
    :type target_cell_ind: int
    :param compartment_specific: whether to use compartment-specific mapping
    :type compartment_specific: bool
    :param method: mapping method ('fused' or 'fused_unbalanced')
    :type method: str
    :param fused_channel: channel used for fused cost
    :type fused_channel: str
    :param fused_cost: cost multiplier for fused channel
    :type fused_cost: float
    :param fused_param: alpha parameter for fused GW
    :type fused_param: float
    :param unbalanced_param: regularization for unbalanced GW
    :type unbalanced_param: float
    :param parallel: whether to run in parallel
    :type parallel: bool
    :param num_processes: number of processes for parallel execution
    :type num_processes: int
    :param chunksize: chunk size for parallel map
    :type chunksize: int
    :return: array of mapped distributions with shape (len(channels), N, n_target_pixels)
    :rtype: numpy.ndarray
    """
    print('Mapping cells to target cell:')
    N = len(cell_objects)
    index_pairs = [(i, target_cell_ind) for i in range(N)]
    total_num_pairs = N - 1 
    if parallel:
        # Parallelized
        with Pool(
            initializer=_init_fgw_map_pool, initargs=(cell_objects, channels, compartment_specific, method, 
                                                    fused_channel, fused_cost, fused_param, unbalanced_param, nuclear_fraction), 
            processes=num_processes
        ) as pool:
            res = pool.imap_unordered(_fgw_map_index, index_pairs, chunksize=chunksize)
            target_cell_object = cell_objects[target_cell_ind]
            # load target GW_OT_Cell object if path specified
            if isinstance(target_cell_object, str):
                target_cell_object = pickle.load(open(target_cell_object, 'rb'))
            mapped_distbs = np.zeros((len(channels),N,target_cell_object.coord_dmat.shape[0]))
            for i, j, gw_dist, mapped_distb in tqdm(res, total=total_num_pairs, position=0, leave=True):
                mapped_distbs[:,i,:] = mapped_distb
    else:
        # Non-parallelized
        _init_fgw_map_pool(cell_objects, channels, compartment_specific, method, fused_channel, fused_cost, 
                           fused_param, unbalanced_param, nuclear_fraction)
        mapped_distbs = np.zeros((len(channels),N,cell_objects[target_cell_ind].coord_dmat.shape[0]))
        for p in tqdm(index_pairs):
            i, j, gw_dist, mapped_distb = _fgw_map_index(p)
            mapped_distbs[:,i,:] = mapped_distb
    return mapped_distbs


def _init_gw_mapped_ot_pool(cell_object: GW_OT_Cell, mapped_cell_dists: np.ndarray):
    """Initialize global state for OT computation on mapped distributions.

    :param cell_object: target ``GW_OT_Cell`` containing coordinate distance matrix
    :type cell_object: GW_OT_Cell
    :param mapped_cell_dists: mapped distributions array (n_channels, N, n_target_pixels)
    :type mapped_cell_dists: numpy.ndarray
    """
    global _CELL_OBJECT
    _CELL_OBJECT = cell_object # GW_OT_Cell object
    global _MAPPED_CELL_DISTS
    _MAPPED_CELL_DISTS = mapped_cell_dists # numpy array storing mapped cell protein distributions


def _gw_mapped_ot_index(p: tuple[int, int]):
    """Worker that computes OT distances between two mapped distributions.

    :param p: tuple (i, j) of indices
    :type p: tuple[int, int]
    :return: (i, j, ot_dists) where ot_dists is an array per channel
    :rtype: tuple[int, int, numpy.ndarray]
    """
    global _CELL_OBJECT
    i, j = p
    if isinstance(_CELL_OBJECT, str):
        _CELL_OBJECT = pickle.load(open(_CELL_OBJECT, 'rb'))
    n_channels = _MAPPED_CELL_DISTS.shape[0]
    ot_dists = np.zeros(n_channels)

    # protein OT
    for channel_i in range(n_channels):
        a = _MAPPED_CELL_DISTS[channel_i,i,:]
        b = _MAPPED_CELL_DISTS[channel_i,j,:]
        coupling_mat_prot, emd_dict = ot.emd(a, b, _CELL_OBJECT.coord_dmat, log=True)
        gw_dist_prot = emd_dict['cost']
        ot_dists[channel_i] = gw_dist_prot

    return (i, j, ot_dists)


def gw_mapped_ot_pairwise_parallel(cell_object, mapped_cell_dists, num_processes=4, chunksize=20, index_pairs=None, n_approx_anchors=None, initial_anchor=0):
    """Compute pairwise OT distances between mapped protein distributions.

    Calculates pairwise optimal transport distances for protein distribution after 
    mapping to a common cell morphology.

    :param cell_object: target ``GW_OT_Cell`` or path to pickled object
    :type cell_object: GW_OT_Cell or str
    :param mapped_cell_dists: array of mapped distributions (len(channels), N, n_target_pixels)
    :type mapped_cell_dists: numpy.ndarray
    :param num_processes: number of processes for parallel execution
    :type num_processes: int
    :param chunksize: chunk size for parallel imap
    :type chunksize: int
    :param index_pairs: optional iterable of (i, j) index pairs to compute
    :type index_pairs: iterable[tuple[int, int]] or None
    :param n_approx_anchors: number of anchors for triangle inequality approximation
    :type n_approx_anchors: int or None
    :param initial_anchor: initial anchor index
    :type initial_anchor: int
    :return: if ``index_pairs`` is None returns array (len(channels), N, N), else (len(channels), len(index_pairs))
    :rtype: numpy.ndarray
    """
    print('Computing pairwise OT distances:')
    N = mapped_cell_dists.shape[1]
    # Compute all pairwise OT distances or specific specific pairs
    if n_approx_anchors is None:
        if index_pairs is None:
            index_pairs = list(it.combinations(range(N), 2))
            total_num_pairs = int((N * (N - 1)) / 2)
            output_full_matrix = True
        else:
            index_pairs = list(index_pairs)
            total_num_pairs = len(index_pairs)
            output_full_matrix = False

        with Pool(
            initializer=_init_gw_mapped_ot_pool, initargs=(cell_object, mapped_cell_dists,), processes=num_processes
        ) as pool:
            res = pool.imap(_gw_mapped_ot_index, index_pairs, chunksize=chunksize)
            if output_full_matrix:
                ot_dmats = np.zeros((mapped_cell_dists.shape[0], N, N))
                for i, j, ot_dists in tqdm(res, total=total_num_pairs, position=0, leave=True):
                    ot_dmats[:, i, j] = ot_dists
                    ot_dmats[:, j, i] = ot_dists
                return ot_dmats
            else:
                ot_dists_arr = np.zeros((mapped_cell_dists.shape[0], total_num_pairs))
                for idx, (i, j, ot_dists) in enumerate(tqdm(res, total=total_num_pairs, position=0, leave=True)):
                    ot_dists_arr[:, idx] = ot_dists
                return ot_dists_arr
    # Approximate OT distances using triangle inequality
    else:
        if index_pairs is not None:
            raise ValueError("index_pairs cannot be specified when using triangle inequality approximation.")
        anchor_ind = initial_anchor
        all_anchor_ot_dists = np.zeros((mapped_cell_dists.shape[0], n_approx_anchors, N))
        for i_anchor in range(n_approx_anchors):
            anchor_ot_dists = np.zeros((mapped_cell_dists.shape[0], N))
            index_pairs = it.product(iter(range(N)), [anchor_ind]) 
            total_num_pairs = N 
            with Pool(
                initializer=_init_gw_mapped_ot_pool, initargs=(cell_object, mapped_cell_dists,), processes=num_processes
            ) as pool:
                res = pool.imap(_gw_mapped_ot_index, index_pairs, chunksize=chunksize)
                for i, j, ot_dists in tqdm(res, total=total_num_pairs, position=0, leave=True):
                    anchor_ot_dists[:, i] = ot_dists
                all_anchor_ot_dists[:,i_anchor,:] = anchor_ot_dists
                anchor_ind = np.argmax(all_anchor_ot_dists.mean(axis=0)[:i_anchor+1,:].min(axis=0)) # next anchor
        ot_dmats = np.zeros((mapped_cell_dists.shape[0], N, N))
        for channel_i in range(mapped_cell_dists.shape[0]):
            for i,j in it.combinations(range(N), 2):
                d = min(all_anchor_ot_dists[channel_i,i] + all_anchor_ot_dists[channel_i,j]) 
                ot_dmats[channel_i, i, j] = d
                ot_dmats[channel_i, j, i] = d
        return ot_dmats