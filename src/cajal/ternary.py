import matplotlib.pyplot as plt
import mpltern  # noqa : F401
from math import sqrt
import numpy as np
import numpy.typing as npt
from typing import Literal, Optional

from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
from .run_gw import DistanceMatrix, Matrix


def normalize_to_unit_mean(A: DistanceMatrix):
    """
    :param A: A distance matrix.
    :returns: The distance matrix `A`, rescaled so that the
        mean distance between distinct points is 1.
    """
    v = squareform(A, force="tovector")
    mean = np.average(v)
    return A / mean


# This is the inner radius of the standard unit simplex, i.e. the distance
# between (1/3,1/3,1/3) and (0,0.5,0.5).
MAGIC_NUMBER_1 = sqrt(1 / 6)

# This is a scaling constant
MAGIC_NUMBER_2 = (1 - sqrt(3)) / 2


def two_d_projection(xyz: Matrix) -> Matrix:
    """
    Given a matrix of points (x,y,z) lying on the plane x+y+z=1, project down
    onto the plane z=0 in an isometric way.  This rotates the plane about
    x+y=1,z=0 to send points in the plane x+y+z=1 to the plane z=0.

    :param xyz: shape (n,3).
    :return: shape (n,2), an (isometric) projection of xyz onto the plane z=0.
    """
    return xyz[:, 0:2] + MAGIC_NUMBER_2 * xyz[:, 2][:, np.newaxis]


def histogram_density(xyz: Matrix, bins: int) -> npt.NDArray[np.float64]:
    """
    Compute density estimates for the point cloud `xyz`. Assigns each point a
    nonnegative floating point number estimating the local density of the point
    cloud in that region using a simple two-dimensional histogram.

    :param xyz: A matrix of shape (n,3), where each row represents one point;
        points are assumed to lie in the plane x+y+z=1.
    :param bins: How many bins to use for the histogram in each dimension.
        If fewer bins are chosen, the coloring will be more homogeneous and
        change gradually. If more bins are chosen, the coloring will vary more.
    """
    xy = two_d_projection(xyz)
    density, xbins, ybins = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=40, range=None, density=True
    )
    xinds = np.minimum(np.searchsorted(xbins, xy[:, 0]), xbins.shape[0] - 2)
    yinds = np.minimum(np.searchsorted(ybins, xy[:, 1]), ybins.shape[0] - 2)
    return density[xinds, yinds]


def gaussian_density(xyz: Matrix) -> npt.NDArray[np.float64]:
    """
    Compute density estimates for the point cloud `xyz`. Assigns each point a
    nonnegative floating point number estimating the local density of the
    point cloud in that region using the scipy.stats.gaussian_kde function.

    :param xyz: A matrix of shape (n,3), where each row represents one point;
        points are assumed to lie in the plane x+y+z=1.
    """
    xy = two_d_projection(xyz)
    return gaussian_kde(xy.T)(xy.T)


def normalize(
    feature_dispersion: DistanceMatrix,
    add_one: bool = False,
    center_at_zero: bool = True,
):
    """
    Standardize the data as controlled by the arguments.
    """
    fd = squareform(feature_dispersion, force="tovector")
    if add_one:
        fd += 1
    fd = np.log(fd)
    if center_at_zero:
        fd -= np.mean(fd)
    fd /= np.std(fd)
    return fd


def normalized_relative_dispersion(
    feature1_dispersion: DistanceMatrix,
    feature2_dispersion: DistanceMatrix,
    feature3_dispersion: DistanceMatrix,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Given three morphology spaces, normalize each one by scaling it to its
    average, compute the three pairwise distributions of relative differences,
    and rescale this data so that >90% of it lies in the unit simplex in R^3.
    """
    assert feature1_dispersion.shape == feature2_dispersion.shape
    assert feature2_dispersion.shape == feature3_dispersion.shape

    f1d = normalize(feature1_dispersion)
    f2d = normalize(feature2_dispersion)
    f3d = normalize(feature3_dispersion)

    d12 = f1d - f2d
    d23 = f2d - f3d
    d31 = f3d - f1d

    d = np.stack((d12, d23, d31), axis=1)
    p01 = np.percentile(np.ndarray.flatten(d), 1)
    # print(p01)
    d -= p01
    p99 = np.percentile(np.ndarray.flatten(d), 99)
    # print(p99)
    d /= p99

    # After this transformation, 98% of values lie within the unit cube.
    # This puts points onto the unit plane.
    d /= np.sum(d[0, :])
    assert np.allclose(np.sum(d, axis=1), np.ones((f1d.shape[0],), dtype=float))
    return (d[:, 0], d[:, 1], d[:, 2])

    # percent = 90
    # percentile = np.percentile(norm(np.stack((d12, d23, d31), axis=1), axis=1),
    #                            percent)
    # normalizing_constant = MAGIC_NUMBER_1 / percentile
    # d12 = d12 * normalizing_constant + 1 / 3
    # d23 = d23 * normalizing_constant + 1 / 3
    # d31 = d31 * normalizing_constant + 1 / 3

    # assert np.allclose(d12 + d23 + d31, np.ones((f1d.shape[0],), dtype=float))
    # return d12, d23, d31


def ternary_distance(
    axis,  # Matplotlib Axes object
    d12,
    feature1_name: str,
    d23,
    feature2_name: str,
    d31,
    feature3_name: str,
    density_estimation: Literal["histogram"] | Literal["gaussian_kde"],
    title,
    bins: int,
    contour_lines: int = 4,
    mpl_params: dict = {},
):
    """
    Construct a ternary distance plot illustrating the relative variation in
    any one feature with respect to the others.

    :param bins: How many bins to use for the histogram in each dimension when
        estimating the gradient. If fewer bins are chosen, the coloring will be
        more homogeneous and change gradually. If more bins are chosen, the
        coloring will vary more.
    :param levels: How many contour lines to draw.
    """

    # (d12, d23, d31) = normalized_relative_dispersion(
    #     feature1_dispersion, feature2_dispersion, feature3_dispersion)
    xyz = np.stack((d12, d23, d31), axis=1)

    if density_estimation == "histogram":
        coloring = histogram_density(xyz, bins)
    else:
        coloring = gaussian_density(xyz)

        ticks = [0.33333]
        labels = ["0"]
        axis.taxis.set_ticks(ticks, labels)
        axis.laxis.set_ticks(ticks, labels)
        axis.raxis.set_ticks(ticks, labels)

    axis.set_tlabel(feature1_name + " - " + feature2_name)
    axis.set_llabel(feature3_name + " - " + feature1_name)
    axis.set_rlabel(feature2_name + " - " + feature3_name)
    axis.set_title(title)
    axis.grid()
    axis.scatter(d12, d31, d23, **mpl_params)
    level_marks = np.linspace(np.min(coloring), np.max(coloring), contour_lines + 2)
    plot = axis.tricontour(d12, d31, d23, coloring, level_marks)
    return plot


def ternary_distance_clusters(
    feature1_dispersion: DistanceMatrix,
    feature1_name: str,
    feature2_dispersion: DistanceMatrix,
    feature2_name: str,
    feature3_dispersion: DistanceMatrix,
    feature3_name: str,
    density_estimation: Literal["histogram"] | Literal["gaussian_kde"],
    bins: Optional[int] = None,
    contour_lines: int = 4,
    figsize: int = 4,
    clusters: Optional[npt.NDArray] = None,
    min_cluster_size=30,
    mpl_params: dict = {"s": 1, "alpha": 0.3},
):
    """
    :param density_estimation: Controls the method by which density of the input space is estimated.
    :param bins: How many bins to use for the histogram in each dimension when
        estimating the gradient. If fewer bins are chosen, the coloring will be
        more homogeneous and change gradually. If more bins are chosen, the
        coloring will vary more.
    :param contour_lines: How many contour lines to draw.
    :param figsize: Passed to matplotlib.pyplot.subplots.
    :param clusters: Labels for clusters, should be the same length as the distance matrices
        featurei_dispersion
    :param min_cluster_size: Ignore clusters below the threshold size (density plots are somewhat
        useless when there are very few observations)
    :param mpl_params: Passed to matplotlib.
    """

    d12, d23, d31 = normalized_relative_dispersion(
        feature1_dispersion, feature2_dispersion, feature3_dispersion
    )

    if clusters is not None:
        unique_clusters = list(np.unique(clusters))
        unique_clusters = np.array(
            [
                c
                for c in unique_clusters
                if np.count_nonzero(clusters == c) >= min_cluster_size
            ]
        )
        nfig = unique_clusters.shape[0]
    else:
        nfig = 1

    if clusters is not None:
        fig = plt.figure(figsize=(figsize, figsize * nfig))
        new_axes = []
        for i in range(0, nfig):
            cluster = unique_clusters[i]
            axis = plt.subplot(nfig, 1, i + 1, projection="ternary", ternary_sum=1.0)
            indices = clusters == cluster
            indices = np.logical_and(indices[:, np.newaxis], indices[np.newaxis, :])
            indices = squareform(indices, force="tovector", checks=False)

            f1 = d12[indices]
            f2 = d23[indices]
            f3 = d31[indices]
            new_axes.append(
                ternary_distance(
                    axis,
                    f1,
                    feature1_name,
                    f2,
                    feature2_name,
                    f3,
                    feature3_name,
                    density_estimation,
                    cluster,
                    bins,
                    contour_lines,
                    mpl_params,
                )
            )
        return fig, new_axes
    else:
        axis = ternary_distance(
            axis,
            feature1_dispersion,
            feature1_name,
            feature2_dispersion,
            feature2_name,
            feature3_dispersion,
            feature3_name,
            density_estimation,
            bins,
            contour_lines,
            mpl_params,
        )
        return fig, axis
