import matplotlib.pyplot as plt
import mpltern  # noqa : F401
from math import sqrt
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from typing import Literal

from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
from .run_gw import DistanceMatrix, Matrix


def normalize_to_unit_mean(A: DistanceMatrix):
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


def histogram_density(xyz: Matrix, bins: int) -> npt.NDArray[np.float_]:
    """
    Compute density estimates for the point cloud `xyz`. Assigns each point a
    nonnegative floating point number estimating the local density of the point cloud in that region
    using a simple two-dimensional histogram.

    :param xyz: A matrix of shape (n,3), where each row represents one point;
        points are assumed to lie in the plane x+y+z=1.
    :param bins: How many bins to use for the histogram in each dimension. If fewer bins
        are chosen, the coloring will be more homogeneous and change gradually. If more bins
        are chosen, the coloring will vary more.
    """
    xy = two_d_projection(xyz)
    density, xbins, ybins = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=40, range=None, density=True
    )
    xinds = np.minimum(np.searchsorted(xbins, xy[:, 0]), xbins.shape[0] - 2)
    yinds = np.minimum(np.searchsorted(ybins, xy[:, 1]), ybins.shape[0] - 2)
    return density[xinds, yinds]


def gaussian_density(xyz: Matrix) -> npt.NDArray[np.float_]:
    """
    Compute density estimates for the point cloud `xyz`. Assigns each point a
    nonnegative floating point number estimating the local density of the point cloud in that region
    using the scipy.stats.gaussian_kde function.

    :param xyz: A matrix of shape (n,3), where each row represents one point;
        points are assumed to lie in the plane x+y+z=1.
    """
    xy = two_d_projection(xyz)
    return gaussian_kde(xy.T)(xy.T)


def normalized_relative_dispersion(
    feature1_dispersion: DistanceMatrix,
    feature2_dispersion: DistanceMatrix,
    feature3_dispersion: DistanceMatrix,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Given three morphology spaces, normalize each one by scaling it to its
    average, compute the three pairwise distributions of relative differences,
    and rescale this data so that >90% of it lies in the unit simplex in R^3.
    """
    assert feature1_dispersion.shape == feature2_dispersion.shape
    assert feature2_dispersion.shape == feature3_dispersion.shape
    feature1_dispersion = squareform(feature1_dispersion, force="tovector")
    f1d_normal = feature1_dispersion / np.average(feature1_dispersion)

    feature2_dispersion = squareform(feature2_dispersion, force="tovector")
    f2d_normal = feature2_dispersion / np.average(feature2_dispersion)

    feature3_dispersion = squareform(feature3_dispersion, force="tovector")
    f3d_normal = feature3_dispersion / np.average(feature3_dispersion)

    d12 = f1d_normal - f2d_normal
    d23 = f2d_normal - f3d_normal
    d31 = f3d_normal - f1d_normal

    percent = 90
    percentile = np.percentile(norm(np.stack((d12, d23, d31), axis=1), axis=1), percent)
    normalizing_constant = MAGIC_NUMBER_1 / percentile
    d12 = d12 * normalizing_constant + 1 / 3
    d23 = d23 * normalizing_constant + 1 / 3
    d31 = d31 * normalizing_constant + 1 / 3

    assert np.allclose(d12 + d23 + d31, np.ones((f1d_normal.shape[0],), dtype=float))
    return d12, d23, d31


def ternary_distance(
    feature1_dispersion: DistanceMatrix,
    feature1_name: str,
    feature2_dispersion: DistanceMatrix,
    feature2_name: str,
    feature3_dispersion: DistanceMatrix,
    feature3_name: str,
    density_estimation: Literal["histogram"] | Literal["gaussian_kde"],
    bins: int,
    contour_lines: int = 4,
    **kwargs
):
    """
    Construct a ternary distance plot illustrating the relative variation in any one
    feature with respect to the others.

    :param bins: How many bins to use for the histogram in each dimension when
        estimating the gradient. If fewer bins are chosen, the coloring will be
        more homogeneous and change gradually. If more bins are chosen, the
        coloring will vary more.
    :param levels: How many contour lines to draw.
    """
    d12, d23, d31 = normalized_relative_dispersion(
        feature1_dispersion, feature2_dispersion, feature3_dispersion
    )
    xyz = np.stack((d12, d23, d31), axis=1)
    if density_estimation == "histogram":
        coloring = histogram_density(xyz, bins)
    else:
        coloring = gaussian_density(xyz)
    ax = plt.subplot(projection="ternary", ternary_sum=1.0)
    ax.set_tlabel(feature1_name + " - " + feature2_name)
    ax.set_llabel(feature2_name + " - " + feature3_name)
    ax.set_llabel(feature3_name + " - " + feature1_name)

    ax.grid()
    ax.scatter(d12, d23, d31, **kwargs)

    level_marks = np.linspace(np.min(coloring), np.max(coloring), contour_lines + 2)
    ax.tricontour(d12, d23, d31, coloring, level_marks)
    plt.show()
