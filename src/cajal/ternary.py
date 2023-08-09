import matplotlib.pyplot as plt
import mpltern
from typing import Optional, Any
import numpy as np

from scipy.spatial.distance import squareform
from .run_gw import DistanceMatrix


def normalize_to_unit_mean(A: DistanceMatrix):
    v = squareform(A, force="tovector")
    mean = np.average(v)
    return A / mean


def ternary_distance(
    feature1_dispersion: DistanceMatrix,
    feature1_name: str,
    feature2_dispersion: DistanceMatrix,
    feature2_name: str,
    feature3_dispersion: DistanceMatrix,
    feature3_name: str,
    **kwargs
):
    """
    Construct a ternary distance plot illustrating the relative variation in any one
    feature with respect to the others.

    """
    assert feature1_dispersion.shape == feature2_dispersion.shape
    assert feature2_dispersion.shape == feature3_dispersion.shape
    feature1_dispersion = squareform(feature1_dispersion, force="tovector")
    f1d_normal = feature1_dispersion / np.average(feature1_dispersion)

    feature2_dispersion = squareform(feature2_dispersion, force="tovector")
    f2d_normal = feature2_dispersion / np.average(feature2_dispersion)

    feature3_dispersion = squareform(feature3_dispersion, force="tovector")
    f3d_normal = feature3_dispersion / np.average(feature3_dispersion)

    d12 = f1d_normal - f2d_normal + 1 / 3
    d23 = f2d_normal - f3d_normal + 1 / 3
    d31 = f3d_normal - f1d_normal + 1 / 3
    assert np.allclose(d12 + d23 + d31, np.ones((f1d_normal.shape[0],), dtype=float))

    ax = plt.subplot(projection="ternary", ternary_sum=1.0)

    ax.set_tlabel(feature1_name + " - " + feature2_name)
    ax.set_llabel(feature2_name + " - " + feature3_name)
    ax.set_llabel(feature3_name + " - " + feature1_name)

    ax.grid()
    ax.scatter(d12, d23, d31, **kwargs)

    plt.show()
