from cajal.ternary import two_d_projection
from scipy.spatial.distance import pdist

import numpy as np


def test_isometry():
    rng = np.random.default_rng()
    d1, d2, d3 = (rng.normal(loc=0, scale=1, size=(100, 3)) for i in range(3))
    d1_dm_vf, d2_dm_vf, d3_dm_vf = [pdist(d) for d in [d1, d2, d3]]
    d12 = d1_dm_vf - d2_dm_vf + (1 / 3)
    d23 = d2_dm_vf - d3_dm_vf + (1 / 3)
    d31 = d3_dm_vf - d1_dm_vf + (1 / 3)
    assert np.allclose(
        d12 + d23 + d31,
        np.ones(
            (4950),
        ),
    )
    xyz = np.stack((d12, d23, d31), axis=1)
    xy = two_d_projection(xyz)
    assert np.allclose(pdist(xyz), pdist(xy))
