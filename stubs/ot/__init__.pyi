from . import bregman as bregman, da as da, datasets as datasets, gromov as gromov, lp as lp, optim as optim, utils as utils
from .bregman import barycenter as barycenter, sinkhorn as sinkhorn, sinkhorn2 as sinkhorn2
from .da import sinkhorn_lpl1_mm as sinkhorn_lpl1_mm
from .lp import emd as emd, emd2 as emd2, emd2_1d as emd2_1d, emd_1d as emd_1d, wasserstein_1d as wasserstein_1d
from .unbalanced import barycenter_unbalanced as barycenter_unbalanced, sinkhorn_unbalanced as sinkhorn_unbalanced, sinkhorn_unbalanced2 as sinkhorn_unbalanced2
from .utils import dist as dist, tic as tic, toc as toc, toq as toq, unif as unif
