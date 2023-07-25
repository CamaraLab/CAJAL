from .bregman import jcpot_barycenter as jcpot_barycenter, sinkhorn as sinkhorn
from .lp import emd as emd
from .optim import cg as cg, gcg as gcg
from .unbalanced import sinkhorn_unbalanced as sinkhorn_unbalanced
from .utils import BaseEstimator as BaseEstimator, check_params as check_params, cost_normalization as cost_normalization, dist as dist, dots as dots, kernel as kernel, label_normalization as label_normalization, laplacian as laplacian, unif as unif
from _typeshed import Incomplete

def sinkhorn_lpl1_mm(a, labels_a, b, M, reg, eta: float = ..., numItermax: int = ..., numInnerItermax: int = ..., stopInnerThr: float = ..., verbose: bool = ..., log: bool = ...): ...
def sinkhorn_l1l2_gl(a, labels_a, b, M, reg, eta: float = ..., numItermax: int = ..., numInnerItermax: int = ..., stopInnerThr: float = ..., verbose: bool = ..., log: bool = ...): ...
def joint_OT_mapping_linear(xs, xt, mu: int = ..., eta: float = ..., bias: bool = ..., verbose: bool = ..., verbose2: bool = ..., numItermax: int = ..., numInnerItermax: int = ..., stopInnerThr: float = ..., stopThr: float = ..., log: bool = ..., **kwargs): ...
def joint_OT_mapping_kernel(xs, xt, mu: int = ..., eta: float = ..., kerneltype: str = ..., sigma: int = ..., bias: bool = ..., verbose: bool = ..., verbose2: bool = ..., numItermax: int = ..., numInnerItermax: int = ..., stopInnerThr: float = ..., stopThr: float = ..., log: bool = ..., **kwargs): ...
def OT_mapping_linear(xs, xt, reg: float = ..., ws: Incomplete | None = ..., wt: Incomplete | None = ..., bias: bool = ..., log: bool = ...): ...
def emd_laplace(a, b, xs, xt, M, sim: str = ..., sim_param: Incomplete | None = ..., reg: str = ..., eta: int = ..., alpha: float = ..., numItermax: int = ..., stopThr: float = ..., numInnerItermax: int = ..., stopInnerThr: float = ..., log: bool = ..., verbose: bool = ...): ...
def distribution_estimation_uniform(X): ...

class BaseTransport(BaseEstimator):
    cost_: Incomplete
    limit_max: Incomplete
    mu_s: Incomplete
    mu_t: Incomplete
    xs_: Incomplete
    xt_: Incomplete
    def fit(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...
    def fit_transform(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...
    def transform(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ..., batch_size: int = ...): ...
    def transform_labels(self, ys: Incomplete | None = ...): ...
    def inverse_transform(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ..., batch_size: int = ...): ...
    def inverse_transform_labels(self, yt: Incomplete | None = ...): ...

class LinearTransport(BaseTransport):
    bias: Incomplete
    log: Incomplete
    reg: Incomplete
    distribution_estimation: Incomplete
    def __init__(self, reg: float = ..., bias: bool = ..., log: bool = ..., distribution_estimation=...) -> None: ...
    mu_s: Incomplete
    mu_t: Incomplete
    log_: Incomplete
    A1_: Incomplete
    B1_: Incomplete
    def fit(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...
    def transform(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ..., batch_size: int = ...): ...
    def inverse_transform(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ..., batch_size: int = ...): ...

class SinkhornTransport(BaseTransport):
    reg_e: Incomplete
    max_iter: Incomplete
    tol: Incomplete
    verbose: Incomplete
    log: Incomplete
    metric: Incomplete
    norm: Incomplete
    limit_max: Incomplete
    distribution_estimation: Incomplete
    out_of_sample_map: Incomplete
    def __init__(self, reg_e: float = ..., max_iter: int = ..., tol: float = ..., verbose: bool = ..., log: bool = ..., metric: str = ..., norm: Incomplete | None = ..., distribution_estimation=..., out_of_sample_map: str = ..., limit_max=...) -> None: ...
    coupling_: Incomplete
    log_: Incomplete
    def fit(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...

class EMDTransport(BaseTransport):
    metric: Incomplete
    norm: Incomplete
    log: Incomplete
    limit_max: Incomplete
    distribution_estimation: Incomplete
    out_of_sample_map: Incomplete
    max_iter: Incomplete
    def __init__(self, metric: str = ..., norm: Incomplete | None = ..., log: bool = ..., distribution_estimation=..., out_of_sample_map: str = ..., limit_max: int = ..., max_iter: int = ...) -> None: ...
    coupling_: Incomplete
    log_: Incomplete
    def fit(self, Xs, ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...

class SinkhornLpl1Transport(BaseTransport):
    reg_e: Incomplete
    reg_cl: Incomplete
    max_iter: Incomplete
    max_inner_iter: Incomplete
    tol: Incomplete
    log: Incomplete
    verbose: Incomplete
    metric: Incomplete
    norm: Incomplete
    distribution_estimation: Incomplete
    out_of_sample_map: Incomplete
    limit_max: Incomplete
    def __init__(self, reg_e: float = ..., reg_cl: float = ..., max_iter: int = ..., max_inner_iter: int = ..., log: bool = ..., tol: float = ..., verbose: bool = ..., metric: str = ..., norm: Incomplete | None = ..., distribution_estimation=..., out_of_sample_map: str = ..., limit_max=...) -> None: ...
    coupling_: Incomplete
    log_: Incomplete
    def fit(self, Xs, ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...

class EMDLaplaceTransport(BaseTransport):
    reg: Incomplete
    reg_lap: Incomplete
    reg_src: Incomplete
    metric: Incomplete
    norm: Incomplete
    similarity: Incomplete
    sim_param: Incomplete
    max_iter: Incomplete
    tol: Incomplete
    max_inner_iter: Incomplete
    inner_tol: Incomplete
    log: Incomplete
    verbose: Incomplete
    distribution_estimation: Incomplete
    out_of_sample_map: Incomplete
    def __init__(self, reg_type: str = ..., reg_lap: float = ..., reg_src: float = ..., metric: str = ..., norm: Incomplete | None = ..., similarity: str = ..., similarity_param: Incomplete | None = ..., max_iter: int = ..., tol: float = ..., max_inner_iter: int = ..., inner_tol: float = ..., log: bool = ..., verbose: bool = ..., distribution_estimation=..., out_of_sample_map: str = ...) -> None: ...
    coupling_: Incomplete
    log_: Incomplete
    def fit(self, Xs, ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...

class SinkhornL1l2Transport(BaseTransport):
    reg_e: Incomplete
    reg_cl: Incomplete
    max_iter: Incomplete
    max_inner_iter: Incomplete
    tol: Incomplete
    verbose: Incomplete
    log: Incomplete
    metric: Incomplete
    norm: Incomplete
    distribution_estimation: Incomplete
    out_of_sample_map: Incomplete
    limit_max: Incomplete
    def __init__(self, reg_e: float = ..., reg_cl: float = ..., max_iter: int = ..., max_inner_iter: int = ..., tol: float = ..., verbose: bool = ..., log: bool = ..., metric: str = ..., norm: Incomplete | None = ..., distribution_estimation=..., out_of_sample_map: str = ..., limit_max: int = ...) -> None: ...
    coupling_: Incomplete
    log_: Incomplete
    def fit(self, Xs, ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...

class MappingTransport(BaseEstimator):
    metric: Incomplete
    norm: Incomplete
    mu: Incomplete
    eta: Incomplete
    bias: Incomplete
    kernel: Incomplete
    sigma: Incomplete
    max_iter: Incomplete
    tol: Incomplete
    max_inner_iter: Incomplete
    inner_tol: Incomplete
    log: Incomplete
    verbose: Incomplete
    verbose2: Incomplete
    def __init__(self, mu: int = ..., eta: float = ..., bias: bool = ..., metric: str = ..., norm: Incomplete | None = ..., kernel: str = ..., sigma: int = ..., max_iter: int = ..., tol: float = ..., max_inner_iter: int = ..., inner_tol: float = ..., log: bool = ..., verbose: bool = ..., verbose2: bool = ...) -> None: ...
    xs_: Incomplete
    xt_: Incomplete
    log_: Incomplete
    def fit(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...
    def transform(self, Xs): ...

class UnbalancedSinkhornTransport(BaseTransport):
    reg_e: Incomplete
    reg_m: Incomplete
    method: Incomplete
    max_iter: Incomplete
    tol: Incomplete
    verbose: Incomplete
    log: Incomplete
    metric: Incomplete
    norm: Incomplete
    distribution_estimation: Incomplete
    out_of_sample_map: Incomplete
    limit_max: Incomplete
    def __init__(self, reg_e: float = ..., reg_m: float = ..., method: str = ..., max_iter: int = ..., tol: float = ..., verbose: bool = ..., log: bool = ..., metric: str = ..., norm: Incomplete | None = ..., distribution_estimation=..., out_of_sample_map: str = ..., limit_max: int = ...) -> None: ...
    coupling_: Incomplete
    log_: Incomplete
    def fit(self, Xs, ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...

class JCPOTTransport(BaseTransport):
    reg_e: Incomplete
    max_iter: Incomplete
    tol: Incomplete
    verbose: Incomplete
    log: Incomplete
    metric: Incomplete
    out_of_sample_map: Incomplete
    def __init__(self, reg_e: float = ..., max_iter: int = ..., tol: float = ..., verbose: bool = ..., log: bool = ..., metric: str = ..., out_of_sample_map: str = ...) -> None: ...
    xs_: Incomplete
    xt_: Incomplete
    coupling_: Incomplete
    proportions_: Incomplete
    log_: Incomplete
    def fit(self, Xs, ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ...): ...
    def transform(self, Xs: Incomplete | None = ..., ys: Incomplete | None = ..., Xt: Incomplete | None = ..., yt: Incomplete | None = ..., batch_size: int = ...): ...
    def transform_labels(self, ys: Incomplete | None = ...): ...
    def inverse_transform_labels(self, yt: Incomplete | None = ...): ...
