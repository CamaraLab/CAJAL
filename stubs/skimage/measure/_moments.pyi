from .._shared.utils import check_nD as check_nD
from _typeshed import Incomplete

def moments_coords(coords, order: int = ...): ...
def moments_coords_central(coords, center: Incomplete | None = ..., order: int = ...): ...
def moments(image, order: int = ...): ...
def moments_central(image, center: Incomplete | None = ..., order: int = ..., **kwargs): ...
def moments_normalized(mu, order: int = ...): ...
def moments_hu(nu): ...
def centroid(image): ...
def inertia_tensor(image, mu: Incomplete | None = ...): ...
def inertia_tensor_eigvals(image, mu: Incomplete | None = ..., T: Incomplete | None = ...): ...
