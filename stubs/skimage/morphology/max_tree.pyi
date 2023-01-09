from ..util import invert as invert
from _typeshed import Incomplete

unsigned_int_types: Incomplete
signed_int_types: Incomplete
signed_float_types: Incomplete

def max_tree(image, connectivity: int = ...): ...
def area_opening(image, area_threshold: int = ..., connectivity: int = ..., parent: Incomplete | None = ..., tree_traverser: Incomplete | None = ...): ...
def diameter_opening(image, diameter_threshold: int = ..., connectivity: int = ..., parent: Incomplete | None = ..., tree_traverser: Incomplete | None = ...): ...
def area_closing(image, area_threshold: int = ..., connectivity: int = ..., parent: Incomplete | None = ..., tree_traverser: Incomplete | None = ...): ...
def diameter_closing(image, diameter_threshold: int = ..., connectivity: int = ..., parent: Incomplete | None = ..., tree_traverser: Incomplete | None = ...): ...
def max_tree_local_maxima(image, connectivity: int = ..., parent: Incomplete | None = ..., tree_traverser: Incomplete | None = ...): ...