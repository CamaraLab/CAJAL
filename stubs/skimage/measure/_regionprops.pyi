from ._regionprops_utils import euler_number as euler_number, perimeter as perimeter, perimeter_crofton as perimeter_crofton
from _typeshed import Incomplete
from typing import Optional, Iterable, Callable, Tuple, List
from numpy.typing import NDArray

class RegionProperties:
    label: Incomplete
    slice: Incomplete
    def __init__(self, slice, label, label_image, intensity_image, cache_active, *, extra_properties: Incomplete | None = ...) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, name, value) -> None: ...
    @property
    def area(self): ...
    @property
    def bbox(self): ...
    @property
    def area_bbox(self): ...
    @property
    def centroid(self): ...
    @property
    def area_convex(self): ...
    @property
    def image_convex(self): ...
    @property
    def coords(self): ...
    @property
    def eccentricity(self): ...
    @property
    def equivalent_diameter_area(self): ...
    @property
    def euler_number(self): ...
    @property
    def extent(self): ...
    @property
    def feret_diameter_max(self): ...
    @property
    def area_filled(self): ...
    @property
    def image_filled(self): ...
    @property
    def image(self): ...
    @property
    def inertia_tensor(self): ...
    @property
    def inertia_tensor_eigvals(self): ...
    @property
    def image_intensity(self): ...
    @property
    def centroid_local(self): ...
    @property
    def intensity_max(self): ...
    @property
    def intensity_mean(self): ...
    @property
    def intensity_min(self): ...
    @property
    def axis_major_length(self): ...
    @property
    def axis_minor_length(self): ...
    @property
    def moments(self): ...
    @property
    def moments_central(self): ...
    @property
    def moments_hu(self): ...
    @property
    def moments_normalized(self): ...
    @property
    def orientation(self): ...
    @property
    def perimeter(self): ...
    @property
    def perimeter_crofton(self): ...
    @property
    def solidity(self): ...
    @property
    def centroid_weighted(self): ...
    @property
    def centroid_weighted_local(self): ...
    @property
    def moments_weighted(self): ...
    @property
    def moments_weighted_central(self): ...
    @property
    def moments_weighted_hu(self): ...
    @property
    def moments_weighted_normalized(self): ...
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def __eq__(self, other): ...

def regionprops(label_image, intensity_image: Incomplete | None = ..., cache: bool = ..., coordinates: Incomplete | None = ..., *, extra_properties: Incomplete | None = ...): ...
def regionprops_table(label_image : NDArray,
                      intensity_image: Optional[NDArray],
                      properties : Optional[Tuple[str,...]|List[str]] =('label', 'bbox'),
                      *,
                      cache: Optional[bool] =True,
                      separator: Optional[str] ='-',
                      extra_properties: Optional[Iterable[Callable]]=None) -> dict: ...