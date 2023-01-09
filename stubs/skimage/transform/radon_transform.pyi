from _typeshed import Incomplete
from collections.abc import Generator

def radon(image, theta: Incomplete | None = ..., circle: bool = ..., *, preserve_range: bool = ...): ...
def iradon(radon_image, theta: Incomplete | None = ..., output_size: Incomplete | None = ..., filter_name: str = ..., interpolation: str = ..., circle: bool = ..., preserve_range: bool = ...): ...
def order_angles_golden_ratio(theta) -> Generator[Incomplete, None, None]: ...
def iradon_sart(radon_image, theta: Incomplete | None = ..., image: Incomplete | None = ..., projection_shifts: Incomplete | None = ..., clip: Incomplete | None = ..., relaxation: float = ..., dtype: Incomplete | None = ...): ...