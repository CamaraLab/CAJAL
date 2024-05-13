from _typeshed import Incomplete
from skimage._shared.utils import deprecate_kwarg as deprecate_kwarg
from typing import List
import numpy.typing as npt
import numpy as np

def find_contours(image,
                  level: Incomplete | None = ...,
                  fully_connected: str = ...,
                  positive_orientation: str = ..., *,
                  mask: Incomplete | None = ...
                  ) -> List[npt.NDArray[np.float_]]: ...
