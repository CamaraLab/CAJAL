import sys
from typing import NewType
import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 10):
    from typing import TypeAlias

Distribution: TypeAlias = npt.NDArray[np.float64]
# A DistanceMatrix is a square symmetric matrix with zeros along the diagonal
# and nonnegative entries.
DistanceMatrix: TypeAlias = npt.NDArray[np.float64]
Matrix = NewType("Matrix", npt.NDArray[np.float64])
# An Array is a one-dimensional matrix.
Array = NewType("Array", npt.NDArray[np.float64])
# A MetricMeasureSpace is a pair consisting of a DistanceMatrix `A` and a Distribution `a`
# such that `A.shape[0]==`a.shape[0]`.
MetricMeasureSpace: TypeAlias = tuple[DistanceMatrix, Distribution]
