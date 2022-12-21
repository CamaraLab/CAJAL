from pandas._typing import F as F
from pandas.core.base import PandasObject as PandasObject
from pandas.core.computation.common import result_type_many as result_type_many
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.indexes.api import Index as Index
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util._exceptions import find_stack_level as find_stack_level

def align_terms(terms): ...
def reconstruct_object(typ, obj, axes, dtype): ...
