from pandas import DataFrame as DataFrame
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.inference import is_list_like as is_list_like
from pandas.io.common import stringify_path as stringify_path
from pathlib import Path
from typing import Sequence

def read_spss(path: Union[str, Path], usecols: Union[Sequence[str], None] = ..., convert_categoricals: bool = ...) -> DataFrame: ...
