from _typeshed import Incomplete
from pandas import DataFrame as DataFrame
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from typing import Any

def read_gbq(query: str, project_id: Union[str, None] = ..., index_col: Union[str, None] = ..., col_order: Union[list[str], None] = ..., reauth: bool = ..., auth_local_webserver: bool = ..., dialect: Union[str, None] = ..., location: Union[str, None] = ..., configuration: Union[dict[str, Any], None] = ..., credentials: Incomplete | None = ..., use_bqstorage_api: Union[bool, None] = ..., max_results: Union[int, None] = ..., progress_bar_type: Union[str, None] = ...) -> DataFrame: ...
def to_gbq(dataframe: DataFrame, destination_table: str, project_id: Union[str, None] = ..., chunksize: Union[int, None] = ..., reauth: bool = ..., if_exists: str = ..., auth_local_webserver: bool = ..., table_schema: Union[list[dict[str, str]], None] = ..., location: Union[str, None] = ..., progress_bar: bool = ..., credentials: Incomplete | None = ...) -> None: ...
