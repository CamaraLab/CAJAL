from _typeshed import Incomplete
from pandas._config.config import is_bool as is_bool, is_callable as is_callable, is_instance_factory as is_instance_factory, is_int as is_int, is_nonnegative_int as is_nonnegative_int, is_one_of_factory as is_one_of_factory, is_str as is_str, is_text as is_text
from pandas.util._exceptions import find_stack_level as find_stack_level

use_bottleneck_doc: str

def use_bottleneck_cb(key) -> None: ...

use_numexpr_doc: str

def use_numexpr_cb(key) -> None: ...

use_numba_doc: str

def use_numba_cb(key) -> None: ...

pc_precision_doc: str
pc_colspace_doc: str
pc_max_rows_doc: str
pc_min_rows_doc: str
pc_max_cols_doc: str
pc_max_categories_doc: str
pc_max_info_cols_doc: str
pc_nb_repr_h_doc: str
pc_pprint_nest_depth: str
pc_multi_sparse_doc: str
float_format_doc: str
max_colwidth_doc: str
colheader_justify_doc: str
pc_expand_repr_doc: str
pc_show_dimensions_doc: str
pc_east_asian_width_doc: str
pc_ambiguous_as_wide_doc: str
pc_latex_repr_doc: str
pc_table_schema_doc: str
pc_html_border_doc: str
pc_html_use_mathjax_doc: str
pc_max_dir_items: str
pc_width_doc: str
pc_chop_threshold_doc: str
pc_max_seq_items: str
pc_max_info_rows_doc: str
pc_large_repr_doc: str
pc_memory_usage_doc: str
pc_latex_escape: str
pc_latex_longtable: str
pc_latex_multicolumn: str
pc_latex_multicolumn_format: str
pc_latex_multirow: str

def table_schema_cb(key) -> None: ...
def is_terminal() -> bool: ...

max_cols: int
tc_sim_interactive_doc: str
use_inf_as_null_doc: str
use_inf_as_na_doc: str

def use_inf_as_na_cb(key) -> None: ...

data_manager_doc: str
copy_on_write_doc: str
chained_assignment: str
string_storage_doc: str
reader_engine_doc: str
writer_engine_doc: str
parquet_engine_doc: str
sql_engine_doc: str
plotting_backend_doc: str

def register_plotting_backend_cb(key) -> None: ...

register_converter_doc: str

def register_converter_cb(key) -> None: ...

styler_sparse_index_doc: str
styler_sparse_columns_doc: str
styler_render_repr: str
styler_max_elements: str
styler_max_rows: str
styler_max_columns: str
styler_precision: str
styler_decimal: str
styler_thousands: str
styler_na_rep: str
styler_escape: str
styler_formatter: str
styler_multirow_align: str
styler_multicol_align: str
styler_hrules: str
styler_environment: str
styler_encoding: str
styler_mathjax: str
val_mca: Incomplete
