from typing import Any

import datetime

class ABCTimestamp(datetime.datetime):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce_cython__(self, *args, **kwargs) -> Any: ...
    def __setstate_cython__(self, *args, **kwargs) -> Any: ...

def __pyx_unpickle_ABCTimestamp(*args, **kwargs) -> Any: ...
