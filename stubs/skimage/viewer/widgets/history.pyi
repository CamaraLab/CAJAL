from .core import BaseWidget
from _typeshed import Incomplete

class OKCancelButtons(BaseWidget):
    ok: Incomplete
    cancel: Incomplete
    layout: Incomplete
    def __init__(self, button_width: int = ...) -> None: ...
    def update_original_image(self) -> None: ...
    def close_plugin(self) -> None: ...

class SaveButtons(BaseWidget):
    default_format: Incomplete
    name_label: Incomplete
    save_file: Incomplete
    save_stack: Incomplete
    layout: Incomplete
    def __init__(self, name: str = ..., default_format: str = ...) -> None: ...
    def save_to_stack(self) -> None: ...
    def save_to_file(self, filename: Incomplete | None = ...) -> None: ...
