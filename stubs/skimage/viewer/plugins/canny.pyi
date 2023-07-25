from ...feature import canny as canny
from ..widgets import ComboBox as ComboBox, Slider as Slider
from .overlayplugin import OverlayPlugin as OverlayPlugin

class CannyPlugin(OverlayPlugin):
    name: str
    def __init__(self, *args, **kwargs) -> None: ...
    def attach(self, image_viewer) -> None: ...
