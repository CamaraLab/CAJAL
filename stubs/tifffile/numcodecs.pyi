from _typeshed import Incomplete
from numcodecs.abc import Codec

class Tiff(Codec):
    codec_id: str
    key: Incomplete
    series: Incomplete
    level: Incomplete
    bigtiff: Incomplete
    byteorder: Incomplete
    imagej: Incomplete
    ome: Incomplete
    photometric: Incomplete
    planarconfig: Incomplete
    extrasamples: Incomplete
    volumetric: Incomplete
    tile: Incomplete
    rowsperstrip: Incomplete
    compression: Incomplete
    compressionargs: Incomplete
    predictor: Incomplete
    subsampling: Incomplete
    metadata: Incomplete
    extratags: Incomplete
    truncate: Incomplete
    maxworkers: Incomplete
    def __init__(self, key: Incomplete | None = ..., series: Incomplete | None = ..., level: Incomplete | None = ..., bigtiff: Incomplete | None = ..., byteorder: Incomplete | None = ..., imagej: bool = ..., ome: Incomplete | None = ..., photometric: Incomplete | None = ..., planarconfig: Incomplete | None = ..., extrasamples: Incomplete | None = ..., volumetric: Incomplete | None = ..., tile: Incomplete | None = ..., rowsperstrip: Incomplete | None = ..., compression: Incomplete | None = ..., compressionargs: Incomplete | None = ..., predictor: Incomplete | None = ..., subsampling: Incomplete | None = ..., metadata=..., extratags=..., truncate: bool = ..., maxworkers: Incomplete | None = ...) -> None: ...
    def encode(self, buf): ...
    def decode(self, buf, out: Incomplete | None = ...): ...

def register_codec(cls=..., codec_id: Incomplete | None = ...) -> None: ...
