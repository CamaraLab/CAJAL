from futhark_ffi.build import build

build_single_core = build("src/cajal/ugw/c/unbalanced_gw_c", "cajal/ugw/single_core")
build_multicore = build(
    "src/cajal/ugw/multicore/unbalanced_gw_multicore", "cajal/ugw/multicore"
)
build_opencl = build("src/cajal/ugw/opencl/unbalanced_gw_opencl", "cajal/ugw/opencl")
build_cuda = build("src/cajal/ugw/cuda/unbalanced_gw_cuda", "cajal/ugw/cuda")
