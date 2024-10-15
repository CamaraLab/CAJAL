su = scaling_unbalanced
ugw = unbalanced_gw

sc = $(ugw)_single_core
mc = $(ugw)_multicore
opencl = $(ugw)_opencl
cuda = $(ugw)_cuda

# check_compile: src/fused_gromov_wasserstein.fut src/scaling_unbalanced.fut \
# 	src/entropic_gw.fut
# 	futhark check src/fused_gromov_wasserstein.fut
# 	futhark check src/scaling_unbalanced.fut

.PHONY: check

check:
	for i in *.fut; do futhark check $$i; done

all: single_core/$(sc).fut multicore/$(mc).fut opencl/$(opencl).fut cuda/$(cuda).fut

single_core/$(sc).fut: $(ugw).fut
	cp $(ugw).fut $(sc).fut
	futhark-bindgen run $(sc).fut single_core/$(sc).ml
	rm $(sc).fut
	# mv $(sc).* single_core

multicore/$(mc).fut: $(ugw).fut
	cp $(ugw).fut $(mc).fut
	futhark-bindgen run --backend multicore $(mc).fut multicore/$(mc).ml
	rm $(mc).fut
	# mv $(mc).* multicore

opencl/$(opencl).fut: $(ugw).fut
	cp $(ugw).fut $(opencl).fut
	futhark-bindgen run --backend opencl $(opencl).fut opencl/$(opencl).ml
	rm $(opencl).fut
	# mv $(opencl).* opencl

cuda/$(cuda).fut: $(ugw).fut
	cp $(ugw).fut $(cuda).fut
	futhark-bindgen run --backend cuda $(cuda).fut cuda/$(cuda).ml
	rm $(cuda).fut
	# mv $(cuda).* cuda
