.SUFFIXES:
# This line helps get rid of weird bugs.

# all_futhark_files := $(*.fut)
ugw = unbalanced_gw

# sc = single core
sc = c
mc = multicore
opencl = opencl
cuda = cuda

.PHONY: check

check:
	for i in *.fut; do futhark check $$i; done

.PHONY: all ml

all: $(sc)/$(ugw).c $(mc)/$(ugw).c $(opencl)/$(ugw).c $(cuda)/$(ugw).c
ml: $(sc)/$(ugw).ml $(mc)/$(ugw).ml $(opencl)/$(ugw).ml $(cuda)/$(ugw).ml

%.c: *.fut
	futhark $(@D) $(ugw).fut --library -o $*

%.ml: *.fut
	futhark-bindgen run $(ugw).fut $@ --backend $(@D)