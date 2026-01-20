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

all: $(sc)/$(ugw)_$(sc).c $(mc)/$(ugw)_$(mc).c $(opencl)/$(ugw)_$(opencl).c $(cuda)/$(ugw)_$(cuda).c
ml: $(sc)/$(ugw)_$(sc).ml $(mc)/$(ugw)_$(mc).ml $(opencl)/$(ugw)_$(opencl).ml $(cuda)/$(ugw)_$(cuda).ml

%.c: *.fut
	futhark $(@D) $(ugw).fut --library -o $*

%.ml: *.fut
	futhark-bindgen run $(ugw).fut $@ --backend $(@D);\
	rm -f $*.c;\
	rm -f $*.json;\
	rm -f $*.h;\
	mv $(dir $@)$(ugw).c $*.c;\
	mv $(dir $@)$(ugw).json $*.json;\
	mv $(dir $@)$(ugw).h $*.h

