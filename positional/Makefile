# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# mlops-positional: positional encodings, RoPE (rope/).
#   make lib        libmlops_positional.a for $(ARCHS) (default: every architecture sm_50..sm_90; SASS only)
#   make test       mlops_rope against a double-precision reference and Hugging Face's arithmetic (GPU $(GPU))
#   make install    headers, static library and pkg-config file under PREFIX (default /usr/local)
# Toolchain, architectures and parallelism: ../build/arch.mk. No dependency besides the CUDA toolkit.
# Python / PyTorch bindings: python/ (pip install, or ../setup.sh positional); benchmark: bench/bench_rope.py.
MLOPS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(MLOPS_ROOT)/build/arch.mk

VERSION := $(shell sed -n 's/^\#define MLOPS_POSITIONAL_VERSION "\(.*\)"/\1/p' include/mlops_positional.h)
FLAGS    = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -std=c++17 -Xcompiler -ffp-contract=off,-fPIC -Iinclude -Irope/src

ROPE_H = include/mlops_positional.h include/mlops_rope.h rope/src/rope.cuh rope/src/rope_kernels.cuh
# one object per dtype of the RoPE kernel instances (BF16 0, FP16 1, FP32 2), so the build runs in parallel
OBJS   = rope $(foreach t,0 1 2,rope_inst_$(t))
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)

all: lib test-bin

define LIBRARY  # $(1) = object dir, $(2) = architecture flags
$(1)/rope.o: rope/src/rope.cu $(ROPE_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/rope_inst_%.o: rope/src/rope_inst.cu $(ROPE_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DMLOPS_ROPE_DT=$$* -c -o $$@ $$<
$(1)/libmlops_positional.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
OBJDIR ?= out/obj
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS)))
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH)))
out/libmlops_positional.a: $(OBJDIR)/libmlops_positional.a
	cp $< $@
lib: out/libmlops_positional.a

# need mlops-positional (and, for bench, transformers) installed in $(PYTHON)
bench:
	$(RUN) $(PYTHON) bench/bench_rope.py
accuracy:
	$(RUN) $(PYTHON) bench/bench_rope_accuracy.py

out/test_rope: rope/tests/test_rope.cu out/native/libmlops_positional.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $< out/native/libmlops_positional.a
test-bin: out/test_rope
test: out/test_rope
	$(RUN) ./out/test_rope

PREFIX ?= /usr/local
install: $(OBJDIR)/libmlops_positional.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/mlops_positional.h include/mlops_rope.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libmlops_positional.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: mlops-positional\nDescription: positional encodings (RoPE) for GPUs sm_50..sm_90\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -lmlops_positional -L$(CUDA_HOME)/lib64 -lcudart -lstdc++\n' \
	       '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/mlops-positional.pc

clean:
	rm -rf out

.PHONY: all lib test-bin test bench accuracy install clean
