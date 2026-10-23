# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# mlops-attention: FBLA compressed-KV FlashAttention prefill and decode (fbla/).
#   make lib        libmlops_attention.a for $(ARCHS) (default: every DP4A architecture sm_61..sm_90)
#   make test       both formats against a float64 model; GQA 1/2/4, capacity, 384 .. 256K tokens (GPU $(GPU))
#   make bench      local run: 3+3 / 4+2 at 16K .. 1M; KV heads 1/8, GQA 2/4, traffic and NVML clocks
#                   (bench/run_bench.sh: controlled board run -> benches/<gpu>.md and benches/table.md)
#   make install    headers, static library and pkg-config file under PREFIX (default /usr/local)
# Toolchain, architectures and parallelism: ../build/arch.mk. No dependency besides the CUDA toolkit;
# the benchmark additionally links NVML. Python / PyTorch bindings: python/ (../setup.sh attention).
MLOPS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(MLOPS_ROOT)/build/arch.mk

# FBLA's integer matrix products use DP4A (introduced in sm_61).
override ARCHS := $(filter 61 62 70 72 75 80 86 87 89 90,$(ARCHS))
ifeq ($(ARCHS),)
  $(error mlops-attention needs an sm_61 or newer target (DP4A))
endif
ifeq ($(filter $(TEST_ARCH),61 62 70 72 75 80 86 87 89 90),)
  $(error mlops-attention tests need TEST_ARCH=61 or newer (DP4A))
endif

VERSION := $(shell sed -n 's/^#define MLOPS_ATTENTION_VERSION "\(.*\)"/\1/p' include/mlops_attention.h)
FLAGS   = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -std=c++17 -Xcompiler -ffp-contract=off,-fPIC -Iinclude -Ifbla/src
HDRS    = include/mlops_attention.h include/mlops_fbla.h fbla/src/common.cuh fbla/src/fbla_kernels.cuh \
          fbla/src/fbla_prefill.cuh
OBJS    = fbla fbla_inst_3p3 fbla_inst_4p2
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)

all: lib test-bin bench-bin

define LIBRARY  # $(1) = object dir, $(2) = architecture flags
$(1)/fbla.o: fbla/src/fbla.cu $(HDRS) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/fbla_inst_3p3.o: fbla/src/fbla_inst.cu $(HDRS) | $(1)
	$(NVCC) $(FLAGS) $(2) -DMLOPS_FBLA_CB=3 -DMLOPS_FBLA_FB=3 -c -o $$@ $$<
$(1)/fbla_inst_4p2.o: fbla/src/fbla_inst.cu $(HDRS) | $(1)
	$(NVCC) $(FLAGS) $(2) -DMLOPS_FBLA_CB=4 -DMLOPS_FBLA_FB=2 -c -o $$@ $$<
$(1)/libmlops_attention.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
OBJDIR ?= out/obj
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS)))
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH)))
out/libmlops_attention.a: $(OBJDIR)/libmlops_attention.a
	cp $< $@
lib: out/libmlops_attention.a

NATIVE_FLAGS = $(FLAGS) -arch=sm_$(TEST_ARCH)
# ---- tests and benchmark (local architecture) ----------------------------------------------------
out/test_fbla: fbla/tests/test_fbla.cu out/native/libmlops_attention.a
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< out/native/libmlops_attention.a
out/bench_fbla: fbla/bench/bench_fbla.cu fbla/bench/bench_util.h out/native/libmlops_attention.a
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< out/native/libmlops_attention.a -lnvidia-ml
test-bin: out/test_fbla
bench-bin: out/bench_fbla
test: out/test_fbla
	$(RUN) ./out/test_fbla
bench: out/bench_fbla
	$(RUN) ./out/bench_fbla

# ---- install (C / C++ projects: pkg-config --cflags --libs mlops-attention) ----------------------
PREFIX ?= /usr/local
install: $(OBJDIR)/libmlops_attention.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/mlops_attention.h include/mlops_fbla.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libmlops_attention.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: mlops-attention\nDescription: FBLA compressed-KV FlashAttention prefill and decode\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -lmlops_attention -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -lm\n' '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/mlops-attention.pc

clean:
	rm -rf out

.PHONY: all lib test-bin bench-bin test bench install clean
