# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# mlops-optimizers: AdamW and the NVFP4 AdamW-QAT (adamw/).
#   make lib        libmlops_optimizers.a for $(ARCHS) (default: every architecture sm_50..sm_90; SASS only)
#   make test       mlops_adamw against its host emulation, every dtype pair (GPU $(GPU))
#   make qat        mlops_adamw_qat: codecs, one step against a double-precision reference, behaviour over many
#                   steps; on the E5M2EffMA path and on the sm_89+ cvt path (cuda_fp8.h's model of it)
#   make bench      AdamW / AdamW-QAT step time and memory traffic
#   make install    headers, static library and pkg-config file under PREFIX (default /usr/local)
# Toolchain, architectures, parallelism and the NVFP4EffMA / E5M2EffMA locations: ../build/arch.mk.
# Python / PyTorch bindings: python/ (pip install, or ../setup.sh optimizers).
MLOPS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(MLOPS_ROOT)/build/arch.mk

VERSION := $(shell sed -n 's/^\#define MLOPS_OPTIMIZERS_VERSION "\(.*\)"/\1/p' include/mlops_optimizers.h)
FLAGS    = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -std=c++17 -Xcompiler -ffp-contract=off,-fPIC -Iinclude -Iadamw/src

HDRS      = include/mlops_optimizers.h include/mlops_adamw.h include/mlops_adamw_qat.h adamw/src/common.cuh
ADAMW_H   = $(HDRS) adamw/src/adamw_kernels.cuh
QAT_H     = $(HDRS) adamw/src/adamw_qat_kernels.cuh adamw/src/qat_codec.cuh
# one object per kernel instance so the build runs in parallel: AdamW per (param, grad) dtype pair (codes
# BF16 0, FP16 1, FP32 2), the QAT step per gradient type (FP32 2, E4M3 3, E5M2 4); adamw_qat_init.o holds
# the only reference to NVFP4EffMA's library, so a static link takes it only into programs that call it
DT    = 0 1 2
INST  = $(foreach p,$(DT),$(foreach g,$(DT),adamw_inst_$(p)_$(g))) $(foreach g,2 3 4,adamw_qat_inst_$(g))
OBJS  = adamw adamw_qat adamw_qat_init $(INST)
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)
word_ = $(word $(1),$(subst _, ,$(2)))

all: lib test-bin bench-bin

# ---- library: $(ARCHS) (OBJDIR), the local architecture (out/native), and the local architecture on the
#      sm_89+ FP8 path through cuda_fp8.h's model of its cvt instructions (out/cvt, tests only) ----------
define LIBRARY  # $(1) = object dir, $(2) = flags (architectures, defines), $(3) = architectures
$(1)/adamw.o: adamw/src/adamw.cu $(ADAMW_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/adamw_inst_%.o: adamw/src/adamw_inst.cu $(ADAMW_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DMLOPS_PDT=$$(call word_,1,$$*) -DMLOPS_GDT=$$(call word_,2,$$*) -c -o $$@ $$<
$(1)/adamw_qat.o: adamw/src/adamw_qat.cu $(QAT_H) | $(1)
	$(NVCC) $(FLAGS) $(2) $$(call dep_inc,$(3)) -c -o $$@ $$<
$(1)/adamw_qat_init.o: adamw/src/adamw_qat_init.cu $(QAT_H) | $(1)
	$(NVCC) $(FLAGS) $(2) $$(call dep_inc,$(3)) -c -o $$@ $$<
$(1)/adamw_qat_inst_%.o: adamw/src/adamw_qat_inst.cu $(QAT_H) | $(1)
	$(NVCC) $(FLAGS) $(2) $$(call dep_inc,$(3)) -DMLOPS_QAT_GT=$$* -c -o $$@ $$<
$(1)/libmlops_optimizers.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
OBJDIR ?= out/obj
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS),$(ARCHS)))
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH),$(TEST_ARCH)))
$(eval $(call LIBRARY,out/cvt,-arch=sm_$(TEST_ARCH) -DQAT_FP8_CVT_MODEL,))
out/libmlops_optimizers.a: $(OBJDIR)/libmlops_optimizers.a
	cp $< $@
lib: out/libmlops_optimizers.a

# ---- tests and benchmark (local architecture) --------------------------------------------------------
# -ffp-contract=off (FLAGS): the host side of the codec tests must not fuse what qat_codec.cuh keeps separate
NATIVE_FLAGS = $(FLAGS) -arch=sm_$(TEST_ARCH)
NVFP4_LINK   = $(or $(NVFP4EFFMA_LIB),$(error libnvfp4effma.a not found (tests and bench call mlops_adamw_qat_init): \
               build NVFP4EffMA (make) or set NVFP4EFFMA_LIB))
out/test_adamw: adamw/tests/test_adamw.cu out/native/libmlops_optimizers.a
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< out/native/libmlops_optimizers.a
out/test_adamw_qat: adamw/tests/test_adamw_qat.cu out/native/libmlops_optimizers.a
	$(NVCC) $(NATIVE_FLAGS) $(call dep_inc,$(TEST_ARCH)) -o $@ $< out/native/libmlops_optimizers.a $(NVFP4_LINK)
out/test_adamw_qat_cvt: adamw/tests/test_adamw_qat.cu out/cvt/libmlops_optimizers.a
	$(NVCC) $(NATIVE_FLAGS) -DQAT_FP8_CVT_MODEL $(call dep_inc,) -o $@ $< out/cvt/libmlops_optimizers.a $(NVFP4_LINK)
out/bench_adamw: adamw/bench/bench_adamw.cu adamw/bench/bench_util.h out/native/libmlops_optimizers.a
	$(NVCC) $(NATIVE_FLAGS) $(call dep_inc,$(TEST_ARCH)) -o $@ $< out/native/libmlops_optimizers.a $(NVFP4_LINK) -lnvidia-ml

test-bin: out/test_adamw out/test_adamw_qat out/test_adamw_qat_cvt
bench-bin: out/bench_adamw

test: out/test_adamw
	$(RUN) ./out/test_adamw
qat: out/test_adamw_qat out/test_adamw_qat_cvt
	$(RUN) ./out/test_adamw_qat
	$(RUN) ./out/test_adamw_qat_cvt
bench: out/bench_adamw
	$(RUN) ./out/bench_adamw

# ---- install (C / C++ projects: pkg-config --cflags --libs mlops-optimizers) --------------------------
PREFIX ?= /usr/local
install: $(OBJDIR)/libmlops_optimizers.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/mlops_optimizers.h include/mlops_adamw.h include/mlops_adamw_qat.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libmlops_optimizers.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: mlops-optimizers\nDescription: AdamW and NVFP4 AdamW-QAT for GPUs without native FP4\nVersion: $(VERSION)\nRequires: nvfp4effma\nCflags: -I$${includedir}\nLibs: -L$${libdir} -lmlops_optimizers\n' \
	       '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/mlops-optimizers.pc

clean:
	rm -rf out

.PHONY: all lib test-bin bench-bin test qat bench install clean
