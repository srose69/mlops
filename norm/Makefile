# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# mlops-norm: normalization layers (layer/).
#   make lib        libmlops_norm.a for $(ARCHS) (default: every architecture sm_50..sm_90; SASS only)
#   make test       mlops_norm against a host emulation, the INT8 and NVFP4 outputs also against nvfp4a_quantize, and its
#                   items through NVRTC (GPU $(GPU))
#   make bench      the norms against a device-to-device copy of the same bytes
#   make install    headers, static library and pkg-config file under PREFIX (default /usr/local)
# Toolchain, architectures and parallelism: ../build/arch.mk. NVFP4EffMA's headers (the installed package, or ../NVFP4EffMA
# or NVFP4EFFMA_DIR; the quantizer's primitives are imported from it) and the CUDA toolkit.
MLOPS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(MLOPS_ROOT)/build/arch.mk

VERSION := $(shell sed -n 's/^\#define MLOPS_NORM_VERSION "\(.*\)"/\1/p' include/mlops_norm.h)
NVFP4_INC = $(call dep_inc,)
FLAGS    = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -std=c++17 -Xcompiler -ffp-contract=off,-fPIC -Iinclude -Ilayer/src $(NVFP4_INC)

LAYER_H   = include/mlops_norm.h include/mlops_norm_layer.h include/mlops_norm_items.h include/mlops_norm_items.cuh layer/src/norm.cuh
OBJS      = norm
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)

NVFP4_LINK = $(or $(NVFP4EFFMA_LIB),$(error libnvfp4effma.a not found (the tests quantize with it as the reference): build NVFP4EffMA (make) or set NVFP4EFFMA_LIB))

all: lib test-bin

define LIBRARY  # $(1) = object dir, $(2) = architecture flags
$(1)/norm.o: layer/src/norm.cu $(LAYER_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/libmlops_norm.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
OBJDIR ?= out/obj
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS)))
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH)))
out/libmlops_norm.a: $(OBJDIR)/libmlops_norm.a
	cp $< $@
lib: out/libmlops_norm.a

out/test_norm: layer/tests/test_norm.cu out/native/libmlops_norm.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $< out/native/libmlops_norm.a $(NVFP4_LINK)
out/test_items: layer/tests/test_items.cu $(LAYER_H) out/native/libmlops_norm.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -DNORM_INCLUDE='"$(CURDIR)/include"' -DNVFP4_INCLUDE='"$(NVFP4EFFMA_DIR)/include"' -DNVFP4_SRC='"$(NVFP4EFFMA_DIR)/src"' -DCUDA_INCLUDE='"$(CUDA_HOME)/include"' -o $@ $< out/native/libmlops_norm.a -lnvrtc -lcuda
test-bin: out/test_norm out/test_items
test: out/test_norm out/test_items
	$(RUN) ./out/test_norm
	$(RUN) ./out/test_items
bench: out/test_norm
	$(RUN) ./out/test_norm bench

PREFIX ?= /usr/local
install: $(OBJDIR)/libmlops_norm.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/mlops_norm.h include/mlops_norm_layer.h include/mlops_norm_items.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libmlops_norm.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: mlops-norm\nDescription: RMSNorm and LayerNorm with residual add, gates and INT8 / NVFP4 quantization for GPUs sm_50..sm_90\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -lmlops_norm -L$(CUDA_HOME)/lib64 -lcudart -lstdc++\n' \
	       '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/mlops-norm.pc

clean:
	rm -rf out

.PHONY: all lib test-bin test bench install clean
