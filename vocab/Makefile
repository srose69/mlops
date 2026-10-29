# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# mlops-vocab: everything along the vocabulary axis (embed/, lm_head/, sample/).
#   make lib        libmlops_vocab.a for $(ARCHS) (default: every architecture sm_50..sm_90; SASS only)
#   make test       mlops_embed, mlops_lm_head and mlops_sample against host emulations (GPU $(GPU))
#   make bench      the LM head's variants against a device-to-device copy; the sampler per configuration
#   make install    headers, static library and pkg-config file under PREFIX (default /usr/local)
# Toolchain, architectures and parallelism: ../build/arch.mk. No dependency besides the CUDA toolkit.
MLOPS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(MLOPS_ROOT)/build/arch.mk

VERSION := $(shell sed -n 's/^\#define MLOPS_VOCAB_VERSION "\(.*\)"/\1/p' include/mlops_vocab.h)
FLAGS    = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -std=c++17 -Xcompiler -ffp-contract=off,-fPIC -Iinclude -Icommon -Isample/src -Ilm_head/src -Iembed/src

SAMPLE_H = include/mlops_vocab.h include/mlops_sample.h include/mlops_vocab_items.h common/items.h sample/src/sample.cuh
HEAD_H   = include/mlops_vocab.h include/mlops_lm_head.h include/mlops_vocab_items.h common/items.h lm_head/src/lm_head.cuh lm_head/src/lm_head_mma.cuh
EMBED_H  = include/mlops_vocab.h include/mlops_embed.h include/mlops_vocab_items.h common/items.h embed/src/embed.cuh
OBJS     = sample lm_head embed items
HEADERS_ALL = $(SAMPLE_H) $(HEAD_H) $(EMBED_H) include/mlops_vocab_items.cuh
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)

all: lib test-bin

define LIBRARY  # $(1) = object dir, $(2) = architecture flags
$(1)/sample.o: sample/src/sample.cu $(SAMPLE_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/lm_head.o: lm_head/src/lm_head.cu $(HEAD_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/embed.o: embed/src/embed.cu $(EMBED_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/items.o: common/items.cu common/items.h include/mlops_vocab_items.h | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/libmlops_vocab.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
OBJDIR ?= out/obj
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS)))
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH)))
out/libmlops_vocab.a: $(OBJDIR)/libmlops_vocab.a
	cp $< $@
lib: out/libmlops_vocab.a

out/test_sample: sample/tests/test_sample.cu out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $< out/native/libmlops_vocab.a
out/test_lm_head: lm_head/tests/test_lm_head.cu out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $< out/native/libmlops_vocab.a
out/test_items: common/tests/test_items.cu $(HEADERS_ALL) out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -DVOCAB_INCLUDE='"$(CURDIR)/include"' -DCUDA_INCLUDE='"$(CUDA_HOME)/include"' -o $@ $< out/native/libmlops_vocab.a -lnvrtc -lcuda
out/test_embed: embed/tests/test_embed.cu out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $< out/native/libmlops_vocab.a
test-bin: out/test_sample out/test_lm_head out/test_embed out/test_items
test: out/test_sample out/test_lm_head out/test_embed out/test_items
	$(RUN) ./out/test_embed
	$(RUN) ./out/test_items
	$(RUN) ./out/test_sample
	$(RUN) ./out/test_lm_head
bench: out/test_lm_head out/test_sample
	$(RUN) ./out/test_lm_head bench
	$(RUN) ./out/test_sample bench

PREFIX ?= /usr/local
install: $(OBJDIR)/libmlops_vocab.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/mlops_vocab.h include/mlops_sample.h include/mlops_lm_head.h include/mlops_embed.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libmlops_vocab.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: mlops-vocab\nDescription: sampling along the vocabulary axis for GPUs sm_50..sm_90\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -lmlops_vocab -L$(CUDA_HOME)/lib64 -lcudart -lstdc++\n' \
	       '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/mlops-vocab.pc

clean:
	rm -rf out

.PHONY: all lib test-bin test bench install clean
