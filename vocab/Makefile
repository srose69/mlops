# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# mlops-vocab: everything along the vocabulary axis (embed/, lm_head/, sample/); targets lib, test, bench, install (described beside each).
# Toolchain, architectures and parallelism: ../build/arch.mk; no dependency besides the CUDA toolkit.
MLOPS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(MLOPS_ROOT)/build/arch.mk

VERSION := $(shell sed -n 's/^\#define MLOPS_VOCAB_VERSION "\(.*\)"/\1/p' include/mlops_vocab.h)
FLAGS    = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -std=c++17 -Xcompiler -ffp-contract=off,-fPIC -Iinclude -Icommon -Isample/src -Ilm_head/src -Iembed/src

SAMPLE_H = include/mlops_vocab.h include/mlops_sample.h include/mlops_vocab_items.h common/items.h sample/src/sample.cuh
HEAD_H   = include/mlops_vocab.h include/mlops_lm_head.h include/mlops_vocab_items.h common/items.h lm_head/src/lm_head.cuh lm_head/src/lm_head_mma.cuh
EMBED_H  = include/mlops_vocab.h include/mlops_embed.h include/mlops_vocab_items.h common/items.h embed/src/embed.cuh
SLICES   = 0 1 2 3 4 5 6  # the head's instances, compiled a slice an object (lm_head/src/instances.h)
OBJS     = sample lm_head embed items $(addprefix lm_head_inst_,$(SLICES))
HEADERS_ALL = $(SAMPLE_H) $(HEAD_H) $(EMBED_H) include/mlops_vocab_items.cuh
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)

all: lib test-bin

define SLICE  # $(1) = object dir, $(2) = architecture flags, $(3) = slice
$(1)/lm_head_inst_$(3).o: lm_head/src/lm_head_inst.cu lm_head/src/instances.h $(HEAD_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DMLOPS_LM_HEAD_SLICE=$(3) -c -o $$@ $$<
endef
define LIBRARY  # $(1) = object dir, $(2) = architecture flags
$(foreach k,$(SLICES),$(eval $(call SLICE,$(1),$(2),$(k))))
$(1)/sample.o: sample/src/sample.cu $(SAMPLE_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/lm_head.o: lm_head/src/lm_head.cu lm_head/src/instances.h $(HEAD_H) | $(1)
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
# libmlops_vocab.a for $(ARCHS) (default: every architecture sm_50..sm_90; SASS only)
lib: out/libmlops_vocab.a

# A program's object compiles beside the library's (they do not depend on each other); the link waits for both.
out/native/test_sample.o: sample/tests/test_sample.cu $(HEADERS_ALL) | out/native
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -c -o $@ $<
out/native/test_lm_head.o: lm_head/tests/test_lm_head.cu $(HEADERS_ALL) | out/native
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -c -o $@ $<
out/native/test_items.o: common/tests/test_items.cu $(HEADERS_ALL) | out/native
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -DVOCAB_INCLUDE='"$(CURDIR)/include"' -DCUDA_INCLUDE='"$(CUDA_HOME)/include"' -c -o $@ $<
out/native/test_embed.o: embed/tests/test_embed.cu $(HEADERS_ALL) | out/native
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -c -o $@ $<
out/test_sample: out/native/test_sample.o out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $^
out/test_lm_head: out/native/test_lm_head.o out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $^
out/test_items: out/native/test_items.o out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $^ -lnvrtc -lcuda
out/test_embed: out/native/test_embed.o out/native/libmlops_vocab.a
	$(NVCC) $(FLAGS) -arch=sm_$(TEST_ARCH) -o $@ $^
test-bin: out/test_sample out/test_lm_head out/test_embed out/test_items
# mlops_embed, mlops_lm_head and mlops_sample against host emulations (GPU $(GPU)); the programs run together, each one's report printed whole
MAKEFLAGS += -Otarget
test: run-test_embed run-test_items run-test_sample run-test_lm_head
run-test_embed: out/test_embed
	$(RUN) ./out/test_embed
run-test_items: out/test_items
	$(RUN) ./out/test_items
run-test_sample: out/test_sample
	$(RUN) ./out/test_sample
run-test_lm_head: out/test_lm_head
	$(RUN) ./out/test_lm_head
# the LM head's variants against a device-to-device copy; the sampler per configuration
bench: out/test_lm_head out/test_sample
	$(RUN) ./out/test_lm_head bench
	$(RUN) ./out/test_sample bench

PREFIX ?= /usr/local
# headers, static library and pkg-config file under PREFIX
install: $(OBJDIR)/libmlops_vocab.a
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 include/mlops_vocab.h include/mlops_sample.h include/mlops_lm_head.h include/mlops_embed.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(OBJDIR)/libmlops_vocab.a $(DESTDIR)$(PREFIX)/lib/
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: mlops-vocab\nDescription: sampling along the vocabulary axis for GPUs sm_50..sm_90\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -lmlops_vocab -L$(CUDA_HOME)/lib64 -lcudart -lstdc++\n' \
	       '$(PREFIX)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/mlops-vocab.pc

clean:
	rm -rf out

.PHONY: all lib test-bin test run-test_embed run-test_items run-test_sample run-test_lm_head bench install clean
