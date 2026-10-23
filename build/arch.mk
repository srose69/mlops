# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# Shared by every package Makefile (include $(MLOPS_ROOT)/build/arch.mk): toolchain, target architectures,
# build parallelism and the dependencies' headers.
#
#   ARCHS="61 89"   architectures of the library (default: all of sm_50..sm_90, the GPUs without native
#                   FP4); ARCHS=native: only the local GPU's. A package that uses sm_89's FP8 cvt sets
#                   SM89_FP8 := 1 before including this file: it needs nvcc 12.1+ (PTX ISA 8.1), and with an
#                   older nvcc sm_89 is dropped
#   JOBS=n          parallel jobs (default: min(cores, half of MemAvailable / JOB_MB)); a multi-architecture
#                   object compiles its architectures in parallel (nvcc --threads) when there are more slots
#                   than objects
#   NVFP4EFFMA_DIR  NVFP4EffMA's include/ and src/ (nvfp4effma.h, nvfp4effma.cuh): by default the installed
#                   Python package ($(PYTHON) -c "import nvfp4effma"; get_include()), else ../NVFP4EffMA
#   E5M2EFFMA_DIR   E5M2EffMA's include/ and src/, the same way; needed only for architectures below sm_89
#   NVFP4EFFMA_LIB  libnvfp4effma.a for the programs that call mlops_*_init (tests, benchmarks): by default
#                   from NVFP4EFFMA_DIR or ../NVFP4EffMA, a checkout (out/native, out/) or an unpacked C archive
#                   (lib/)
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
# host compiler for nvcc; g++-12 if present (CUDA 12.0-12.3 do not accept GCC 13), else nvcc's default
CCBIN     ?= $(shell command -v g++-12 2>/dev/null)
PYTHON    ?= python3
GPU       ?= 0
RUN        = CUDA_VISIBLE_DEVICES=$(GPU)
DETECTED  := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $(GPU) 2>/dev/null | head -1 | sed -n 's/^\([0-9][0-9]*\)\.\([0-9][0-9]*\)$$/\1\2/p')
TEST_ARCH ?= $(if $(DETECTED),$(DETECTED),61)

ALL_ARCHS := 50 52 53 60 61 62 70 72 75 80 86 87 89 90
ARCHS     ?= $(ALL_ARCHS)
ifeq ($(ARCHS),native)
  override ARCHS := $(TEST_ARCH)
endif
ifeq ($(ARCHS),all)
  override ARCHS := $(ALL_ARCHS)
endif
UNSUPPORTED := $(filter-out $(ALL_ARCHS),$(ARCHS) $(TEST_ARCH))
ifneq ($(UNSUPPORTED),)
  $(error sm_$(UNSUPPORTED) outside sm_50..sm_90 (native FP4, or too old): ARCHS / TEST_ARCH)
endif
NVCC_VER  := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
NVCC_121  := $(shell set -- $(NVCC_VER); [ "$${1:-0}" -gt 12 ] || { [ "$${1:-0}" -eq 12 ] && [ "$${2:-0}" -ge 1 ]; } && echo 1)
ifeq ($(SM89_FP8):$(NVCC_121):$(filter 89,$(ARCHS)),1::89)
  $(warning nvcc $(subst $(eval) ,.,$(NVCC_VER)) < 12.1: sm_89 dropped (FP8 cvt needs PTX ISA 8.1))
  override ARCHS := $(filter-out 89,$(ARCHS))
endif
GENCODE    = $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
# architectures below sm_89 decode / encode FP8 with E5M2EffMA's codecs (native cvt from sm_89 on)
pre89      = $(filter 50 52 53 60 61 62 70 72 75 80 86 87,$(1))

# parallelism within half of the available RAM (JOB_MB: peak of one nvcc process)
JOB_MB    ?= 512
NPROC     := $(shell nproc 2>/dev/null || echo 4)
MEM_MB    := $(shell awk '/^MemAvailable:/ {print int($$2 / 1024)}' /proc/meminfo 2>/dev/null)
JOBS      ?= $(shell n=$$(( $(or $(MEM_MB),4096) / 2 / $(JOB_MB) )); [ $$n -gt $(NPROC) ] && n=$(NPROC); [ $$n -lt 1 ] && n=1; echo $$n)
MAKEFLAGS += -j$(JOBS)

# ---- dependencies ----------------------------------------------------------------------------------
# the directory of an installed Python package (without importing it: e5m2effma's import changes torch)
pkg_dir    = $(shell $(PYTHON) -c "import importlib.util as u; s = u.find_spec('$(1)'); print(s.submodule_search_locations[0] if s else '')" 2>/dev/null)
dep_dir    = $(firstword $(foreach d,$(call pkg_dir,$(1)) $(abspath $(MLOPS_ROOT)/../$(2)),$(if $(wildcard $(d)/include/$(1).h),$(d))))
ifeq ($(origin NVFP4EFFMA_DIR),undefined)  # looked up once, not at every expansion
  NVFP4EFFMA_DIR := $(call dep_dir,nvfp4effma,NVFP4EffMA)
endif
ifeq ($(origin E5M2EFFMA_DIR),undefined)
  E5M2EFFMA_DIR := $(call dep_dir,e5m2effma,E5M2EffMA)
endif
NVFP4EFFMA_LIB ?= $(firstword $(wildcard $(foreach d,$(NVFP4EFFMA_DIR) $(abspath $(MLOPS_ROOT)/../NVFP4EffMA),\
                    $(d)/out/native/libnvfp4effma.a $(d)/out/libnvfp4effma.a $(d)/lib/libnvfp4effma.a)))

# -I flags for a list of architectures; stops the build with what to install when a dependency is missing
define dep_inc
$(if $(wildcard $(NVFP4EFFMA_DIR)/src/nvfp4effma.cuh),,$(error NVFP4EffMA not found: install the nvfp4effma \
  package (its release wheel, or pip install ./NVFP4EffMA/binds/python --no-build-isolation) or set NVFP4EFFMA_DIR \
  to a directory with its include/ and src/))-I$(NVFP4EFFMA_DIR)/include -I$(NVFP4EFFMA_DIR)/src \
$(if $(call pre89,$(1)),$(if $(wildcard $(E5M2EFFMA_DIR)/src/e5m2effma.cuh),,$(error E5M2EffMA not found, needed \
  for sm_$(call pre89,$(1)) (below sm_89): install the e5m2effma package or set E5M2EFFMA_DIR))-I$(E5M2EFFMA_DIR)/include \
  -I$(E5M2EFFMA_DIR)/src)
endef
