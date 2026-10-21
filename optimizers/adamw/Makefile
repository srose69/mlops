CUDA_HOME    ?= /usr/local/cuda
NVCC         ?= $(CUDA_HOME)/bin/nvcc
CCBIN        ?= g++-12
GPU          ?= 0

# Architectures: NVFP4EffMA's range (GPUs without native FP4). The tests and the bench run on GPU $(GPU),
# so by default they are built for it alone; ARCHS="61 89" or ARCHS=all for others. sm_89's FP8 cvt
# needs PTX ISA 8.1 (CUDA 12.1+): with an older nvcc sm_89 is dropped.
ALL_ARCHS    := 50 52 53 60 61 62 70 72 75 80 86 87 89 90
DETECTED     := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $(GPU) 2>/dev/null | head -1 | tr -d '.')
ARCHS        ?= $(if $(DETECTED),$(DETECTED),$(ALL_ARCHS))
ifeq ($(ARCHS),all)
  override ARCHS := $(ALL_ARCHS)
endif
NVCC_VER     := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
NVCC_PTX81   := $(shell set -- $(NVCC_VER); [ "$${1:-0}" -gt 12 ] || { [ "$${1:-0}" -eq 12 ] && [ "$${2:-0}" -ge 1 ]; } && echo 1)
UNSUPPORTED  := $(filter-out $(ALL_ARCHS),$(ARCHS))
ifneq ($(UNSUPPORTED),)
  $(error ARCHS: sm_$(UNSUPPORTED) outside NVFP4EffMA's range ($(ALL_ARCHS)): native FP4 or too old)
endif
ifeq ($(NVCC_PTX81)$(filter 89,$(ARCHS)),89)
  $(warning nvcc $(subst $(eval) ,.,$(NVCC_VER)) < 12.1: sm_89 dropped (FP8 cvt needs PTX ISA 8.1))
  override ARCHS := $(filter-out 89,$(ARCHS))
endif
GENCODE      := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
CXXFLAGS     := -O3 -std=c++17 $(GENCODE) -ccbin $(CCBIN) -Iinclude -Isrc

# NVFP4EffMA / E5M2EffMA checkouts: the QAT code includes their src/ (E2M1 / E4M3 / E5M2 codecs, swizzled
# scale addressing) and links NVFP4EffMA's static library (built with its own make) for the initial quantize.
NVFP4EFFMA   ?= ../../../NVFP4EffMA
E5M2EFFMA    ?= ../../../E5M2EffMA
NVFP4_FLAGS  := -I$(NVFP4EFFMA)/include -I$(NVFP4EFFMA)/src -I$(E5M2EFFMA)/include -I$(E5M2EFFMA)/src -L$(NVFP4EFFMA)/out -lnvfp4effma

OUT := out
RUN := CUDA_VISIBLE_DEVICES=$(GPU)

.PHONY: all test bench qat clean
all: $(OUT)/test_adamw $(OUT)/bench_adamw $(OUT)/test_adamw_qat

$(OUT):
	mkdir -p $(OUT)

$(OUT)/test_adamw: tests/test_adamw.cu src/adamw.cu include/mlops_adamw.h src/common.cuh | $(OUT)
	$(NVCC) $(CXXFLAGS) tests/test_adamw.cu src/adamw.cu -o $@

$(OUT)/bench_adamw: bench/bench_adamw.cu src/adamw.cu src/adamw_qat.cu src/qat_codec.cuh include/mlops_adamw.h include/mlops_adamw_qat.h src/common.cuh bench/bench_util.h | $(OUT)
	$(NVCC) $(CXXFLAGS) $(NVFP4_FLAGS) bench/bench_adamw.cu src/adamw.cu src/adamw_qat.cu -o $@ -lnvidia-ml

# -ffp-contract=off: the host side of the codec tests must not fuse what qat_codec.cuh keeps separate
$(OUT)/test_adamw_qat: tests/test_adamw_qat.cu src/adamw.cu src/adamw_qat.cu src/qat_codec.cuh include/mlops_adamw_qat.h include/mlops_adamw.h src/common.cuh | $(OUT)
	$(NVCC) $(CXXFLAGS) -Xcompiler -ffp-contract=off $(NVFP4_FLAGS) tests/test_adamw_qat.cu src/adamw.cu src/adamw_qat.cu -o $@

# the sm_89+ FP8 path (native cvt) on any GPU: the same wrappers through cuda_fp8.h, NVIDIA's model of
# those instructions (QAT_FP8_CVT_MODEL in qat_codec.cuh)
$(OUT)/test_adamw_qat_cvt: tests/test_adamw_qat.cu src/adamw.cu src/adamw_qat.cu src/qat_codec.cuh include/mlops_adamw_qat.h include/mlops_adamw.h src/common.cuh | $(OUT)
	$(NVCC) $(CXXFLAGS) -DQAT_FP8_CVT_MODEL -Xcompiler -ffp-contract=off $(NVFP4_FLAGS) tests/test_adamw_qat.cu src/adamw.cu src/adamw_qat.cu -o $@

test: $(OUT)/test_adamw
	$(RUN) ./$(OUT)/test_adamw

bench: $(OUT)/bench_adamw
	$(RUN) ./$(OUT)/bench_adamw

qat: $(OUT)/test_adamw_qat $(OUT)/test_adamw_qat_cvt
	$(RUN) ./$(OUT)/test_adamw_qat
	$(RUN) ./$(OUT)/test_adamw_qat_cvt

clean:
	rm -rf $(OUT)
