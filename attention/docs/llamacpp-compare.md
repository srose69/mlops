# FBLA against llama.cpp

Qwen3-8B on one GTX 1080 (8 GB, about 6.1 GB free beside a desktop), the tests of `llama-bench`: random
tokens, `pp512` processes 512 prompt tokens in one batch, `tg128` generates 128 tokens one at a time, `@ dD`
after a context of D tokens is already in the cache; tokens/s, mean ± standard deviation.

* **llama.cpp** master `970b51c` (2026-10-22), CUDA 12.0, `-DCMAKE_CUDA_ARCHITECTURES=61`, flash attention on
  (`-fa 1`), all layers on the GPU (`-ngl 99`). Q4_K_M: the official `Qwen/Qwen3-8B-GGUF`. NVFP4:
  `nvidia/Qwen3-8B-NVFP4` converted by llama.cpp's own `convert_hf_to_gguf.py` (the same weights, not
  re-quantized). The KV cache in f16, q8_0 (8.5 bits per element) or q4_0 (4.5 bits).
* **mlops**: [`bench/llama_bench.py`](../bench/llama_bench.py) on the decoder of
  [`samples/04_generate.py`](../samples/04_generate.py): `nvidia/Qwen3-8B-NVFP4` as it is on NVFP4EffMA (int8
  activations by DP4A in prefill, exact ones in decode), BF16 LM head, RoPE from `mlops.positional`, FBLA 4+2
  prefill and decode (6.25 bits per element and the BF16 sink / open windows).

| test | Q4_K_M, KV f16 | Q4_K_M, KV q8_0 | Q4_K_M, KV q4_0 | NVFP4, KV f16 | NVFP4, KV q8_0 | NVFP4, KV q4_0 | **mlops NVFP4 + FBLA** |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp512 | 604 | 567 | | 510 | 506 | | **778** |
| tg128 | 35.1 | 31.7 | | 33.2 | 32.1 | | **35.4** |
| pp512 @ d4096 | 386 | 366 | | 358 | 356 | | **585** |
| tg128 @ d4096 | 29.8 | 27.6 | | 30.5 | 28.1 | | **30.5** |
| pp512 @ d16384 | out of memory | 196 | | out of memory | 192 | | **361** |
| tg128 @ d16384 | out of memory | 20.6 | | out of memory | 20.0 | | **29.5** |
| pp512 @ d32768 | out of memory | out of memory | 127 | out of memory | out of memory | 122 | **253** |
| tg128 @ d32768 | out of memory | out of memory | 15.9 | out of memory | out of memory | 15.5 | **17.1** |

At 32K the FBLA cache (1.9 GB) and the weights do not fit together: 29 of the 36 layers stay on the GPU and 7
stream from pinned host memory, spread evenly through the depth so that each copy overlaps the compute of the
layers before it. Those copies (7 × 101 MiB at 12.6 GB/s, 56 ms per token) bound `tg128 @ d32768` at 17.8
tokens/s; it runs at 17.1. Everywhere else every layer is on the GPU.

Against llama.cpp on the same NVFP4 weights, with the KV cache that fits:

| context | prefill | decode |
|---:|---:|---:|
| 0 | 1.5× | 1.07× |
| 4K | 1.6× | 1.0× |
| 16K | 1.9× (q8_0) | 1.5× (q8_0) |
| 32K | 2.1× (q4_0) | 1.1× (q4_0) |

## Quality

Loss of Qwen3-8B on one window of 6,144 tokens of this repository's documentation, every position scored;
the reference is BF16 weights with exact FP32 attention. The BF16 model runs on the same GPU with its layers
streamed from host memory.

| weights | attention / KV cache | loss | delta |
|---|---|---:|---:|
| BF16 | FP32, exact (reference) | 2.7051 | |
| BF16 | PyTorch SDPA, BF16 | 2.7048 | -0.0003 |
| BF16 | **FBLA 4+2** | 2.7031 | -0.0020 |
| NVFP4 (ModelOpt) | FP32, exact | 2.7297 | +0.0246 |
| NVFP4 | PyTorch SDPA, BF16 | 2.7311 | +0.0260 |
| NVFP4 | **FBLA 4+2** | 2.7299 | +0.0248 |
| NVFP4, LM head also NVFP4 (round to nearest) | FBLA 4+2 | 2.7773 | +0.0722 |

On Qwen3-8B the FBLA cache is within the noise of one window (±0.002) of exact attention with either weights;
the loss of the NVFP4 model is its weight quantization (this 8B table was measured on this repository's
documentation and has not been re-run on WikiText). On Qwen3-0.6B (WikiText-2, 11 windows of 16K, 11,264 scored
tokens, [overview.md](overview.md#quality)) FBLA 3+3 (+0.0013), FBLA 4+2 (+0.0014) and q8_0 (+0.0011) are all
within the noise of BF16 (loss 2.784), while q4_0 costs +0.20 (7 %). FBLA stores 6.25 bits per element against
q8_0's 8.5; q4_0 is smaller (4.5 bits) and raises the loss by 7 % on that text.

## Reproduce

```sh
# llama.cpp (CUDA 12.0 needs a host compiler up to GCC 12)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
      -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-bench
python convert_hf_to_gguf.py Qwen3-8B-NVFP4 --outfile Qwen3-8B-NVFP4.gguf
./build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -d 0,4096,16384 -r 3

# mlops
python bench/llama_bench.py Qwen3-8B-NVFP4 -p 512 -n 128 -d 0,4096,16384,32768 -r 3
```

`bench/llama_bench.py` places the model per depth (the cache first, then the layers that fit) and prints the
streaming plan when layers stream; `-ngl` caps the layers on the GPU, `--nbuf` the streamed layers in flight.
