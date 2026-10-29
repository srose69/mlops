# Results by GPU

One row per GPU, produced by `bench/run_bench.sh` (see [docs/performance.md](../docs/performance.md));
the linked file contains all 36 rows and the run conditions. FBLA columns use the Qwen3-0.6B
geometry: 8 KV heads, GQA 2, D = 128.

| GPU | CC | D2D copy | 3+3 16K | 4+2 16K | 3+3 1M | 4+2 1M | Date |
|---|---|---:|---:|---:|---:|---:|---|
| [NVIDIA GeForce GTX 1080](1080.md) | 6.1 | 282 GB/s | 0.106 ms, 190 GB/s | 0.108 ms, 193 GB/s | 4.534 ms, 274 GB/s | 4.623 ms, 280 GB/s | 2026-10-29 |
