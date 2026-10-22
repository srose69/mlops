# Performance

Results per GPU are in [benches/table.md](../benches/table.md); each linked `benches/<gpu>.md` contains the
full 36-row table and the conditions of that run.

## Running the benchmark

```sh
bench/run_bench.sh
GPU=1 COOLING="open bench, 100 % fan" NOTES="..." bench/run_bench.sh
```

The script builds `bench_fbla`, raises the board power limit, allows the highest SM clock, locks the memory
clock where the driver supports it, records clocks/temperature/power/throttling, and restores the original
settings even on Ctrl+C. Raw output stays in `out/bench_<gpu>/`; `bench/report.py` writes or replaces that
GPU's `benches/<gpu>.md` file and its row in `benches/table.md`. It follows the E5M2EffMA/NVFP4EffMA runner
and needs `sudo` for `nvidia-smi`.

For an uncontrolled local run that does not change board settings:

```sh
make bench
```

## Methodology

- One cache is allocated for 1M tokens and decoded at 16K, 64K, 128K, 256K, 512K and 1M without repacking.
- The scenarios are both 3+3 and 4+2 at KV1/GQA2, KV1/GQA4 and KV8/GQA2. KV8/GQA2 is Qwen3-0.6B geometry.
- Four 2048³ FP32 products warm GPU Boost. Every result row then gets one untimed call, one second idle and
  enough timed calls for roughly one second of work on the GTX 1080 reference GPU. The reported time is the
  median; NVML samples the SM clock every 5 ms during the row.
- `MLOPS_BENCH_REPS` overrides every repeat count. `MLOPS_BENCH_COOL` and `MLOPS_BENCH_WARMUP` override the
  one-second pause and four-product warm-up.
- Synthetic scores are thresholded to the Qwen3-0.6B per-query refinement rates: 12.3% for 3+3 and 4.8% for
  4+2. The reported refinement rate is the union read once for all queries sharing a KV head.
- Traffic includes exact sink/open groups, metadata, the complete coarse K/V pass and coarse plus fine K/V
  reread for every selected correction. `effective GB/s` is that physical traffic divided by latency, not
  BF16-equivalent bandwidth.
- DRAM peak comes from memory clock and bus width. A 256 MiB device-to-device copy, counted as read + write,
  is the practical streaming baseline.
