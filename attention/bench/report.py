#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Turn a bench/run_bench.sh result directory into benches/<gpu>.md and benches/table.md."""
import csv
import os
import re
import sys


def slug(name):
    words = [w for w in re.split(r"\s+", name.strip())
             if w.lower() not in ("nvidia", "geforce", "tesla", "quadro", "gtx", "rtx")]
    return re.sub(r"[^a-z0-9]+", "-", " ".join(words).lower()).strip("-") or "gpu"


def read_env(path):
    env = {}
    for line in open(path):
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def bench_rows(path):
    rows, head = [], {}
    for line in open(path):
        h = re.search(r"^(.*?) sm_(\d+), (\d+) SMs", line)
        if h:
            head.update(gpu=h.group(1), cc=f"{h.group(2)[0]}.{h.group(2)[1:]}", sms=h.group(3))
        h = re.search(r"DRAM peak ([\d.]+) GB/s \((\d+)-bit bus\), D2D copy ([\d.]+) GB/s", line)
        if h:
            head.update(dram=float(h.group(1)), bus=h.group(2), copy=float(h.group(3)))
        p = line.split()
        if len(p) != 11 or p[0] not in ("3+3", "4+2"):
            continue
        rows.append(dict(fmt=p[0], kv=int(p[1]), gqa=int(p[2]), context=int(p[3]), us=float(p[4]),
                         gbs=float(p[5]), peak=float(p[6].rstrip("%")), read=float(p[7]), bf16=float(p[8]),
                         refined=float(p[9].rstrip("%")), mhz=float(p[10])))
    return rows, head


def clocks(path):
    sm, mem, temp, watt, reasons = [], [], [], [], {}
    if not os.path.exists(path):
        return None
    for r in csv.reader(open(path), skipinitialspace=True):
        if len(r) < 6 or int(r[5], 16) & 0x1:
            continue
        try:
            sm.append(float(r[1].split()[0])); mem.append(float(r[2].split()[0]))
            temp.append(float(r[3])); watt.append(float(r[4].split()[0]))
        except ValueError:
            continue
        reasons[r[5]] = reasons.get(r[5], 0) + 1
    if not sm:
        return None
    capped = sum(n for k, n in reasons.items() if int(k, 16) & 0x4)
    thermal = sum(n for k, n in reasons.items() if int(k, 16) & 0x60)
    s = sorted(sm)
    return dict(sm=(s[len(s) // 20], s[len(s) // 2], s[-1]), mem=sorted(set(mem)),
                temp=(min(temp), max(temp)), watt=max(watt), n=len(sm), capped=capped, thermal=thermal)


def fmt_clock(c):
    if not c:
        return "—"
    return (f"SM median {c['sm'][1]:.0f} MHz ({c['sm'][0]:.0f}–{c['sm'][2]:.0f}), memory "
            f"{'/'.join(f'{x:.0f}' for x in c['mem'])} MHz, {c['temp'][0]:.0f}–{c['temp'][1]:.0f} °C, "
            f"up to {c['watt']:.0f} W; power-capped in {c['capped']} of {c['n']} samples, "
            f"thermal slowdown in {c['thermal']}")


def main(res, out_dir):
    env = read_env(os.path.join(res, "env.txt"))
    rows, head = bench_rows(os.path.join(res, "bench_fbla.txt"))
    if not rows:
        raise SystemExit("no FBLA rows in bench_fbla.txt")
    gpu = env.get("name", head.get("gpu", "unknown GPU")); key = env.get("file") or slug(gpu)
    if head.get("sms") and not env.get("sms"):
        env["sms"] = head["sms"]
    if head.get("bus") and "bus" not in env.get("memory", ""):
        env["memory"] = f"{env.get('memory', '?')}, {head['bus']}-bit bus"
    clock = clocks(os.path.join(res, "clocks_bench_fbla.csv"))
    L = [f"# {gpu}", "", "Results of `bench/run_bench.sh`. Methodology: "
         "[docs/performance.md](../docs/performance.md).", "", "| | |", "|---|---|"]
    for k, label in (("name", "GPU"), ("compute_cap", "Compute capability"), ("sms", "SMs"),
                     ("memory", "Memory"), ("driver", "Driver"), ("cuda", "CUDA toolkit"),
                     ("library", "mlops-attention"), ("date", "Date"), ("power_limit", "Power limit"),
                     ("clock_lock", "Clock settings"), ("warmup", "Warm-up"), ("row_cool", "Idle before row"),
                     ("repeats", "Timed calls"), ("cooling", "Cooling"), ("notes", "Notes")):
        if env.get(k):
            L.append(f"| {label} | {env[k]} |")
    if clock:
        L.append(f"| During bench_fbla | {fmt_clock(clock)} |")
    L += ["", f"DRAM peak {head.get('dram', 0):.0f} GB/s; device-to-device copy "
          f"{head.get('copy', 0):.0f} GB/s (read + write).", "",
          "| format | KV heads | GQA | context | latency, us | effective GB/s | DRAM peak | read, MB | BF16 KV, MB | refined | SM MHz |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['fmt']} | {r['kv']} | {r['gqa']} | {r['context']} | {r['us']:.1f} | "
                 f"{r['gbs']:.1f} | {r['peak']:.1f}% | {r['read']:.1f} | {r['bf16']:.1f} | "
                 f"{r['refined']:.1f}% | {r['mhz']:.0f} |")
    L += ["", "`effective GB/s` is physical compressed traffic implied by the algorithm divided by latency; "
          "it is not BF16-equivalent bandwidth."]
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, f"{key}.md"), "w").write("\n".join(L) + "\n")

    def get(fmt, context):
        return next((r for r in rows if r["fmt"] == fmt and r["kv"] == 8 and r["gqa"] == 2
                     and r["context"] == context), None)

    def cell(r):
        return f"{r['us'] / 1000:.3f} ms, {r['gbs']:.0f} GB/s" if r else "—"

    a16, b16, a1m, b1m = get("3+3", 16384), get("4+2", 16384), get("3+3", 1048576), get("4+2", 1048576)
    row = (f"| [{gpu}]({key}.md) | {env.get('compute_cap', head.get('cc', '—'))} | "
           f"{head.get('copy', 0):.0f} GB/s | {cell(a16)} | {cell(b16)} | {cell(a1m)} | {cell(b1m)} | "
           f"{env.get('date', '—')} |")
    tpath = os.path.join(out_dir, "table.md")
    table_head = ["# Results by GPU", "",
                  "One row per GPU, produced by `bench/run_bench.sh` (see [docs/performance.md](../docs/performance.md));",
                  "the linked file contains all 36 rows and the run conditions. FBLA columns use the Qwen3-0.6B",
                  "geometry: 8 KV heads, GQA 2, D = 128.", "",
                  "| GPU | CC | D2D copy | 3+3 16K | 4+2 16K | 3+3 1M | 4+2 1M | Date |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
    old = []
    if os.path.exists(tpath):
        old = [x for x in open(tpath).read().splitlines() if x.startswith("| [") and f"]({key}.md)" not in x]
    open(tpath, "w").write("\n".join(table_head + sorted(old + [row])) + "\n")
    print(f"wrote {os.path.join(out_dir, key + '.md')} and its row in {tpath}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benches"))
