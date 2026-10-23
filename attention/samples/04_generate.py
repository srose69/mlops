# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Text generation end to end: a dense decoder (Qwen3, Qwen2, Llama) written out over the mlops ops.

    python 04_generate.py MODEL_DIR [--prompt TEXT] [--file DOC ...] [--mode 4+2] [--tokens 256]
                          [--nvfp4 mlp|all|none] [--attn fbla|sdpa] [--temperature 0] [--chunk 2048]

No transformers model: the weights come from the safetensors files, the layers are below.  Per layer

  x -> RMSNorm -> QKV (one GEMM) -> q / k RMSNorm (Qwen3) -> RoPE (mlops.positional, in place)
    -> attention: prefill(q, k, v, cache) for the prompt, chunk by chunk; append + decode per new token
    -> O projection -> + x -> RMSNorm -> gate / up (one GEMM) -> SiLU * -> down -> + x

The KV cache is FBLA's compressed one (mlops.attention): 6.25 bits per element.  It is allocated once for
prompt + new tokens, and the BF16 K / V never stay in memory.  Linear layers run on NVFP4EffMA: NVFP4 weights,
int8 activations by DP4A beyond 2 rows (prefill), the exact ones below (decode).  A BF16 checkpoint is quantized
at load time by `--nvfp4`: `mlp` only the MLP weights, `all` every projection and the LM head, `none` nothing
(PyTorch matmuls).  An NVFP4 checkpoint from ModelOpt (e.g. nvidia/Qwen3-8B-NVFP4) is used as it is (its
input_scale / k_scale / v_scale are not: activations stay int8 / exact, the cache is FBLA), and `--nvfp4 all`
also quantizes its BF16 LM head (on Qwen3-8B: +0.047 loss, so keep it BF16 when it fits).

Memory: the embedding table stays in host memory (a lookup moves only its rows); the KV cache is allocated
first, then as many layers as fit beside it go to the GPU and the rest, spread evenly through the depth, are
streamed from pinned host memory with their copies running ahead of the compute.  Qwen3-8B-NVFP4 runs on an 8 GB GTX 1080 with a desktop on it.  `--attn sdpa` swaps the attention
for PyTorch SDPA over a dense BF16 cache, as the baseline.

The tokenizer (and its chat template) is the model's, loaded through transformers' AutoTokenizer.
"""
import argparse
import collections
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

import nvfp4effma
from mlops.attention import append, decode, empty, prefill
from mlops.positional import RoPE


def load_weights(path):
    """Every tensor of the checkpoint (one file or an index of shards), by name, in host memory; floating-point
    weights as BF16 (NVFP4 data, block scales and FP32 global factors as they are)."""
    index = os.path.join(path, "model.safetensors.index.json")
    files = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) else ["model.safetensors"]
    w = {}
    for f in files:
        with safe_open(os.path.join(path, f), framework="pt", device="cpu") as s:
            for name in s.keys():
                t = s.get_tensor(name)
                keep = t.dtype in (torch.uint8, torch.float8_e4m3fn) or "_scale" in name
                w[name] = t if keep else t.to(torch.bfloat16)
    return w


class Linear:
    """y = x W^T (+ b) with W BF16 (PyTorch matmul) or NVFP4 (NVFP4EffMA: [data, block scales, global factor] in
    ModelOpt's layout; int8 activations by DP4A beyond 2 rows, the exact ones below).  `t` holds W's tensors,
    on the GPU or (streamed layers) in pinned host memory."""
    def __init__(self, t, nvfp4, bias=None):
        self.t, self.nvfp4, self.bias = t, nvfp4, bias

    @classmethod
    def load(cls, w, names, quantize):
        """The matrices `names` (checkpoint prefixes) stacked by rows into one.  An NVFP4 checkpoint's are used as
        they are, each keeping its global factor (one per row); BF16 ones are quantized when `quantize`."""
        bias = [w.get(n + ".bias") for n in names]
        bias = torch.cat(bias).cuda() if bias[0] is not None else None
        if names[0] + ".weight_scale" in w:  # ModelOpt NVFP4: packed E2M1, E4M3 scales per 16, FP32 global
            g = torch.cat([w[n + ".weight_scale_2"].float().reshape(1).expand(w[n + ".weight"].shape[0]) for n in names])
            g = g[:1].clone() if bool((g == g[0]).all()) else g
            return cls([torch.cat([w[n + ".weight"] for n in names]), torch.cat([w[n + ".weight_scale"] for n in names]), g],
                       True, bias)
        wt = torch.cat([w[n + ".weight"] for n in names])
        if not quantize:
            return cls([wt], False, bias)  # in host memory until placed
        wt = wt.cuda()
        a = wt.abs().amax().float().reshape(1)
        q = nvfp4effma.NVFP4Tensor.from_float(wt, a / (6.0 * 448.0))  # ModelOpt's global factor amax / (6 * 448)
        return cls([q.data, q.scales, q.global_scale], True, bias)

    def __call__(self, x, t=None):
        t = self.t if t is None else t
        y = nvfp4effma.linear(x, nvfp4effma.NVFP4Tensor(*t)) if self.nvfp4 else F.linear(x, t[0])
        return y if self.bias is None else y + self.bias


def rms_norm(x, w, eps):
    """RMSNorm in FP32, rounded to BF16 before the weight (as transformers' Llama / Qwen)."""
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class Decoder:
    LINEARS = ("qkv", "o", "gate_up", "down")
    NBUF = 3  # streamed layers on the GPU at a time: the one computing and two copies ahead (keeps the copy
    #           engine busy; with two, the copies hid about 85 % behind compute, with three 96 %)

    def __init__(self, path, nvfp4="mlp", attn="fbla", mode="4+2"):
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg, self.attn, self.mode = cfg, attn, mode
        self.H, self.Hq, self.Hkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.D = cfg.get("head_dim") or self.H // self.Hq
        self.eps = cfg["rms_norm_eps"]
        if attn == "fbla" and (self.D != 128 or self.Hq // self.Hkv not in (1, 2, 4)):
            raise SystemExit(f"FBLA needs head dim 128 and 1, 2 or 4 query heads per KV head; this model has "
                             f"{self.D} and {self.Hq}/{self.Hkv}")
        w = load_weights(path)
        self.prequantized = any(k.endswith(".weight_scale") for k in w)
        proj, mlp = nvfp4 == "all", nvfp4 in ("mlp", "all")
        self.embed = w["model.embed_tokens.weight"]  # stays in host memory: a lookup moves only the rows it needs
        w.setdefault("lm_head.weight", self.embed)    # tied embeddings: the same matrix
        self.lm_head = Linear.load(w, ["lm_head"], proj)
        self.lm_head.t = [x.cuda() for x in self.lm_head.t]
        self.norm = w["model.norm.weight"].cuda()
        self.rope = RoPE(self.D, cfg.get("rope_theta", 10000.0), scaling=cfg.get("rope_scaling"),
                         max_position_embeddings=cfg.get("max_position_embeddings"))
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):  # linear weights stay where they are made until place()
            p = f"model.layers.{i}."
            a = p + "self_attn."
            q_norm = w.get(a + "q_norm.weight")
            self.layers.append(dict(
                ln1=w[p + "input_layernorm.weight"].cuda(), ln2=w[p + "post_attention_layernorm.weight"].cuda(),
                q_norm=q_norm.cuda() if q_norm is not None else None,
                k_norm=w[a + "k_norm.weight"].cuda() if q_norm is not None else None,
                qkv=Linear.load(w, [a + "q_proj", a + "k_proj", a + "v_proj"], proj),
                o=Linear.load(w, [a + "o_proj"], proj),
                gate_up=Linear.load(w, [p + "mlp.gate_proj", p + "mlp.up_proj"], mlp),
                down=Linear.load(w, [p + "mlp.down_proj"], mlp)))
        del w
        self.streamed, self.queue, self.next = [], collections.deque(), 0  # layers in pinned host memory
        self.copy = torch.cuda.Stream()

    def place(self, capacity, chunk, gpu_layers=None):
        """The KV cache first, then as many layers' weights on the GPU as fit beside it and the activations of a
        prefill chunk; the rest in pinned host memory, streamed in per forward pass (NBUF - 1 layers ahead).  Can be
        called again (another capacity): the cache is replaced and the layers re-placed."""
        self.cache, self.streamed = None, []
        for L in self.layers:
            for n in self.LINEARS:
                L[n].t = [x.cpu() for x in L[n].t]
        torch.cuda.empty_cache()
        self.cache = self.new_cache(capacity)
        I = self.layers[0]["down"].t[0].shape[1] * (2 if self.layers[0]["down"].nvfp4 else 1)  # intermediate size
        reserve = chunk * (6 * I + 16 * self.H) * 2 + (256 << 20)
        if self.attn == "sdpa":  # the baseline's GQA-expanded K / V, scores and softmax of 512 queries (BF16)
            reserve += self.Hq * capacity * (4 * self.D + 4 * 512)
        n = len(self.layers)
        b = max(sum(x.numel() * x.element_size() for m in self.LINEARS for x in L[m].t) for L in self.layers)
        budget = torch.cuda.mem_get_info()[0] - reserve
        resident = n if n * b <= budget else max(0, min(n, (budget - self.NBUF * b) // b))  # + copy buffers
        resident = resident if gpu_layers is None else min(resident, gpu_layers)
        # Streaming plan.  A layer's copy takes c = b / (host -> GPU bandwidth); the Z streamed layers cost Z c of
        # copying per pass, the bound of a pass when it exceeds the compute.  Spread evenly, a copy runs while the
        # resident layers before its layer compute and the copy engine never waits for a slot: a pass takes
        # max(compute, Z c).  Packed together, the copies of a block could only overlap each other's (short)
        # compute and a pass would take about compute + (Z - NBUF + 1) c.
        streamed = {int((k + 0.5) * n / (n - resident)) for k in range(n - resident)} if resident < n else set()
        self.plan = dict(layer_mb=b / 2**20, bandwidth=self.bandwidth() if streamed else 0.0)
        self.plan["copy_ms"] = len(streamed) * b / self.plan["bandwidth"] * 1e3 if streamed else 0.0
        for i, L in enumerate(self.layers):
            for m in self.LINEARS:
                L[m].t = [x.cpu().pin_memory() if i in streamed else x.cuda() for x in L[m].t]
        self.streamed, self.queue, self.next = sorted(streamed), collections.deque(), 0
        return n - len(self.streamed)

    def new_cache(self, capacity):
        """Per layer: an FBLA cache (the quantization mode is fixed here), or dense BF16 K / V for SDPA."""
        if self.attn == "fbla":
            return [empty(1, self.Hkv, capacity, self.mode) for _ in self.layers]
        return [[torch.empty(1, self.Hkv, capacity, self.D, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
                for _ in self.layers]

    def attention(self, q, k, v, cache, pos):
        """q [1, Hq, S, D], k / v [1, Hkv, S, D] of positions pos .. pos + S - 1 -> [1, Hq, S, D]."""
        S = q.shape[2]
        if self.attn == "fbla":
            if S > 1:  # the prompt (a chunk of it): appended, its queries through the cache
                return prefill(q, k, v, cache)
            append(cache, k, v)  # a new token: appended, then one query per head
            return decode(q[:, :, 0], cache)[:, :, None]
        kc, vc = cache
        kc[:, :, pos:pos + S], vc[:, :, pos:pos + S] = k, v
        T, g = pos + S, self.Hq // self.Hkv
        K, V = kc[:, :, :T].repeat_interleave(g, 1), vc[:, :, :T].repeat_interleave(g, 1)
        if S == 1:
            return F.scaled_dot_product_attention(q, K, V)
        # 512 queries at a time: on GPUs without a fused BF16 kernel SDPA materializes the scores
        mask = torch.arange(T, device="cuda")[None] <= torch.arange(pos, T, device="cuda")[:, None]
        return torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + 512], K, V, attn_mask=mask[i:i + 512])
                          for i in range(0, S, 512)], 2)

    @staticmethod
    def bandwidth(nbytes=256 << 20):
        """Pinned host -> GPU copy rate (bytes / s), measured once."""
        if not hasattr(Decoder, "_bw"):
            h, d = torch.empty(nbytes, dtype=torch.uint8).pin_memory(), torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            d.copy_(h)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                d.copy_(h, non_blocking=True)
            torch.cuda.synchronize()
            Decoder._bw = 3 * nbytes / (time.perf_counter() - t0)
            del h, d
            torch.cuda.empty_cache()
        return Decoder._bw

    def prefetch(self):
        """Keep NBUF - 1 streamed layers copying ahead, cyclically: after a pass's last streamed layer the next
        pass's first ones are already on their way (the weights do not change)."""
        while self.streamed and len(self.queue) < self.NBUF - 1:
            i = self.streamed[self.next]
            self.next = (self.next + 1) % len(self.streamed)
            self.queue.append((i, *self.fetch(i)))

    def fetch(self, i):
        """A streamed layer's weights to the GPU on the copy stream: ({name: tensors}, event)."""
        with torch.cuda.stream(self.copy):
            t = {n: [x.to("cuda", non_blocking=True) for x in self.layers[i][n].t] for n in self.LINEARS}
            ev = torch.cuda.Event()
            ev.record()
        return t, ev

    @torch.no_grad()
    def forward(self, ids, pos, last=True):
        """ids [S] (host) at positions pos .. pos + S - 1 -> the logits of the last one [vocab] (FP32); last=False:
        the final hidden states of all S (normalized, before the LM head)."""
        S, D = ids.shape[0], self.D
        x = self.embed[ids].to("cuda", non_blocking=True)
        self.prefetch()
        for i, (L, c) in enumerate(zip(self.layers, self.cache)):
            wt = dict.fromkeys(self.LINEARS)
            if self.streamed and self.queue[0][0] == i:  # a streamed layer: wait for its copy, queue the next one
                _, wt, ev = self.queue.popleft()
                torch.cuda.current_stream().wait_event(ev)
                for ts in wt.values():
                    for t in ts:
                        t.record_stream(torch.cuda.current_stream())
                self.prefetch()
            qkv = L["qkv"](rms_norm(x, L["ln1"], self.eps), wt["qkv"])
            q, k, v = qkv.split([self.Hq * D, self.Hkv * D, self.Hkv * D], -1)
            q, k, v = q.view(1, S, self.Hq, D), k.view(1, S, self.Hkv, D), v.view(1, S, self.Hkv, D)
            if L["q_norm"] is not None:  # Qwen3: RMSNorm of every head's q and k
                q, k = rms_norm(q, L["q_norm"], self.eps), rms_norm(k, L["k_norm"], self.eps)
            else:
                q, k = q.contiguous(), k.contiguous()
            self.rope(q, k, layout="bshd", offset=pos, inplace=True)
            o = self.attention(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                               v.transpose(1, 2).contiguous(), c, pos)
            x = x + L["o"](o[0].transpose(0, 1).reshape(S, self.Hq * D), wt["o"])
            g, u = L["gate_up"](rms_norm(x, L["ln2"], self.eps), wt["gate_up"]).chunk(2, -1)
            x = x + L["down"](F.silu(g) * u, wt["down"])
        if not last:
            return rms_norm(x, self.norm, self.eps)
        return self.lm_head(rms_norm(x[-1:], self.norm, self.eps))[0].float()


def sample(logits, temperature, top_p, gen):
    if temperature <= 0:
        return int(logits.argmax())
    p = torch.softmax(logits / temperature, -1)
    ps, idx = p.sort(descending=True)
    keep = ps.cumsum(-1) - ps < top_p  # the smallest set with mass >= top_p
    ps = ps * keep
    return int(idx[torch.multinomial(ps / ps.sum(), 1, generator=gen)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="Explain in a few paragraphs how a KV cache speeds up text generation.")
    ap.add_argument("--file", nargs="*", default=[], help="documents placed before the prompt (long context)")
    ap.add_argument("--mode", default="4+2", choices=["4+2", "3+3"], help="FBLA quantization mode")
    ap.add_argument("--attn", default="fbla", choices=["fbla", "sdpa"], help="sdpa: dense BF16 cache, the baseline")
    ap.add_argument("--nvfp4", default="mlp", choices=["mlp", "all", "none"], help="linear layers in NVFP4")
    ap.add_argument("--tokens", type=int, default=256, help="new tokens at most")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk (bounds the activations' memory)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0: greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ignore-eos", action="store_true", help="generate all --tokens (tests, benchmarks)")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--think", action="store_true", help="Qwen3: let the model think first")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(open(f).read() for f in a.file)
    user = f"{text}\n\n{a.prompt}" if text else a.prompt
    if a.raw or tok.chat_template is None:
        ids = tok(user).input_ids
    else:
        chat = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=a.think)
        ids = tok(chat, add_special_tokens=False).input_ids
    ids = torch.tensor(ids)
    n = ids.shape[0]

    t0 = time.perf_counter()
    model = Decoder(a.model, a.nvfp4, a.attn, a.mode)
    resident = model.place(n + a.tokens, a.chunk)
    cache = model.cache
    attn = f"FBLA {a.mode}" if a.attn == "fbla" else "SDPA, BF16 cache"
    linear = "NVFP4 checkpoint" + (", LM head NVFP4" if a.nvfp4 == "all" else "") if model.prequantized else f"NVFP4: {a.nvfp4}"
    print(f"{a.model}: {len(model.layers)} layers, {model.Hq}/{model.Hkv} heads; linear {linear}; attention {attn}; "
          f"loaded in {time.perf_counter() - t0:.1f} s")
    if model.streamed:
        print(f"  GPU memory: {resident} layers resident, {len(model.streamed)} streamed from pinned host memory")

    # prefill: the prompt in chunks; only the last position's logits are needed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, n, a.chunk):
        logits = model.forward(ids[s:s + a.chunk], s)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    print(f"prompt {n} tokens\n" + "-" * 80)

    # decode: one token per step, streamed as text
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    stop = {tok.eos_token_id}
    gc = os.path.join(a.model, "generation_config.json")
    if os.path.exists(gc):  # an id or a list of ids
        eos = json.load(open(gc)).get("eos_token_id")
        stop |= set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    out, shown = [], ""
    t0 = time.perf_counter()
    for i in range(a.tokens):
        t = sample(logits, a.temperature, a.top_p, gen)
        if t in stop and not a.ignore_eos:
            break
        out.append(t)
        s = tok.decode(out, skip_special_tokens=True)
        if not s.endswith("�"):  # a multi-byte character completes with a later token
            print(s[len(shown):], end="", flush=True)
            shown = s
        if i + 1 < a.tokens:
            logits = model.forward(torch.tensor([t]), n + i)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0
    print("\n" + "-" * 80)

    nbytes = lambda xs: sum(x.numel() * x.element_size() for x in xs) / 2**20
    total = n + len(out)
    dense = 2 * len(model.layers) * model.Hkv * total * model.D * 2 / 2**20
    print(f"prefill {n} tokens: {t_prefill * 1e3:.0f} ms ({n / t_prefill:.0f} tokens/s); "
          f"decode {len(out)} tokens: {len(out) / max(t_decode, 1e-9):.1f} tokens/s")
    if a.attn == "fbla":  # the records are laid out for the capacity; the sink and the open group are BF16 windows
        rec = nbytes(x for c in cache for x in (c.coarse, c.fine, c.meta))
        win = nbytes(x for c in cache for x in (c.sink_k, c.sink_v, c.open_k, c.open_v))
        print(f"KV cache, {total} tokens ({(total - 1) // 128} closed groups): FBLA {rec + win:.1f} MiB = packed records "
              f"{rec:.1f} MiB (6.25 bits per element, laid out for {n + a.tokens} tokens) + BF16 sink and open "
              f"windows {win:.1f} MiB (fixed); BF16 K / V would be {dense:.1f} MiB")
    else:
        print(f"KV cache, {total} tokens: BF16 K / V {dense:.1f} MiB")


if __name__ == "__main__":
    main()
