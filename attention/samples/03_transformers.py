# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""FBLA inside a Hugging Face transformers model: prefill and decode against the FBLA cache.

    python 03_transformers.py MODEL_DIR [--mode 4+2] [--tokens 64] [--context 4096] [TEXT_FILE ...]

Works with models whose attention has head dim 128 and 1, 2 or 4 query heads per KV head (Qwen3, Llama 3, ...).
Where FBLA goes: the model's attention function.  transformers calls it per layer with q, k, v after the
projections, the q / k norm and RoPE; it is replaced through AttentionInterface:

  prefill (many queries)  prefill(q, k, v, cache): the prompt's K / V go into that layer's FBLA cache and its
                          queries attend through it (full 6-bit codes, streamed DP4A tiles: no BF16 score matrix)
  decode  (one query)     the new token's K / V are appended to the cache, the query decodes against it

transformers still keeps its own dense KV cache here (it is how the new token's K / V reach this function);
an inference engine writes the FBLA layout instead and drops the BF16 cache.  The sample generates greedily
with SDPA and with FBLA from the same prompt and compares the continuations.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from mlops.attention import append, decode, empty, prefill


class FBLAState:
    """Per-layer FBLA caches of one generation; `mode` None runs plain SDPA (the baseline)."""
    def __init__(self):
        self.mode, self.capacity, self.caches = None, 0, {}

    def reset(self, mode, capacity):
        self.mode, self.capacity, self.caches = mode, capacity, {}


STATE = FBLAState()


def fbla_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers attention interface: query [B, Hq, L, D], key / value [B, Hkv, T, D] (the whole context,
    the L new tokens last); returns [B, L, Hq, D]."""
    L = query.shape[2]
    if STATE.mode is None:  # baseline: exact causal attention
        out = F.scaled_dot_product_attention(query, key, value, is_causal=L > 1, scale=scaling,
                                             enable_gqa=query.shape[1] != key.shape[1])
    elif L > 1:  # prefill (or a later chunk of it): the L new tokens into the cache, their queries through it
        if module.layer_idx not in STATE.caches:  # the mode is fixed here, with the layer's empty cache
            STATE.caches[module.layer_idx] = empty(key.shape[0], key.shape[1], STATE.capacity, STATE.mode,
                                                    device=key.device)
        out = prefill(query, key[:, :, -L:], value[:, :, -L:], STATE.caches[module.layer_idx], scale=scaling)
    else:  # decode: the new token's K / V into the cache, then attention over everything so far
        cache = STATE.caches[module.layer_idx]
        append(cache, key[:, :, -1:], value[:, :, -1:])
        out = decode(query[:, :, 0].contiguous(), cache, scale=scaling)[:, :, None]
    return out.transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("text", nargs="*", help="prompt files (default: this package's README)")
    ap.add_argument("--mode", default="4+2", choices=["4+2", "3+3"], help="FBLA quantization mode (fixed per cache)")
    ap.add_argument("--context", type=int, default=4096, help="prompt tokens")
    ap.add_argument("--tokens", type=int, default=64, help="tokens to generate")
    a = ap.parse_args()

    AttentionInterface.register("fbla", fbla_attention)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="fbla").cuda().eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    if hd != 128 or cfg.num_attention_heads // cfg.num_key_value_heads not in (1, 2, 4):
        raise SystemExit(f"FBLA needs head dim 128 and GQA 1/2/4; this model has {hd} and "
                         f"{cfg.num_attention_heads}/{cfg.num_key_value_heads} heads")

    files = a.text or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")]
    text = "\n\n".join(open(f).read() for f in files)
    ids = tok(text, return_tensors="pt").input_ids[:, -a.context:].cuda()
    print(f"{a.model}: prompt {ids.shape[1]} tokens, {a.tokens} generated greedily\n")

    runs = {}
    for name, mode in (("SDPA", None), (f"FBLA {a.mode}", a.mode)):
        STATE.reset(mode, capacity=ids.shape[1] + a.tokens)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False)
        runs[name] = out[0, ids.shape[1]:]
        print(f"--- {name}\n{tok.decode(runs[name])!r}\n")

    base, fbla = runs.values()
    diff = (base != fbla).nonzero()
    same = diff[0].item() if len(diff) else len(base)
    print(f"identical tokens before the first difference: {same} / {len(base)}")

    # Teacher-forced: both attentions on the SDPA continuation, next-token distributions compared per step.
    seq = torch.cat([ids[0], base])[None]
    logp = {}
    for name, mode in (("SDPA", None), ("FBLA", a.mode)):
        STATE.reset(mode, capacity=seq.shape[1])
        with torch.no_grad():
            past = model(seq[:, :ids.shape[1]], use_cache=True).past_key_values  # prefill
            steps = []
            for t in range(ids.shape[1], seq.shape[1]):                          # one decode step per token
                o = model(seq[:, t:t + 1], past_key_values=past, use_cache=True)
                past = o.past_key_values
                steps.append(F.log_softmax(o.logits[0, -1].float(), -1))
        logp[name] = torch.stack(steps)
    kl = (logp["SDPA"].exp() * (logp["SDPA"] - logp["FBLA"])).sum(-1)
    agree = (logp["SDPA"].argmax(-1) == logp["FBLA"].argmax(-1)).float().mean()
    print(f"teacher-forced over {len(kl)} decode steps: KL(SDPA || FBLA) mean {kl.mean():.2e}, max {kl.max():.2e}; "
          f"top-1 agreement {agree:.1%}")


if __name__ == "__main__":
    main()
