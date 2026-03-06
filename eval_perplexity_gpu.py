"""
eval_perplexity_gpu.py  —  GPU Pipeline

Evaluates WikiText-2 perplexity for GPU-compatible quantization points:
  - baseline (FP32)
  - gpu_point01..gpu_point10 (INT8/FP16 mixed from 8-bit sensitivity)
  - uniform_int8 (all layers INT8)

Uses fake-quantization to INT8 only (matching GPU's supported precision).

Output:
    perplexity_results_gpu.json   — {model_name: {point: ppl}}

Usage:
    python eval_perplexity_gpu.py
"""

import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from quant_utils import quantize_weight_per_channel_absmax

# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mamba2_b_1_t_4": {
        "hf_id": "yuji96/mamba2-130m-hf",
        # mamba2 shares the GPTNeoX tokenizer with mamba-130m-hf
        "tokenizer_id": "state-spaces/mamba-130m-hf",
        # No 8-bit sensitivity file yet; 4-bit KL values used as ranking proxy
        "sensitivity_8bit": "sensitivity_results_mamba2-130m_4bits.json",
    },
}

SEQ_LEN      = 2048
N_POINTS     = 10
MAX_WINDOWS  = 20      # 20 windows (~40k tokens)
OUTPUT_JSON  = "perplexity_results_gpu.json"

# ── Sensitivity (8-bit only) ─────────────────────────────────────────────────

def load_sensitivity_8bit(path):
    with open(path) as f:
        data = json.load(f)
    layers = [(name, stats["kl_student_to_teacher"]) for name, stats in data.items()]
    layers.sort(key=lambda x: x[1])
    return layers


def compute_cutoff_indices(n_entries, n_points=10):
    return [n_entries * i // n_points for i in range(1, n_points + 1)]

# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_single_layer(model, layer_name, n_bits):
    for name, mod in model.named_modules():
        if name == layer_name and hasattr(mod, "weight"):
            mod.weight.data = quantize_weight_per_channel_absmax(
                mod.weight.data, n_bits=n_bits
            )
            return
    print(f"    [warn] Layer not found: {layer_name}")

# ── Model Loading ────────────────────────────────────────────────────────────

def load_fresh_model(hf_id):
    config = AutoConfig.from_pretrained(hf_id)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, config=config, trust_remote_code=True
    )
    return model.eval()

# ── Perplexity ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, seq_len=2048):
    enc       = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids
    N         = input_ids.shape[1]
    n_windows = max(1, (N - 1) // seq_len)
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    if MAX_WINDOWS is not None:
        n_windows = min(n_windows, MAX_WINDOWS)

    nlls = []
    for i in tqdm(range(n_windows), desc="    eval", leave=False):
        s     = i * seq_len
        e     = min(s + seq_len + 1, N)
        chunk = input_ids[:, s:e]
        if chunk.shape[1] < 2:
            continue
        inp = chunk[:, :-1]
        tgt = chunk[:, 1:]

        logits = model(input_ids=inp).logits
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            ignore_index=pad_id,
        )
        nlls.append(loss.item())

    return torch.exp(torch.tensor(nlls).mean()).item()

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading WikiText-2 test set ...")
    raw       = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(t for t in raw["text"] if t.strip())
    print(f"  {len(test_text):,} characters")

    all_results = {}

    for model_name, cfg in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  (GPU pipeline — INT8/FP16)")
        print(f"{'='*60}")

        hf_id        = cfg["hf_id"]
        tokenizer_id = cfg.get("tokenizer_id", hf_id)
        tokenizer    = AutoTokenizer.from_pretrained(tokenizer_id)

        sensitivity = load_sensitivity_8bit(cfg["sensitivity_8bit"])
        all_layer_names = [name for name, _ in sensitivity]
        indices = compute_cutoff_indices(len(sensitivity), N_POINTS)

        results = {}

        # ── Baseline ─────────────────────────────────────────────────────
        print("\n  [baseline] FP32 — no quantization")
        model = load_fresh_model(hf_id)
        ppl   = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["baseline"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        # ── GPU mixed-precision points (INT8/FP16) ───────────────────────
        for point_idx, cutoff in enumerate(indices):
            point_name = f"gpu_point{point_idx + 1:02d}"

            int8_layers = [name for name, _ in sensitivity[:cutoff]]
            fp16_layers = [name for name, _ in sensitivity[cutoff:]]

            print(f"\n  [{point_name}] cutoff {cutoff}/{len(sensitivity)}  "
                  f"(INT8:{len(int8_layers)}  FP16:{len(fp16_layers)})")

            model = load_fresh_model(hf_id)
            for layer in int8_layers:
                quantize_single_layer(model, layer, n_bits=8)
            ppl = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
            results[point_name] = round(ppl, 3)
            print(f"    → PPL = {ppl:.3f}")
            del model

        # ── Uniform INT8 ────────────────────────────────────────────────
        print(f"\n  [uniform_int8] all {len(all_layer_names)} layers → 8-bit")
        model = load_fresh_model(hf_id)
        for layer in all_layer_names:
            quantize_single_layer(model, layer, n_bits=8)
        ppl = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["uniform_int8"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        all_results[model_name] = results

    # ── Save ─────────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Saved → {OUTPUT_JSON}")

    for model_name, results in all_results.items():
        print(f"\n  {model_name}:")
        for k, v in results.items():
            print(f"    {k:<16} {v:>8.3f}")


if __name__ == "__main__":
    main()
