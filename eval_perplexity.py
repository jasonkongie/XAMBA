"""
eval_perplexity.py  —  CPU Pipeline

Evaluates WikiText-2 perplexity for each base model at every quantization point:
  - baseline (FP32)
  - point01..point10 (mixed INT4/INT8/FP16 from merged sensitivity)
  - uniform_int4  (all layers INT4)
  - uniform_int8  (all layers INT8)

Uses fake-quantization (quantize_weight_per_channel_absmax) on the PyTorch model,
matching the algorithm from pareto.py exactly.

Output:
    perplexity_results.json   — {model_name: {point: ppl}}

Usage:
    python eval_perplexity.py
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
    "mamba-130m-hf": {
        "hf_id": "state-spaces/mamba-130m-hf",
        "sensitivity_4bit": "mamba130m_sensitivity_results_4bits.json",
        "sensitivity_8bit": "mamba130m_sensitivity_results_8bits.json",
    },
    "mamba-1.4b-hf": {
        "hf_id": "state-spaces/mamba-1.4b-hf",
        "sensitivity_4bit": "mamba1_4b_sensitivity_results_4bits.json",
        "sensitivity_8bit": "mamba1_4b_sensitivity_results_8bits.json",
    },
}

SEQ_LEN      = 2048
N_POINTS     = 10
MAX_WINDOWS  = 20      # 20 windows (~40k tokens)
OUTPUT_JSON  = "perplexity_results.json"

# ── Sensitivity (same as quantize_mixed.py) ──────────────────────────────────

def build_sensitivity_list(path4, path8):
    """Merge 4-bit and 8-bit KL files into [(layer, bit, kl), ...] sorted ASC."""
    sens4 = json.load(open(path4))
    sens8 = json.load(open(path8))
    merged = []
    for layer, stats in sens4.items():
        merged.append((layer, 4, stats["kl_student_to_teacher"]))
    for layer, stats in sens8.items():
        merged.append((layer, 8, stats["kl_student_to_teacher"]))
    return sorted(merged, key=lambda t: t[2])


def compute_cutoff_indices(n_entries, n_points=10):
    return [n_entries * i // n_points for i in range(1, n_points + 1)]


def get_layer_assignments(S, cutoff_idx):
    """Last-wins: 8-bit entries come first, 4-bit overwrites later."""
    assignment = {}
    for layer, bit, kl in S[:cutoff_idx]:
        assignment[layer] = bit
    return assignment

# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_single_layer(model, layer_name, n_bits):
    """In-place fake-quantization of a single layer's weight (from pareto.py)."""
    for name, mod in model.named_modules():
        if name == layer_name and hasattr(mod, "weight"):
            mod.weight.data = quantize_weight_per_channel_absmax(
                mod.weight.data, n_bits=n_bits
            )
            return
    # Don't raise — some layers (e.g., lm_head in Mamba v1) might use
    # a different module structure. Just warn.
    print(f"    [warn] Layer not found: {layer_name}")

# ── Model Loading ────────────────────────────────────────────────────────────

def load_fresh_model(hf_id):
    """Load MambaForCausalLM with use_cache=False."""
    config = AutoConfig.from_pretrained(hf_id)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, config=config, trust_remote_code=True
    )
    return model.eval()

# ── Perplexity ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, seq_len=2048):
    """Non-overlapping windows, cross-entropy loss, exp(mean NLL)."""
    enc       = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids          # [1, N]
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
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        hf_id     = cfg["hf_id"]
        tokenizer = AutoTokenizer.from_pretrained(hf_id)

        S = build_sensitivity_list(cfg["sensitivity_4bit"], cfg["sensitivity_8bit"])
        all_layer_names = list(set(l for l, _, _ in S))
        indices = compute_cutoff_indices(len(S), N_POINTS)

        results = {}

        # ── Baseline ─────────────────────────────────────────────────────
        print("\n  [baseline] FP32 — no quantization")
        model = load_fresh_model(hf_id)
        ppl   = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["baseline"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        # ── Mixed-precision points ───────────────────────────────────────
        for point_idx, cutoff in enumerate(indices):
            point_name = f"point{point_idx + 1:02d}"
            assignment = get_layer_assignments(S, cutoff)
            n4 = sum(1 for b in assignment.values() if b == 4)
            n8 = sum(1 for b in assignment.values() if b == 8)

            print(f"\n  [{point_name}] cutoff {cutoff}/{len(S)}  (INT4:{n4}  INT8:{n8})")
            model = load_fresh_model(hf_id)
            for layer, bit in assignment.items():
                quantize_single_layer(model, layer, n_bits=bit)
            ppl = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
            results[point_name] = round(ppl, 3)
            print(f"    → PPL = {ppl:.3f}")
            del model

        # ── Uniform INT4 ────────────────────────────────────────────────
        print(f"\n  [uniform_int4] all {len(all_layer_names)} layers → 4-bit")
        model = load_fresh_model(hf_id)
        for layer in all_layer_names:
            quantize_single_layer(model, layer, n_bits=4)
        ppl = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["uniform_int4"] = round(ppl, 3)
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

    # Print summary
    for model_name, results in all_results.items():
        print(f"\n  {model_name}:")
        for k, v in results.items():
            print(f"    {k:<16} {v:>8.3f}")


if __name__ == "__main__":
    main()
