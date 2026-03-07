"""
eval_perplexity.py  —  CPU Pipeline

Evaluates WikiText-2 perplexity using NNCF compress_weights applied directly
to the HuggingFace PyTorch model, matching the quantization algorithm used
by quantize_mixed.py.  This avoids the static-shape limitation of the OV
export (which is sized for benchmark latency, not evaluation seq_len).

Configs evaluated per model:
  - FP32 baseline   : unquantized HuggingFace model
  - mixed-precision : NNCF INT4/INT8 per sensitivity ranking (10 points)
  - uniform INT8    : NNCF INT8_SYM on all linear layers
  - uniform INT4    : NNCF INT4_SYM on all linear layers

Output:
    perplexity_results_{METRIC_TAG}.json  —  {model_name: {point: ppl}}

Usage:
    python eval_perplexity.py
"""

import json
import torch
import torch.nn.functional as F
import nncf
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mamba-130m-hf": {
        "hf_id":            "state-spaces/mamba-130m-hf",
        "sensitivity_4bit": "mamba130m_sensitivity_results_4bits.json",
        "sensitivity_8bit": "mamba130m_sensitivity_results_8bits.json",
    },
    "mamba-1.4b-hf": {
        "hf_id":            "state-spaces/mamba-1.4b-hf",
        "sensitivity_4bit": "mamba1_4b_sensitivity_results_4bits.json",
        "sensitivity_8bit": "mamba1_4b_sensitivity_results_8bits.json",
    },
}

SEQ_LEN     = 2048
MAX_WINDOWS = 20
N_POINTS    = 10

# ── Sensitivity metric ────────────────────────────────────────────────────────
SENSITIVITY_METRIC = "kl_student_to_teacher"
METRIC_TAG         = "kl" if SENSITIVITY_METRIC == "kl_student_to_teacher" else "sqnr"
OUTPUT_JSON        = f"perplexity_results_{METRIC_TAG}.json"

# Exclude Conv1d (SSM conv1d) — matches OV pipeline's IgnoredScope(types=["Convolution"])
LINEAR_ONLY_SCOPE = nncf.IgnoredScope(types=["Conv1d"], validate=False)

# ── Per-layer quantization ────────────────────────────────────────────────────

def quantize_weight(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Per-output-channel symmetric absmax quantization.
    Matches NNCF compress_weights INT4_SYM / INT8_SYM with group_size=-1.
    Used for mixed-precision points where two NNCF passes on the same
    PyTorch model would fail (NNCF wraps layers after the first call).

    scale = max_abs / q_max  so that  w / scale  maps to [-q_max, q_max],
    giving 2*q_max+1 distinct quantized values (e.g. 255 for INT8, 15 for INT4).
    Without the / q_max, w / scale maps to [-1, 1] and round() collapses
    everything to {-1, 0, 1} — only 3 values, which destroys quality.
    """
    q_max  = 2 ** (n_bits - 1) - 1
    scales = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) / q_max
    return (w / scales).round().clamp(-q_max, q_max) * scales

# ── Sensitivity helpers ───────────────────────────────────────────────────────

def build_sensitivity_list(path4, path8):
    sens4 = json.load(open(path4))
    sens8 = json.load(open(path8))
    merged = []
    for layer, stats in sens4.items():
        merged.append((layer, 4, stats[SENSITIVITY_METRIC]))
    for layer, stats in sens8.items():
        merged.append((layer, 8, stats[SENSITIVITY_METRIC]))
    reverse = (SENSITIVITY_METRIC == "sqnr_db")
    return sorted(merged, key=lambda t: t[2], reverse=reverse)

def compute_cutoff_indices(n_entries, n_points=10):
    segment_size = (n_entries - 1) // n_points
    return [segment_size * i for i in range(1, n_points + 1)]

def get_layer_assignments(S, cutoff_idx):
    assignment = {}
    for layer, bit, _ in S[:cutoff_idx]:
        assignment[layer] = bit
    return assignment

# ── NNCF mixed-precision ──────────────────────────────────────────────────────

def apply_mixed_precision(model, assignment):
    """
    Per-layer in-place quantization for mixed-precision points.
    Uses the same per-output-channel symmetric absmax algorithm as
    NNCF compress_weights INT4_SYM / INT8_SYM with group_size=-1.
    (Two NNCF passes on the same PyTorch model fail because NNCF wraps
    linear layers after the first call, breaking the second call.)
    Conv1d layers are naturally skipped — they won't appear in assignment.
    """
    for name, module in model.named_modules():
        if name in assignment and hasattr(module, "weight"):
            module.weight.data = quantize_weight(module.weight.data, n_bits=assignment[name])
    return model

# ── Model loading ─────────────────────────────────────────────────────────────

def load_fresh_model(hf_id):
    config = AutoConfig.from_pretrained(hf_id)
    config.use_cache = False
    return AutoModelForCausalLM.from_pretrained(
        hf_id, config=config, trust_remote_code=True
    ).eval()

# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, seq_len=2048):
    enc       = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids
    N         = input_ids.shape[1]
    n_windows = min(MAX_WINDOWS, max(1, (N - 1) // seq_len))
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    nlls = []
    for i in tqdm(range(n_windows), desc="    eval", leave=False):
        s, e  = i * seq_len, min(i * seq_len + seq_len + 1, N)
        chunk = input_ids[:, s:e]
        if chunk.shape[1] < 2:
            continue
        logits = model(input_ids=chunk[:, :-1]).logits
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk[:, 1:].reshape(-1),
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
        S         = build_sensitivity_list(cfg["sensitivity_4bit"], cfg["sensitivity_8bit"])
        indices   = compute_cutoff_indices(len(S), N_POINTS)
        results   = {}

        # ── Baseline (FP32) ───────────────────────────────────────────────
        print("\n  [baseline_fp32]")
        model = load_fresh_model(hf_id)
        ppl   = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["baseline_fp32"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        # ── Mixed-precision points ────────────────────────────────────────
        for point_idx, cutoff in enumerate(indices):
            label      = f"{METRIC_TAG}_point{point_idx + 1:02d}"
            assignment = get_layer_assignments(S, cutoff)
            n4 = sum(1 for b in assignment.values() if b == 4)
            n8 = sum(1 for b in assignment.values() if b == 8)
            print(f"\n  [{label}] cutoff {cutoff}/{len(S)}  INT4:{n4}  INT8:{n8}")
            model = load_fresh_model(hf_id)
            model = apply_mixed_precision(model, assignment)
            ppl   = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
            results[label] = round(ppl, 3)
            print(f"    → PPL = {ppl:.3f}")
            del model

        # ── Uniform INT8 ──────────────────────────────────────────────────
        print(f"\n  [uniform_int8]")
        model = load_fresh_model(hf_id)
        model = nncf.compress_weights(
            model,
            mode          = nncf.CompressWeightsMode.INT8_SYM,
            group_size    = -1,
            ignored_scope = LINEAR_ONLY_SCOPE,
        )
        ppl = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["uniform_int8"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        # ── Uniform INT4 ──────────────────────────────────────────────────
        print(f"\n  [uniform_int4]")
        model = load_fresh_model(hf_id)
        model = nncf.compress_weights(
            model,
            mode          = nncf.CompressWeightsMode.INT4_SYM,
            ratio         = 1.0,
            group_size    = -1,
            ignored_scope = LINEAR_ONLY_SCOPE,
        )
        ppl = compute_perplexity(model, tokenizer, test_text, SEQ_LEN)
        results["uniform_int4"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        all_results[model_name] = results

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Saved → {OUTPUT_JSON}")
    for model_name, results in all_results.items():
        print(f"\n  {model_name}:")
        for k, v in results.items():
            print(f"    {k:<20} {v:>8.3f}")


if __name__ == "__main__":
    main()
