"""
eval_perplexity_gpu.py  —  GPU Pipeline

Evaluates WikiText-2 perplexity using NNCF compress_weights applied directly
to the HuggingFace PyTorch model, matching the GPU quantization (INT8/FP16)
used by quantize_mixed_gpu.py.  This avoids the static-shape limitation of
the OV export (which is sized for benchmark latency, not evaluation seq_len).

Configs evaluated per model:
  - FP32 baseline   : unquantized HuggingFace model
  - mixed-precision : NNCF INT8 per sensitivity ranking (8 points for mamba2)
  - uniform INT8    : NNCF INT8_SYM on all linear layers

Note: SEQ_LEN=256 to match mamba2 chunk_size requirement.

Output:
    perplexity_results_gpu_{METRIC_TAG}.json  —  {model_name: {point: ppl}}

Usage:
    python eval_perplexity_gpu.py
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
    "mamba2_b_1_t_4": {
        "hf_id":            "yuji96/mamba2-130m-hf",
        "tokenizer_id":     "state-spaces/mamba-130m-hf",
        "sensitivity_8bit": "sensitivity_results_mamba2-130m_8bits_XAMBA.json",
        "n_points":         8,    # matches GPU point cap (point09/10 fail OV GPU compile)
        "seq_len":          256,  # must match mamba2 chunk_size
    },
}

MAX_WINDOWS = 20

# ── Sensitivity metric ────────────────────────────────────────────────────────
SENSITIVITY_METRIC = "kl_student_to_teacher"
METRIC_TAG         = "kl" if SENSITIVITY_METRIC == "kl_student_to_teacher" else "sqnr"
OUTPUT_JSON        = f"perplexity_results_gpu_{METRIC_TAG}.json"

# Exclude Conv1d (SSM conv1d) — matches OV pipeline's IgnoredScope(types=["Convolution"])
LINEAR_ONLY_SCOPE = nncf.IgnoredScope(types=["Conv1d"], validate=False)

# ── Per-layer quantization ────────────────────────────────────────────────────

def quantize_weight(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Per-output-channel symmetric absmax quantization.
    Matches NNCF compress_weights INT8_SYM with group_size=-1.
    Used for mixed-precision points where a per-layer approach is needed.
    """
    q_max  = 2 ** (n_bits - 1) - 1
    scales = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return (w / scales).round().clamp(-q_max, q_max) * scales

# ── Sensitivity helpers ───────────────────────────────────────────────────────

def load_sensitivity_8bit(path):
    with open(path) as f:
        data = json.load(f)
    layers = [(name, stats[SENSITIVITY_METRIC]) for name, stats in data.items()]
    reverse = (SENSITIVITY_METRIC == "sqnr_db")
    layers.sort(key=lambda x: x[1], reverse=reverse)
    return layers

def compute_cutoff_indices(n_entries, n_points=10):
    segment_size = (n_entries - 1) // n_points
    return [segment_size * i for i in range(1, n_points + 1)]

# ── Model loading ─────────────────────────────────────────────────────────────

def load_fresh_model(hf_id):
    config = AutoConfig.from_pretrained(hf_id)
    config.use_cache = False
    return AutoModelForCausalLM.from_pretrained(
        hf_id, config=config, trust_remote_code=True
    ).eval()

# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, seq_len):
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
        print(f"  Model: {model_name}  (GPU pipeline — INT8/FP16)")
        print(f"{'='*60}")

        hf_id        = cfg["hf_id"]
        tokenizer_id = cfg.get("tokenizer_id", hf_id)
        tokenizer    = AutoTokenizer.from_pretrained(tokenizer_id)
        seq_len      = cfg["seq_len"]
        n_points     = cfg["n_points"]
        sensitivity  = load_sensitivity_8bit(cfg["sensitivity_8bit"])
        indices      = compute_cutoff_indices(len(sensitivity), n_points)
        results      = {}

        # ── Baseline (FP32) ───────────────────────────────────────────────
        print("\n  [baseline_fp32]")
        model = load_fresh_model(hf_id)
        ppl   = compute_perplexity(model, tokenizer, test_text, seq_len)
        results["baseline_fp32"] = round(ppl, 3)
        print(f"    → PPL = {ppl:.3f}")
        del model

        # ── GPU mixed-precision points (INT8 / FP16) ──────────────────────
        for point_idx, cutoff in enumerate(indices):
            label      = f"gpu_{METRIC_TAG}_point{point_idx + 1:02d}"
            int8_names = [name for name, _ in sensitivity[:cutoff]]
            fp16_names = [name for name, _ in sensitivity[cutoff:]]
            print(f"\n  [{label}] cutoff {cutoff}/{len(sensitivity)}  "
                  f"INT8:{len(int8_names)}  FP16:{len(fp16_names)}")
            model = load_fresh_model(hf_id)
            for name, module in model.named_modules():
                if name in int8_names and hasattr(module, "weight"):
                    module.weight.data = quantize_weight(module.weight.data, n_bits=8)
            ppl = compute_perplexity(model, tokenizer, test_text, seq_len)
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
        ppl = compute_perplexity(model, tokenizer, test_text, seq_len)
        results["uniform_int8"] = round(ppl, 3)
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
            print(f"    {k:<28} {v:>8.3f}")


if __name__ == "__main__":
    main()
