"""
eval_perplexity.py

Measures WikiText-2 perplexity for Mamba2-130m at each Pareto quantization point.

Strategy
--------
We apply nncf.compress_weights() to the *PyTorch* Mamba2ForCausalLM model using the
same sensitivity thresholds as quantize_nncf.py. This is equivalent to the OV model
quality because NNCF weight compression is a pure weight transformation — the same
INT4_SYM compressed weights end up in both the PyTorch and OpenVINO versions.

Evaluation uses the WikiText-2 **test** split (the standard split used in GPTQ,
LLaMA, SqueezeLLM, etc.) with non-overlapping windows of SEQ_LEN tokens.

Usage
-----
    cd /Users/jasonkong/Documents/XAMBA
    python eval_perplexity.py

Output
------
    perplexity_results.json   — JSON dict {point_name: ppl}
"""

import json
import re
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Mamba2ForCausalLM, Mamba2Config
import nncf
from tqdm import tqdm

# ── Configuration ───────────────────────────────────────────────────────────

MODEL_HF         = "yuji96/mamba2-130m-hf"
SENSITIVITY_FILE = "sensitivity_results_mamba2-130m_4bits.json"
SEQ_LEN          = 2048      # tokens per evaluation window (standard for LLM PPL papers)
OUTPUT_JSON      = "perplexity_results.json"

# Must match quantize_nncf.py exactly
PARETO_POINTS = {
    "baseline": None,   # FP32 — no compression
    "pointA":   1.0,    # Conservative  — 9  INT4 layers
    "pointB":   2.0,    # Moderate      — 26 INT4 layers
    "pointC":   2.35,   # Aggressive    — 37 INT4 layers
    "pointD":   7.0,    # Maximum       — 43 INT4 layers
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_sensitivity(path):
    """Load KL sensitivity data sorted ascending by kl_student_to_teacher."""
    with open(path) as f:
        data = json.load(f)
    layers = [(name, v["kl_student_to_teacher"]) for name, v in data.items()]
    layers.sort(key=lambda x: x[1])
    return layers


def get_fp16_layers(sensitivity, kl_threshold):
    """Layers with KL >= threshold are too sensitive — keep them FP16."""
    return [name for name, kl in sensitivity if kl >= kl_threshold]


def load_fresh_model():
    """Load Mamba2ForCausalLM with use_cache=False (full-sequence eval mode)."""
    config = Mamba2Config.from_pretrained(MODEL_HF)
    config.use_cache = False
    model = Mamba2ForCausalLM.from_pretrained(MODEL_HF, config=config)
    return model.eval()


def apply_nncf(model, sensitivity, kl_threshold):
    """
    Apply NNCF INT4_SYM weight compression with the same pareto logic as
    quantize_nncf.py.

    For PyTorch NNCF the layer names in the sensitivity file map *directly*
    to module paths (no backbone→IR transformation needed).
    """
    fp16_layers = get_fp16_layers(sensitivity, kl_threshold)
    n_int4      = len(sensitivity) - len(fp16_layers)
    print(f"  INT4 layers : {n_int4} / {len(sensitivity)}")
    print(f"  FP16 layers : {len(fp16_layers)}")

    # Build regex patterns — sensitivity file already uses PyTorch names
    patterns = [f".*{re.escape(n)}.*" for n in fp16_layers]
    ignored  = nncf.IgnoredScope(patterns=patterns, validate=False) if patterns else None

    kwargs = dict(
        model      = model,
        mode       = nncf.CompressWeightsMode.INT4_SYM,
        ratio      = 1.0,
        group_size = -1,   # per-channel — matches quantize_nncf.py
    )
    if ignored:
        kwargs["ignored_scope"] = ignored

    compressed = nncf.compress_weights(**kwargs)
    return compressed.eval()


@torch.no_grad()
def compute_perplexity(model, tokenizer, text, seq_len=2048):
    """
    Compute perplexity on `text` using non-overlapping windows of `seq_len` tokens.
    Standard methodology from GPTQ / LLaMA / SqueezeLLM papers.
    """
    enc       = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids                          # [1, N]
    N         = input_ids.shape[1]
    n_windows = max(1, (N - 1) // seq_len)
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    print(f"  Tokens: {N:,}  |  Windows: {n_windows}  |  seq_len: {seq_len}")

    nlls = []
    for i in tqdm(range(n_windows), desc="  evaluating", leave=False):
        s      = i * seq_len
        e      = min(s + seq_len + 1, N)      # +1 for target shift
        chunk  = input_ids[:, s:e]
        if chunk.shape[1] < 2:
            continue

        inp    = chunk[:, :-1]                # [1, T]
        tgt    = chunk[:,  1:]                # [1, T]

        logits = model(input_ids=inp).logits  # [1, T, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            ignore_index=pad_id,
        )
        nlls.append(loss.item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Check prerequisites
    if not os.path.exists(SENSITIVITY_FILE):
        print(f"Error: {SENSITIVITY_FILE} not found.")
        return

    print(f"Loading sensitivity: {SENSITIVITY_FILE}")
    sensitivity = load_sensitivity(SENSITIVITY_FILE)
    print(f"  {len(sensitivity)} layers, "
          f"KL range: {sensitivity[0][1]:.2f} – {sensitivity[-1][1]:.2f}")

    print(f"\nLoading tokenizer from {MODEL_HF} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

    print("Loading WikiText-2 test set …")
    raw       = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(t for t in raw["text"] if t.strip())
    n_chars   = len(test_text)
    n_tokens  = tokenizer(test_text, return_tensors="pt", truncation=False).input_ids.shape[1]
    print(f"  {n_chars:,} chars  →  {n_tokens:,} tokens")

    results = {}

    for point_name, kl_thr in PARETO_POINTS.items():
        print(f"\n{'='*55}")
        print(f"  Point: {point_name}   (KL threshold: {kl_thr})")
        print(f"{'='*55}")

        model = load_fresh_model()

        if kl_thr is not None:
            model = apply_nncf(model, sensitivity, kl_thr)
        else:
            print("  Baseline: no compression (FP32)")

        model.eval()

        ppl = compute_perplexity(model, tokenizer, test_text, seq_len=SEQ_LEN)
        results[point_name] = round(ppl, 3)
        print(f"\n  → Perplexity: {ppl:.3f}")

        del model   # free RAM before next model

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  WikiText-2 Perplexity  (seq_len={SEQ_LEN}, test split)")
    print(f"{'='*55}")
    print(f"  {'Model':<12}  {'PPL':>8}")
    print(f"  {'-'*22}")
    for k, v in results.items():
        print(f"  {k:<12}  {v:>8.3f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
