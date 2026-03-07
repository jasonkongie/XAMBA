"""
eval_perplexity.py  —  CPU Pipeline

Evaluates WikiText-2 perplexity by loading the actual OpenVINO IR models
(.xml / .bin) produced by quantize_mixed.py and quantize_uniform.py.

Configs evaluated per model:
  - FP16 baseline   : {model}.xml
  - mixed-precision : {model}_{METRIC_TAG}_point01.xml  ..  point10.xml
  - uniform INT8    : {model}_uniform_int8.xml
  - uniform INT4    : {model}_uniform_int4.xml

Output:
    perplexity_results_{METRIC_TAG}.json  —  {model_name: {point: ppl}}

Usage:
    python eval_perplexity.py
"""

import os
import json
import torch
import torch.nn.functional as F
import openvino as ov
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mamba-130m-hf": {
        "hf_id": "state-spaces/mamba-130m-hf",
    },
    "mamba-1.4b-hf": {
        "hf_id": "state-spaces/mamba-1.4b-hf",
    },
}

OV_DIR      = "ov_models"
DEVICE      = "CPU"
SEQ_LEN     = 2048
MAX_WINDOWS = 20
N_POINTS    = 10

# ── Sensitivity metric ────────────────────────────────────────────────────────
# Must match the METRIC_TAG used when running quantize_mixed.py
SENSITIVITY_METRIC = "kl_student_to_teacher"
METRIC_TAG         = "kl" if SENSITIVITY_METRIC == "kl_student_to_teacher" else "sqnr"
OUTPUT_JSON        = f"perplexity_results_{METRIC_TAG}.json"

# ── Perplexity ───────────────────────────────────────────────────────────────

def compute_perplexity(compiled_model, tokenizer, text, seq_len=2048):
    """Non-overlapping windows over the OV compiled model. No gradients needed."""
    enc       = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids.numpy()        # OV expects numpy
    N         = input_ids.shape[1]
    n_windows = min(MAX_WINDOWS, max(1, (N - 1) // seq_len))
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    nlls = []
    for i in tqdm(range(n_windows), desc="    eval", leave=False):
        s     = i * seq_len
        e     = min(s + seq_len + 1, N)
        chunk = input_ids[:, s:e]
        if chunk.shape[1] < 2:
            continue

        inp = chunk[:, :-1]
        tgt = torch.from_numpy(chunk[:, 1:]).long()

        result = compiled_model({"input_ids": inp})
        logits = torch.from_numpy(result[0])   # result[0] = first output = logits

        loss = F.cross_entropy(
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

    core        = ov.Core()
    all_results = {}

    for model_name, cfg in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  (device: {DEVICE})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
        results   = {}

        # Build ordered list of (label, xml_path) to evaluate
        configs = []

        # FP16 baseline (XAMBA OV export)
        p = os.path.join(OV_DIR, f"{model_name}.xml")
        if os.path.exists(p):
            configs.append(("baseline_fp16", p))
        else:
            print(f"  [!] Baseline not found: {p}  — run convert.py first")

        # Mixed-precision points
        for i in range(1, N_POINTS + 1):
            p = os.path.join(OV_DIR, f"{model_name}_{METRIC_TAG}_point{i:02d}.xml")
            if os.path.exists(p):
                configs.append((f"{METRIC_TAG}_point{i:02d}", p))

        # Uniform endpoints
        for suffix in ["uniform_int8", "uniform_int4"]:
            p = os.path.join(OV_DIR, f"{model_name}_{suffix}.xml")
            if os.path.exists(p):
                configs.append((suffix, p))

        for label, path in configs:
            print(f"\n  [{label}]  {os.path.basename(path)}")
            ov_model   = core.read_model(path)
            # The model may have been exported with a static benchmark shape.
            # Reshape to our eval seq_len before compilation.
            input_name = ov_model.input(0).any_name
            ov_model.reshape({input_name: [1, SEQ_LEN]})
            compiled   = core.compile_model(ov_model, DEVICE)
            ppl        = compute_perplexity(compiled, tokenizer, test_text, SEQ_LEN)
            results[label] = round(ppl, 3)
            print(f"    → PPL = {ppl:.3f}")
            del compiled

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
