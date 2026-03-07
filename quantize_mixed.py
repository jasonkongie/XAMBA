"""
quantize_mixed.py  —  CPU Pipeline

Mixed-precision INT4/INT8/FP16 quantization using merged 4-bit + 8-bit
KL-divergence sensitivity data.

Algorithm (adapted from pareto.py):
  1. Merge 4-bit and 8-bit KL sensitivity into a single sorted list S.
     Each layer appears twice (once per bit-width), sorted ascending by KL.
  2. Pick 10 evenly-spaced cutoff points through S.
  3. At cutoff i, entries S[0:cutoff] are quantized at their assigned bit (4 or 8).
     Last-wins semantics: if a layer has both an 8-bit entry and a later 4-bit
     entry within the cutoff, the 4-bit assignment overwrites the 8-bit.
  4. Two-pass NNCF compression on the OV model:
       Pass 1: INT4_SYM for layers assigned 4-bit
       Pass 2: INT8_SYM for layers assigned 8-bit
     Remaining layers stay FP16/FP32.

Output:  ov_models/{model_name}_point{01..10}.xml

Usage:
    python quantize_mixed.py
"""

import os
import re
import json
import openvino as ov
import nncf

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

N_POINTS  = 10
OUTPUT_DIR = "ov_models"

# ── Sensitivity metric ────────────────────────────────────────────────────────
# "sqnr_db"             → higher SQNR = less sensitive = quantize first (sort DESC)
# "kl_student_to_teacher" → lower KL  = less sensitive = quantize first (sort ASC)
SENSITIVITY_METRIC = "sqnr_db"
METRIC_TAG         = "sqnr" if SENSITIVITY_METRIC == "sqnr_db" else "kl"
# Output filenames will include METRIC_TAG, e.g. mamba-130m-hf_sqnr_point01.xml

# ── Sensitivity ──────────────────────────────────────────────────────────────

def build_sensitivity_list(path4, path8):
    """
    Merge 4-bit and 8-bit KL files into
        [(layer, bit, kl_value), …]  sorted ASC by kl_value
    Same layer appears twice (once per bit-width).
    Adapted from pareto.py build_sensitivity_list().
    """
    sens4 = json.load(open(path4))
    sens8 = json.load(open(path8))

    merged = []
    for layer, stats in sens4.items():
        merged.append((layer, 4, stats[SENSITIVITY_METRIC]))
    for layer, stats in sens8.items():
        merged.append((layer, 8, stats[SENSITIVITY_METRIC]))

    # SQNR: high = less sensitive → sort DESC (index 0 = safest to quantize)
    # KL:   low  = less sensitive → sort ASC  (index 0 = safest to quantize)
    reverse = (SENSITIVITY_METRIC == "sqnr_db")
    return sorted(merged, key=lambda t: t[2], reverse=reverse)


def compute_cutoff_indices(n_entries, n_points=10):
    """
    Divide the sensitivity list into n_points equal segments.
    Only the first (n_entries - 1) entries are partitioned — the last
    entry (most sensitive layer) is always excluded / never quantized.
    segment_size = (n_entries - 1) // n_points
    cutoffs      = [segment_size * i  for i in 1..n_points]
    """
    segment_size = (n_entries - 1) // n_points
    return [segment_size * i for i in range(1, n_points + 1)]


def get_layer_assignments(S, cutoff_idx):
    """
    At cutoff_idx, entries S[0:cutoff_idx] are quantized.
    Returns {layer_name: bit_width} using last-wins semantics.
    (8-bit entries typically appear before 4-bit for the same layer,
     so later 4-bit entries overwrite 8-bit as we include more entries.)
    """
    assignment = {}
    for layer, bit, kl in S[:cutoff_idx]:
        assignment[layer] = bit      # last occurrence wins
    return assignment

# ── IR Name Mapping (from quantize_nncf.py) ──────────────────────────────────

def pt_to_ir_path(pytorch_name):
    """
    Convert PyTorch layer name to OpenVINO IR path fragment.
      backbone.layers.0.mixer.out_proj  →  layers.0/mixer/out_proj
    """
    name = pytorch_name.removeprefix("backbone.")
    name = re.sub(r"\.([a-zA-Z_])", r"/\1", name)
    return name


def build_ignore_patterns(pytorch_layer_names):
    """Build regex patterns for NNCF IgnoredScope from PyTorch names."""
    patterns = []
    for name in pytorch_layer_names:
        ir_path = pt_to_ir_path(name)
        patterns.append(f".*{re.escape(ir_path)}.*")
    return patterns

# ── Two-Pass Quantization ────────────────────────────────────────────────────

def quantize_point(core, input_model_path, assignment, all_layer_names, output_path):
    """
    Two-pass NNCF: INT4 first, then INT8.

    After the INT4 pass, those weights become compressed INT4 constants.
    The INT8 pass only finds remaining FP32 weight nodes.
    We explicitly ignore INT4 layers in the INT8 pass for safety.
    """
    int4_layers = {l for l, b in assignment.items() if b == 4}
    int8_layers = {l for l, b in assignment.items() if b == 8}
    fp16_layers = set(all_layer_names) - int4_layers - int8_layers

    n4, n8, nf = len(int4_layers), len(int8_layers), len(fp16_layers)
    print(f"    INT4: {n4}  |  INT8: {n8}  |  FP16: {nf}")

    ov_model = core.read_model(input_model_path)

    # ── Pass 1: INT4_SYM ─────────────────────────────────────────────────
    if int4_layers:
        non_int4 = int8_layers | fp16_layers
        ignore_4 = build_ignore_patterns(non_int4)
        ignored_4 = nncf.IgnoredScope(patterns=ignore_4, validate=False)
        ov_model = nncf.compress_weights(
            model       = ov_model,
            mode        = nncf.CompressWeightsMode.INT4_SYM,
            ratio       = 1.0,
            group_size  = -1,
            ignored_scope = ignored_4,
        )

    # ── Pass 2: INT8_SYM ─────────────────────────────────────────────────
    if int8_layers:
        non_int8 = int4_layers | fp16_layers
        ignore_8 = build_ignore_patterns(non_int8)
        ignored_8 = nncf.IgnoredScope(patterns=ignore_8, validate=False)
        ov_model = nncf.compress_weights(
            model       = ov_model,
            mode        = nncf.CompressWeightsMode.INT8_SYM,
            group_size  = -1,
            ignored_scope = ignored_8,
        )

    ov.save_model(ov_model, output_path, compress_to_fp16=True)

    bin_path = output_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size_mb = os.path.getsize(bin_path) / (1024 * 1024)
        print(f"    Saved: {output_path}  ({size_mb:.1f} MB)")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    core = ov.Core()

    for model_name, cfg in MODEL_REGISTRY.items():
        input_model = os.path.join(OUTPUT_DIR, f"{model_name}.xml")
        if not os.path.exists(input_model):
            print(f"[!] Baseline model not found: {input_model}  — run convert.py first")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        # Build merged sensitivity list
        S = build_sensitivity_list(cfg["sensitivity_4bit"], cfg["sensitivity_8bit"])
        all_layer_names = list(set(l for l, _, _ in S))   # unique layer names
        print(f"  Merged sensitivity list: {len(S)} entries ({len(all_layer_names)} unique layers)")

        # Compute 10 cutoff points
        indices = compute_cutoff_indices(len(S), N_POINTS)
        print(f"  Cutoff indices: {indices}")

        for point_idx, cutoff in enumerate(indices):
            point_name = f"{METRIC_TAG}_point{point_idx + 1:02d}"
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{point_name}.xml")

            print(f"\n  ── {point_name} (cutoff {cutoff}/{len(S)}) ──")

            assignment = get_layer_assignments(S, cutoff)
            quantize_point(core, input_model, assignment, all_layer_names, output_path)

    print(f"\n{'='*60}")
    print("Done! All mixed-precision CPU models generated.")
    print(f"  Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
