"""
quantize_mixed_gpu_int4.py  —  GPU INT4/INT8/FP16 Pipeline (mamba2_b_1_t_4 only)

Mixed INT4/INT8/FP16 quantization using merged 4+8 bit KL sensitivity from the
XAMBA-specific sensitivity files.

Algorithm:
  1. Merge 4-bit and 8-bit sensitivity, sort ascending by KL.
  2. Pick 10 evenly-spaced cutoff points.
     The last entry (most sensitive layer) is ALWAYS protected — never quantized.
  3. At cutoff i, entries S[0:cutoff] get their assigned bit (4 or 8), rest FP16.
  4. Two-pass NNCF:
       Pass 1: INT4_SYM  — ignore int8_layers + fp16_layers + XAMBA_MatMul
       Pass 2: INT8_SYM  — ignore int4_layers + fp16_layers + XAMBA_MatMul
     XAMBA CumBA MatMul ops always excluded (no INT4/INT8 GPU kernel).

Output:
    ov_models/mamba2_b_1_t_4_gpu_int4_point{01..10}.xml
    ov_models/mamba2_b_1_t_4_gpu_uniform_int4.xml

Usage:
    python quantize_mixed_gpu_int4.py
"""

import os
import re
import json
import openvino as ov
import nncf

# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mamba2_b_1_t_4": {
        "hf_id": "yuji96/mamba2-130m-hf",
        "sensitivity_4bit": "sensitivity_results_mamba2-130m_4bits_XAMBA.json",
        "sensitivity_8bit": "sensitivity_results_mamba2-130m_8bits_XAMBA.json",
    },
}

N_POINTS   = 10
OUTPUT_DIR = "ov_models"

# XAMBA CumBA MatMul ops — replaced cumsum, no INT4/INT8 GPU kernel, always FP16
XAMBA_MATMUL_PATTERN = r".*/mixer/MatMul.*"

# ── Sensitivity metric ────────────────────────────────────────────────────────
# "sqnr_db"             → higher SQNR = less sensitive = quantize first (sort DESC)
# "kl_student_to_teacher" → lower KL  = less sensitive = quantize first (sort ASC)
SENSITIVITY_METRIC = "sqnr_db"

# ── Sensitivity ───────────────────────────────────────────────────────────────

def build_sensitivity_list(path4, path8):
    """
    Merge 4-bit and 8-bit KL files into [(layer, bit, kl), ...] sorted ASC.
    Same layer appears twice (once per bit-width).
    Last-wins: 8-bit entry comes before 4-bit in the sorted list for most layers,
    so the 4-bit assignment overwrites 8-bit as we include more entries.
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
    """Last-wins: later 4-bit entry overwrites earlier 8-bit for the same layer."""
    assignment = {}
    for layer, bit, kl in S[:cutoff_idx]:
        assignment[layer] = bit
    return assignment

# ── IR Name Mapping ───────────────────────────────────────────────────────────

def pt_to_ir_path(pytorch_name):
    name = pytorch_name.removeprefix("backbone.")
    name = re.sub(r"\.([a-zA-Z_])", r"/\1", name)
    return name


def build_ignore_patterns(pytorch_layer_names):
    patterns = []
    for name in pytorch_layer_names:
        ir_path = pt_to_ir_path(name)
        patterns.append(f".*{re.escape(ir_path)}.*")
    return patterns

# ── Two-Pass Quantization ─────────────────────────────────────────────────────

def quantize_gpu_int4_point(core, input_model_path, int4_layers, int8_layers,
                             fp16_layers, output_path):
    """
    Two-pass NNCF: INT4_SYM first, then INT8_SYM.
    XAMBA CumBA MatMul always excluded from both passes.
    """
    print(f"    INT4: {len(int4_layers)}  |  INT8: {len(int8_layers)}  |  FP16: {len(fp16_layers)}")

    ov_model = core.read_model(input_model_path)

    # ── Pass 1: INT4_SYM ─────────────────────────────────────────────────────
    if int4_layers:
        ignore_for_int4 = build_ignore_patterns(list(int8_layers) + list(fp16_layers))
        ignore_for_int4.append(XAMBA_MATMUL_PATTERN)
        ignored_int4 = nncf.IgnoredScope(patterns=ignore_for_int4, validate=False)
        ov_model = nncf.compress_weights(
            model         = ov_model,
            mode          = nncf.CompressWeightsMode.INT4_SYM,
            ratio         = 1.0,
            group_size    = -1,
            ignored_scope = ignored_int4,
        )

    # ── Pass 2: INT8_SYM ─────────────────────────────────────────────────────
    # int4_layers already compressed — ignore them + fp16_layers + XAMBA MatMul
    if int8_layers:
        ignore_for_int8 = build_ignore_patterns(list(int4_layers) + list(fp16_layers))
        ignore_for_int8.append(XAMBA_MATMUL_PATTERN)
        ignored_int8 = nncf.IgnoredScope(patterns=ignore_for_int8, validate=False)
        ov_model = nncf.compress_weights(
            model         = ov_model,
            mode          = nncf.CompressWeightsMode.INT8_SYM,
            group_size    = -1,
            ignored_scope = ignored_int8,
        )

    ov.save_model(ov_model, output_path, compress_to_fp16=True)

    bin_path = output_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size_mb = os.path.getsize(bin_path) / (1024 * 1024)
        print(f"    Saved: {output_path}  ({size_mb:.1f} MB)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    core = ov.Core()

    for model_name, cfg in MODEL_REGISTRY.items():
        input_model = os.path.join(OUTPUT_DIR, f"{model_name}.xml")
        if not os.path.exists(input_model):
            print(f"[!] Baseline model not found: {input_model}  — run convert.py first")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  (GPU INT4/INT8/FP16 pipeline)")
        print(f"{'='*60}")

        S = build_sensitivity_list(cfg["sensitivity_4bit"], cfg["sensitivity_8bit"])
        all_unique_layers = list(set(l for l, _, _ in S))
        print(f"  Sensitivity list: {len(S)} entries  |  {len(all_unique_layers)} unique layers")
        print(f"  Protected (most sensitive): '{S[-1][0]}'  {SENSITIVITY_METRIC}={S[-1][2]:.4f}")

        indices = compute_cutoff_indices(len(S), N_POINTS)
        print(f"  Cutoff indices (safe_max={len(S)-1}): {indices}")

        # ── Mixed-precision points ────────────────────────────────────────
        for point_idx, cutoff in enumerate(indices):
            point_name  = f"gpu_int4_point{point_idx + 1:02d}"
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{point_name}.xml")

            print(f"\n  ── {point_name} (cutoff {cutoff}/{len(S)-1}) ──")

            assignment  = get_layer_assignments(S, cutoff)
            int4_layers = {l for l, b in assignment.items() if b == 4}
            int8_layers = {l for l, b in assignment.items() if b == 8}
            fp16_layers = set(all_unique_layers) - int4_layers - int8_layers

            quantize_gpu_int4_point(
                core, input_model,
                int4_layers, int8_layers, fp16_layers,
                output_path,
            )

        # ── Uniform INT4 baseline ─────────────────────────────────────────
        print(f"\n  ── uniform_int4 (linear layers only, XAMBA MatMul excluded) ──")
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}_gpu_uniform_int4.xml")
        ignored_uniform = nncf.IgnoredScope(
            patterns=[XAMBA_MATMUL_PATTERN],
            types=["Convolution"],     # exclude conv1d layers
            validate=False,
        )
        ov_model = core.read_model(input_model)
        ov_model = nncf.compress_weights(
            model         = ov_model,
            mode          = nncf.CompressWeightsMode.INT4_SYM,
            ratio         = 1.0,
            group_size    = -1,
            ignored_scope = ignored_uniform,
        )
        ov.save_model(ov_model, output_path, compress_to_fp16=True)
        bin_path = output_path.replace(".xml", ".bin")
        if os.path.exists(bin_path):
            size_mb = os.path.getsize(bin_path) / (1024 * 1024)
            print(f"    Saved: {output_path}  ({size_mb:.1f} MB)")

    print(f"\n{'='*60}")
    print("Done! GPU INT4/INT8/FP16 mixed-precision models generated.")


if __name__ == "__main__":
    main()
