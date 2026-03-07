"""
quantize_mixed_gpu.py  —  GPU Pipeline

INT8/FP16 mixed-precision quantization using 8-bit KL sensitivity only.
GPU does not support INT4, so this pipeline uses only INT8_SYM for
quantized layers and keeps sensitive layers in FP16.

Algorithm:
  1. Load 8-bit sensitivity data, sort ascending by KL.
  2. Pick 10 evenly-spaced cutoff points.
  3. At cutoff i, entries 0..cutoff are quantized to INT8, rest stay FP16.
  4. Single-pass NNCF: compress_weights(mode=INT8_SYM, ignored_scope=fp16_layers).

Output:  ov_models/{model_name}_gpu_point{01..10}.xml

Usage:
    python quantize_mixed_gpu.py
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
        "sensitivity_8bit": "mamba130m_sensitivity_results_8bits.json",
    },
    "mamba2_b_1_t_4": {
        "hf_id": "yuji96/mamba2-130m-hf",
        "sensitivity_8bit": "sensitivity_results_mamba2-130m_8bits_XAMBA.json",
    },
}

N_POINTS   = 10
OUTPUT_DIR = "ov_models"

# ── Sensitivity (8-bit only) ─────────────────────────────────────────────────

def load_sensitivity_8bit(path):
    """Load 8-bit KL sensitivity, sorted ascending by kl_student_to_teacher."""
    with open(path) as f:
        data = json.load(f)
    layers = [(name, stats["kl_student_to_teacher"]) for name, stats in data.items()]
    layers.sort(key=lambda x: x[1])
    return layers


def compute_cutoff_indices(n_entries, n_points=10):
    # Protect last entry (most sensitive layer) — never quantize it
    safe_max = n_entries - 1
    return [safe_max * i // n_points for i in range(1, n_points + 1)]

# ── IR Name Mapping ──────────────────────────────────────────────────────────

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

# ── Single-Pass Quantization ─────────────────────────────────────────────────

# XAMBA CumBA MatMul ops (cumsum replacement) have no INT8 GPU kernel — always keep FP16
XAMBA_MATMUL_PATTERN = r".*/mixer/MatMul.*"

def quantize_gpu_point(core, input_model_path, int8_layers, fp16_layers, output_path):
    """
    Single-pass NNCF: INT8_SYM for quantized layers, FP16 for the rest.
    XAMBA CumBA MatMul ops are always excluded (no INT8 GPU kernel for them).
    """
    print(f"    INT8: {len(int8_layers)}  |  FP16: {len(fp16_layers)}")

    ov_model = core.read_model(input_model_path)

    if int8_layers:
        ignore_patterns = build_ignore_patterns(fp16_layers)
        # Always exclude XAMBA CumBA MatMul ops regardless of sensitivity ranking
        ignore_patterns.append(XAMBA_MATMUL_PATTERN)
        ignored = nncf.IgnoredScope(patterns=ignore_patterns, validate=False) if ignore_patterns else None

        kwargs = dict(
            model      = ov_model,
            mode       = nncf.CompressWeightsMode.INT8_SYM,
            group_size = -1,
        )
        if ignored:
            kwargs["ignored_scope"] = ignored

        ov_model = nncf.compress_weights(**kwargs)

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
        print(f"  Model: {model_name}  (GPU pipeline — INT8/FP16)")
        print(f"{'='*60}")

        sensitivity = load_sensitivity_8bit(cfg["sensitivity_8bit"])
        all_layer_names = [name for name, _ in sensitivity]
        print(f"  Sensitivity list: {len(sensitivity)} layers")

        indices = compute_cutoff_indices(len(sensitivity), N_POINTS)
        print(f"  Cutoff indices: {indices}")

        for point_idx, cutoff in enumerate(indices):
            point_name = f"gpu_point{point_idx + 1:02d}"
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{point_name}.xml")

            print(f"\n  ── {point_name} (cutoff {cutoff}/{len(sensitivity)}) ──")

            int8_layers = [name for name, _ in sensitivity[:cutoff]]
            fp16_layers = [name for name, _ in sensitivity[cutoff:]]

            quantize_gpu_point(core, input_model, int8_layers, fp16_layers, output_path)

    print(f"\n{'='*60}")
    print("Done! All GPU mixed-precision models generated.")


if __name__ == "__main__":
    main()
