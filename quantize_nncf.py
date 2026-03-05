"""
NNCF weight compression for Mamba-2 on NPU.

Applies mixed-precision INT4/FP16 quantization to a confirmed-working OpenVINO IR
model using KL-divergence sensitivity data to decide which layers to compress.

Usage:
    python quantize_nncf.py

Prerequisites:
    - A working OV IR model in ov_model/ (produced by convert.py)
    - sensitivity_results_mamba2-130m_4bits.json (from KL_Sensitivity_Analysis)
    - pip install nncf
"""

import json
import os
import re
import openvino as ov
import nncf

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_MODEL = "ov_models/mamba2-130m-hf.xml"
SENSITIVITY_FILE = "sensitivity_results_mamba2-130m_4bits.json"
OUTPUT_DIR = "ov_models"

# 4 Pareto points: KL threshold → layers with KL >= threshold stay FP16
PARETO_POINTS = {
    "pointA": 1.0,     # Conservative: only 9 safest layers quantized
    "pointB": 2.0,     # Moderate: ~26 layers quantized
    "pointC": 2.35,    # Aggressive: ~37 layers quantized
    "pointD": 7.0,     # Maximum: ~44 layers quantized
}


def load_sensitivity(path):
    """Load KL sensitivity data sorted by kl_student_to_teacher ascending."""
    with open(path) as f:
        data = json.load(f)
    layers = []
    for name, metrics in data.items():
        layers.append((name, metrics["kl_student_to_teacher"]))
    layers.sort(key=lambda x: x[1])
    return layers


def get_layers_to_keep_fp16(sensitivity, kl_threshold):
    """Return PyTorch layer names with KL >= threshold (too sensitive to quantize)."""
    return [name for name, kl in sensitivity if kl >= kl_threshold]


def pt_to_ir_path(pytorch_name):
    """
    Convert a PyTorch layer name to the path fragment used in OV IR friendly names.

    PyTorch:  'backbone.layers.0.mixer.out_proj'
    OV IR:    '/layers.0/mixer/out_proj/MatMul'

    Transformation:
      1. Strip leading 'backbone.' prefix (the top-level model wrapper is elided)
      2. Replace '.<letter>' separators with '/<letter>' (dots before digits stay,
         e.g. 'layers.0' remains 'layers.0' because OV keeps ModuleList indices)
    """
    name = pytorch_name.removeprefix("backbone.")
    name = re.sub(r"\.([a-zA-Z_])", r"/\1", name)
    return name


def build_ignore_patterns(pytorch_layer_names):
    """
    Build NNCF IgnoredScope regex patterns from PyTorch layer names.

    NNCF matches patterns against its internal NNCFGraph node names using
    re.fullmatch(), so we wrap each IR path fragment with '.*' on both sides
    for substring matching.  re.escape() protects literal dots in 'layers.N'.

    Example: 'backbone.layers.0.mixer.out_proj'
        → ir_path: 'layers.0/mixer/out_proj'
        → pattern: '.*layers\\.0/mixer/out_proj.*'
        matches:   '/layers.0/mixer/out_proj/MatMul'
    """
    patterns = []
    for name in pytorch_layer_names:
        ir_path = pt_to_ir_path(name)
        patterns.append(f".*{re.escape(ir_path)}.*")
    return patterns


def find_ir_node_names(ov_model, pytorch_layer_names):
    """
    Map PyTorch layer names to OpenVINO IR node names (used for diagnostics only).
    Returns OV op friendly names whose name contains the transformed IR path.
    """
    ir_paths = [pt_to_ir_path(n) for n in pytorch_layer_names]
    ir_names = []
    for op in ov_model.get_ops():
        friendly = op.get_friendly_name()
        for ir_path in ir_paths:
            if ir_path in friendly:
                ir_names.append(friendly)
                break
    return ir_names


def print_matmul_nodes(ov_model):
    """Print all MatMul / Gather nodes visible in the OV IR (helps diagnose NNCF coverage)."""
    matmuls = [op.get_friendly_name() for op in ov_model.get_ops()
               if "MatMul" in op.get_type_name() or "Gather" in op.get_type_name()]
    print(f"  OV IR MatMul/Gather nodes: {len(matmuls)}")
    for name in matmuls[:10]:
        print(f"    {name}")
    if len(matmuls) > 10:
        print(f"    ... and {len(matmuls) - 10} more")


def quantize_pareto_point(core, input_model_path, point_name, kl_threshold,
                          sensitivity, output_dir):
    """Quantize a single pareto point and save the result."""
    print(f"\n{'='*60}")
    print(f"Pareto Point {point_name} (KL threshold: {kl_threshold})")
    print(f"{'='*60}")

    # Determine which layers to keep in FP16
    fp16_layers_pt = get_layers_to_keep_fp16(sensitivity, kl_threshold)
    quantized_count = len(sensitivity) - len(fp16_layers_pt)
    print(f"  Layers to quantize (INT4): {quantized_count}")
    print(f"  Layers to keep (FP16):     {len(fp16_layers_pt)}")

    # Load model fresh for each point
    ov_model = core.read_model(input_model_path)

    # Build regex patterns for layers that must stay FP16 (KL >= threshold).
    # pareto.py logic: cumulative quantization from sorted[0] to sorted[N-1];
    # everything from sorted[N] onward is too sensitive → ignored (kept FP16).
    # NNCF IgnoredScope(patterns=...) uses re.fullmatch against NNCFGraph node names.
    ignore_patterns = build_ignore_patterns(fp16_layers_pt)
    print(f"  FP16 ignore patterns: {len(ignore_patterns)}")
    # validate=False: skip NNCF's check that every pattern matched a node.
    # The sensitivity file includes 'lm_head' (from Mamba2ForCausalLM) but the
    # OV model is Mamba2Model which has no lm_head — so some patterns won't match.
    ignored = nncf.IgnoredScope(patterns=ignore_patterns, validate=False) if ignore_patterns else None

    # Compress weights — group_size=-1 → per-channel INT4 (no minimum-size constraint,
    # avoids fallback to INT8 that group_size=128 can trigger for smaller matrices)
    compress_kwargs = dict(
        model=ov_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,       # compress all non-ignored layers to INT4
        group_size=-1,   # per-channel; avoids INT8 fallback from group_size=128
    )
    if ignored:
        compress_kwargs["ignored_scope"] = ignored

    compressed_model = nncf.compress_weights(**compress_kwargs)

    # Save
    output_path = os.path.join(output_dir, f"mamba2_{point_name}.xml")
    ov.save_model(compressed_model, output_path)
    print(f"  Saved: {output_path}")

    # Report file size
    bin_path = output_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size_mb = os.path.getsize(bin_path) / (1024 * 1024)
        print(f"  Bin size: {size_mb:.1f} MB")

    return output_path


def main():
    # Load sensitivity data
    if not os.path.exists(SENSITIVITY_FILE):
        print(f"Error: {SENSITIVITY_FILE} not found.")
        print(f"Copy it from KL_Sensitivity_Analysis/sensitivity_results_mamba2-130m_4bits.json")
        return

    sensitivity = load_sensitivity(SENSITIVITY_FILE)
    print(f"Loaded sensitivity data: {len(sensitivity)} layers")
    print(f"KL range: {sensitivity[0][1]:.2f} - {sensitivity[-1][1]:.2f}")

    # Show distribution
    print("\nKL Distribution:")
    for threshold in [1.0, 2.0, 2.35, 7.0]:
        below = sum(1 for _, kl in sensitivity if kl < threshold)
        print(f"  KL < {threshold:5.2f}: {below:2d} layers quantizable")

    # Load baseline model
    if not os.path.exists(INPUT_MODEL):
        print(f"\nError: {INPUT_MODEL} not found.")
        print(f"Run convert.py first to produce the baseline IR model.")
        return

    core = ov.Core()

    # Inspect the baseline model to verify node visibility
    print(f"\nInspecting IR model node coverage...")
    ov_model = core.read_model(INPUT_MODEL)
    all_pt_names = [name for name, _ in sensitivity]
    all_ir_matches = find_ir_node_names(ov_model, all_pt_names)
    print(f"  PyTorch layers in sensitivity file: {len(all_pt_names)}")
    print(f"  OV IR nodes matched by name:        {len(all_ir_matches)}")
    print_matmul_nodes(ov_model)

    if len(all_ir_matches) < len(all_pt_names):
        coverage = len(all_ir_matches) / len(all_pt_names) * 100
        print(f"\n  WARNING: Only {coverage:.0f}% of sensitivity layers found in IR.")
        if coverage == 0:
            print("  HINT: The model may have been saved with compress_to_fp16=True in")
            print("  convert.py, which stores weights as FP16 constants and hides them")
            print("  from NNCF's pattern matching.  Re-run convert.py with")
            print("  compress_to_fp16=False, then re-run this script.")
            return
        print("  Remaining layers will be targeted via NNCF regex patterns.")
        print("  For full coverage, re-export with compress_to_fp16=False in convert.py.")

    del ov_model  # free memory

    # Quantize each pareto point
    output_paths = []
    for point_name, kl_threshold in PARETO_POINTS.items():
        path = quantize_pareto_point(
            core, INPUT_MODEL, point_name, kl_threshold,
            sensitivity, OUTPUT_DIR
        )
        output_paths.append(path)

    print(f"\n{'='*60}")
    print("Done! Quantized models saved:")
    for p in output_paths:
        print(f"  {p}")
    print(f"\nRun benchmark.py to measure NPU/CPU latency for each model.")


if __name__ == "__main__":
    main()
