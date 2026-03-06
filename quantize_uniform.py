"""
quantize_uniform.py

Creates two uniform-precision quantized models for comparison baselines:
  - mamba2_uniform_int8.xml  : all layers INT8_SYM (no mixed precision)
  - mamba2_uniform_int4.xml  : all layers INT4_SYM (no mixed precision)

These serve as the two endpoints of the Pareto curve:
  uniform INT8  →  best quality,  larger size
  uniform INT4  →  smallest size, lowest quality
  pointA–D      →  mixed-precision sweet spots in between

Usage:
    python quantize_uniform.py

Prerequisites:
    - ov_models/mamba2-130m-hf.xml  (produced by convert.py)
    - pip install nncf openvino
"""

import os
import openvino as ov
import nncf

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_MODEL = "ov_models/mamba2-130m-hf.xml"
OUTPUT_DIR  = "ov_models"

CONFIGS = {
    "uniform_int8": dict(
        mode       = nncf.CompressWeightsMode.INT8_SYM,
        group_size = -1,     # per-channel
    ),
    "uniform_int4": dict(
        mode       = nncf.CompressWeightsMode.INT4_SYM,
        ratio      = 1.0,    # compress ALL layers
        group_size = -1,     # per-channel; avoids INT8 fallback
    ),
}

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: {INPUT_MODEL} not found. Run convert.py first.")
        return

    core = ov.Core()

    for name, kwargs in CONFIGS.items():
        print(f"\n{'='*55}")
        print(f"  Quantizing: {name}  (mode: {kwargs['mode']})")
        print(f"{'='*55}")

        ov_model = core.read_model(INPUT_MODEL)

        compressed = nncf.compress_weights(model=ov_model, **kwargs)

        output_path = os.path.join(OUTPUT_DIR, f"mamba2_{name}.xml")
        ov.save_model(compressed, output_path, compress_to_fp16=True)
        print(f"  Saved: {output_path}")

        bin_path = output_path.replace(".xml", ".bin")
        if os.path.exists(bin_path):
            size_mb = os.path.getsize(bin_path) / (1024 * 1024)
            print(f"  Bin size: {size_mb:.1f} MB")

    print(f"\n{'='*55}")
    print("Done! Models saved:")
    for name in CONFIGS:
        print(f"  ov_models/mamba2_{name}.xml")
    print("\nAdd these to your benchmark bash script:")
    for name in CONFIGS:
        print(f"  benchmark_app -m ov_models/mamba2_{name}.xml -d CPU -hint latency -t 30 \\")
        print(f"    2>&1 | tee log/benchmark_log/mamba2_{name}_CPU_latency.txt")


if __name__ == "__main__":
    main()
