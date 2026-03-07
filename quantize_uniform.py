"""
quantize_uniform.py

Creates uniform-precision quantized models for each base model:
  - {model}_uniform_int8.xml  : all layers INT8_SYM
  - {model}_uniform_int4.xml  : all layers INT4_SYM

These serve as the two endpoints of the Pareto curve:
  uniform INT8  →  best quality,  larger size
  uniform INT4  →  smallest size, lowest quality
  mixed points  →  sweet spots in between

Usage:
    python quantize_uniform.py
"""

import os
import openvino as ov
import nncf

# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mamba-130m-hf": {
        "hf_id": "state-spaces/mamba-130m-hf",
        "sensitivity_4bit": "mamba130m_sensitivity_results_4bits.json",
        "sensitivity_8bit": "mamba130m_sensitivity_results_8bits.json",
    },
    "mamba2_b_1_t_4": {
        "hf_id": "yuji96/mamba2-130m-hf",
        # XAMBA CumBA MatMul (/mixer/MatMul) has no INT8/INT4 GPU kernel.
        # Even at the uniform level (all 4 linear layers INT8), the surrounding
        # INT8 context causes oneDNN to fail its primitive descriptor for the
        # MatMul — same root cause as mixed-precision point09/10.
        # Must exclude it from quantization so the model compiles on GPU.
        "xamba": True,
    },
}

OUTPUT_DIR = "ov_models"

# Shared ignore scopes
XAMBA_MATMUL_PATTERN = r".*/mixer/MatMul.*"

# Standard: exclude Convolution (SSM conv1d) only
LINEAR_ONLY_SCOPE = nncf.IgnoredScope(types=["Convolution"], validate=False)

# XAMBA: also exclude the CumBA MatMul so GPU compilation succeeds
XAMBA_LINEAR_ONLY_SCOPE = nncf.IgnoredScope(
    types    = ["Convolution"],
    patterns = [XAMBA_MATMUL_PATTERN],
    validate = False,
)

CONFIGS = {
    "uniform_int8": dict(
        mode       = nncf.CompressWeightsMode.INT8_SYM,
        group_size = -1,     # per-channel
    ),
    "uniform_int4": dict(
        mode       = nncf.CompressWeightsMode.INT4_SYM,
        ratio      = 1.0,    # compress ALL eligible layers
        group_size = -1,     # per-channel; avoids INT8 fallback
    ),
}

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    core = ov.Core()

    for model_name, cfg in MODEL_REGISTRY.items():
        input_model = os.path.join(OUTPUT_DIR, f"{model_name}.xml")
        if not os.path.exists(input_model):
            print(f"[!] Baseline model not found: {input_model}  — run convert.py first")
            continue

        # XAMBA models need the MatMul excluded too (no GPU INT8/INT4 kernel)
        scope = XAMBA_LINEAR_ONLY_SCOPE if cfg.get("xamba") else LINEAR_ONLY_SCOPE

        for suffix, kwargs in CONFIGS.items():
            print(f"\n{'='*55}")
            print(f"  {model_name} → {suffix}  (mode: {kwargs['mode']})")
            if cfg.get("xamba"):
                print(f"  [XAMBA] MatMul excluded from quantization (no GPU kernel)")
            print(f"{'='*55}")

            ov_model   = core.read_model(input_model)
            compressed = nncf.compress_weights(model=ov_model, ignored_scope=scope, **kwargs)

            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{suffix}.xml")
            ov.save_model(compressed, output_path, compress_to_fp16=True)

            bin_path = output_path.replace(".xml", ".bin")
            if os.path.exists(bin_path):
                size_mb = os.path.getsize(bin_path) / (1024 * 1024)
                print(f"  Saved: {output_path}  ({size_mb:.1f} MB)")

    print(f"\n{'='*55}")
    print("Done! Uniform baselines generated:")
    for model_name in MODEL_REGISTRY:
        for suffix in CONFIGS:
            print(f"  ov_models/{model_name}_{suffix}.xml")


if __name__ == "__main__":
    main()
