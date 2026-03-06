import os
import torch
import openvino as ov
import onnx
from onnxsim import simplify
from transformers import AutoConfig, AutoModel

os.makedirs("onnx_model", exist_ok=True)
os.makedirs("ov_models", exist_ok=True)

MODELS = [
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-1.4b-hf",
]

tokens = 4

for model_hf in MODELS:
    model_name = model_hf.split("/")[1]

    print(f"\n{'='*60}")
    print(f"Converting: {model_hf}")
    print(f"{'='*60}")

    ### Load model from pretrained config (use_cache=False for static export)
    config = AutoConfig.from_pretrained(model_hf)
    config.use_cache = False
    model = AutoModel.from_pretrained(model_hf, config=config).eval()
    print(model)

    ### ONNX export
    dummy_input = torch.tensor([list(range(tokens))])   # [1, 4]
    onnx_path   = f"onnx_model/{model_name}.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model        = model,
            args         = ({'input_ids': dummy_input},),
            f            = onnx_path,
            verbose      = False,
            input_names  = ['input_ids'],
            output_names = ['last_hidden_state'],
            dynamic_axes = None,
        )
    print(f"ONNX saved: {onnx_path}")

    ### Simplify ONNX (removes Identity ops and cleans up the graph)
    print("Simplifying ONNX model...")
    model_onnx = onnx.load(onnx_path)
    model_simplified, check = simplify(model_onnx)
    assert check, "ONNX simplification failed validation"
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    onnx.save(model_simplified, simplified_path)
    print(f"Simplified ONNX saved: {simplified_path}")

    ### Convert simplified ONNX → OpenVINO IR
    ov_model    = ov.convert_model(input_model=simplified_path)
    output_path = f"ov_models/{model_name}.xml"

    # compress_to_fp16=False: keeps weights as FP32 so NNCF compress_weights
    # can find all MatMul nodes for quantization.
    ov.save_model(ov_model, output_model=output_path, compress_to_fp16=False)

    bin_size_mb = os.path.getsize(output_path.replace(".xml", ".bin")) / (1024 * 1024)
    print(f"Saved: {output_path}  ({bin_size_mb:.1f} MB)")

print(f"\n{'='*60}")
print("All models converted! Files in ov_models/:")
for model_hf in MODELS:
    model_name = model_hf.split("/")[1]
    print(f"  ov_models/{model_name}.xml")
print("\nNext: run benchmark bash script or quantize_uniform.py")
