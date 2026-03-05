import os
import torch
import openvino as ov

os.makedirs("onnx_model", exist_ok=True)
os.makedirs("ov_model", exist_ok=True)

from transformers import Mamba2Config, Mamba2ForCausalLM
from transformers import Mamba2Model
from transformers import AutoTokenizer, AutoModel

model_hf = "yuji96/mamba2-130m-hf"
config = Mamba2Config.from_pretrained(model_hf)
config.use_cache = False
# Full 24-layer model for quantization experiments
config.num_hidden_layers = 24

model_name = model_hf.split("/")[1]
model = Mamba2Model(config=config).eval()
print(model)

tokens = 4

### ONNX export
input_ids = {'input_ids': torch.tensor([list(range(tokens))])}
onnx_path = f"onnx_model/{model_name}.onnx"
input_names = list(input_ids.keys())
output_names = ['last_hidden_state']

with torch.no_grad():
    torch.onnx.export(
        model = model,
        args = ({'input_ids': input_ids['input_ids'],
                }),
        f=onnx_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None
        )

### Simplify ONNX to remove Identity ops (NPU rejects Identity at opset16)
import onnx
from onnxsim import simplify

print("Simplifying ONNX model (removing Identity ops)...")
model_onnx = onnx.load(onnx_path)
model_simplified, check = simplify(model_onnx)
assert check, "ONNX simplification failed validation"
simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
onnx.save(model_simplified, simplified_path)
print(f"Simplified ONNX saved to {simplified_path}")

### Convert simplified ONNX → OpenVINO IR
ov_model = ov.convert_model(input_model=simplified_path)
ov.save_model(ov_model, output_model=f"ov_model/{model_name}.xml",
                compress_to_fp16=False)  # Keep FP32 so NNCF can see all weight nodes

print(f"Saved full Mamba-2 model (24 layers) to ov_model/{model_name}.xml")
