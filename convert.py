import os
import torch
import openvino as ov
from transformers import Mamba2Config, Mamba2ForCausalLM
from transformers import Mamba2Model
from transformers import AutoTokenizer, AutoModel

os.makedirs("onnx_model", exist_ok=True)
os.makedirs("ov_models", exist_ok=True)

model_hf = "yuji96/mamba2-130m-hf"
config = Mamba2Config.from_pretrained(model_hf)
config.use_cache = False
config.num_hidden_layers = 24  # Full 24-layer model to match sensitivity data

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
        model=model,
        args=({'input_ids': input_ids['input_ids']},),
        f=onnx_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )

### Convert ONNX → OpenVINO IR
ov_model = ov.convert_model(input_model=onnx_path)

# compress_to_fp16=False: keeps weights as FP32 so NNCF's compress_weights
# can find all 48 MatMul nodes.  NNCF will apply INT4 compression in quantize_nncf.py.
ov.save_model(ov_model, output_model=f"ov_models/{model_name}.xml",
              compress_to_fp16=False)

print(f"Saved full Mamba-2 model (24 layers) to ov_models/{model_name}.xml")
