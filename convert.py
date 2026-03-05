import os
import torch
import openvino as ov

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

### Direct PyTorch → OpenVINO conversion (no ONNX intermediate)
# The ONNX path inserts Identity ops (opset16) that the NPU compiler rejects.
# Direct conversion avoids these ONNX artifacts entirely.
example_input = torch.tensor([list(range(tokens))])

with torch.no_grad():
    ov_model = ov.convert_model(
        model,
        example_input=example_input,
        input=[("input_ids", ov.PartialShape([1, tokens]))]
    )

ov.save_model(ov_model, output_model=f"ov_model/{model_name}.xml",
                compress_to_fp16=True)

print(f"Saved full Mamba-2 model (24 layers) to ov_model/{model_name}.xml")
