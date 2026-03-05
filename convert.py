import os
import torch
import openvino as ov

os.makedirs("onnx_model", exist_ok=True)
os.makedirs("ov_model", exist_ok=True)

from transformers import MambaConfig, MambaForCausalLM
from transformers import MambaModel
from transformers import AutoTokenizer, AutoModel

config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
config.use_cache = False
config.num_hidden_layers = 1

model = MambaModel(config=config).eval()
print(model)

tokens = 4

### ONNX
input_ids = {'input_ids': torch.tensor([list(range(tokens))])}
onnx_path = f"onnx_model/mamba_b_1_t_{tokens}.onnx"
input_names = list(input_ids.keys())
output_names = ['last_hidden_state']
 
with torch.no_grad():
    torch.onnx.export(
        model=model,
        args=(input_ids['input_ids'],),
        f=onnx_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None
        )


ov_model = ov.convert_model(input_model=onnx_path)
# save generated model file for future use
ov.save_model(ov_model, output_model=f"ov_model/mamba_b_1_t_{tokens}.xml",
                compress_to_fp16=True)
