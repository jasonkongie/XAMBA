import os
import torch
import openvino as ov

os.makedirs("onnx_model", exist_ok=True)
os.makedirs("ov_model", exist_ok=True)

from transformers import MambaConfig, MambaForCausalLM
from transformers import MambaModel
from transformers import AutoTokenizer, AutoModel
from transformers.models.mamba.modeling_mamba import MambaMixer
from modeling_mamba_npu import patched_slow_forward

# Patch MambaMixer.slow_forward with NPU-compatible vectorized scan
# Eliminates 16 loop-generated Gather ops → CumSum + triangular matmul
MambaMixer.slow_forward = patched_slow_forward

config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
config.use_cache = False
config.num_hidden_layers = 1

model = MambaModel(config=config).eval()
print(model)

tokens = 4

### Direct PyTorch → OpenVINO conversion (no ONNX intermediate)
# Docs: "it is recommended to set up static shapes for the inputs at the model
# preparation stage, not at runtime, for performance and NPU compatibility."
example_input = torch.tensor([list(range(tokens))])

with torch.no_grad():
    ov_model = ov.convert_model(
        model,
        example_input=example_input,
        input=[("input_ids", ov.PartialShape([1, tokens]))]
    )

# compress_to_fp16=False for initial debugging (removes weight compression as a variable)
ov.save_model(ov_model, output_model=f"ov_model/mamba_b_1_t_{tokens}.xml",
                compress_to_fp16=False)
