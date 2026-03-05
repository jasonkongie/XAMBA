"""
NPU-compatible patch for Mamba-1's MambaMixer.slow_forward.

Root cause of NPU crash: the original slow_forward's for-loop over seq_len
unrolls (at ONNX trace time) into 16 sequential Gather ops plus a deep
sequential dependency chain — both crash the Lunar Lake NPU compiler.

Fix strategy (v3): Unrolled split-based scan
  - Use split() instead of scalar indexing → VariadicSplit in OV, not Gather
  - Hardcode the 4-step recurrence (tokens=4 is fixed in convert.py)
  - Stay strictly ≤4D tensors throughout — no 5D intermediates
  - All ops confirmed NPU-compatible from working Mamba-2 analysis

Ops introduced: VariadicSplit, Squeeze, Multiply, Add, ReduceSum, Concat
Ops eliminated: Gather (×16), Log, Clamp, 5D Exp/Multiply

Also inlines softplus -> relu (ActiBA) for NPU compatibility.

Usage in convert.py:
    from modeling_mamba_npu import patched_slow_forward
    from transformers.models.mamba.modeling_mamba import MambaMixer
    MambaMixer.slow_forward = patched_slow_forward
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers.models.mamba.modeling_mamba import MambaCache


def patched_slow_forward(
    self,
    input_states,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2*d_in, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        if cache_position.shape[0] == self.conv_kernel_size:
            conv_state = nn.functional.pad(
                hidden_states,
                (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, d_in, seq_len]
        else:
            conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, d_in, 1]
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, d_in, seq_len]

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. State Space Model sequence transformation
    # 3.a. Selection
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))                    # [batch, seq_len, rank+2*d_state]
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )
    discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, d_in]

    # ActiBA: relu instead of softplus for NPU compatibility
    discrete_time_step = nn.functional.relu(discrete_time_step).transpose(1, 2)    # [batch, d_in, seq_len]

    # 3.b. Discretization
    A = -torch.exp(self.A_log.float())                                              # [d_in, d_state]
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])# [batch, d_in, seq_len, d_state]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()      # [batch, d_in, seq_len, d_state]
    deltaB_u   = discrete_B * hidden_states[:, :, :, None].float()                 # [batch, d_in, seq_len, d_state]

    # 3.c. Unrolled split-based SSM scan (NPU-safe, max 4D tensors only)
    #
    # Problem with the original for-loop:
    #   discrete_A[:, :, i, :] → Gather op × 16 → crashes NPU compiler
    #
    # Problem with previous vectorized approach:
    #   factor matrix → 5D tensor [batch, d_in, seq, seq, d_state] → NPU likely
    #   does not support 5D ops
    #
    # This approach:
    #   split() → VariadicSplit (1 op, NPU-confirmed) + Squeeze (standard op)
    #   Recurrence unrolled explicitly → no loop variable, no Gather, max 4D
    #
    # Hardcoded for tokens=4 (seq_len is always 4 in convert.py).
    # Each split() call produces ONE VariadicSplit node with N outputs — not N Gather nodes.

    # Split all SSM tensors along the seq_len dim
    # discrete_A/deltaB_u: split on dim=2; C: split on dim=1
    dA  = discrete_A.split(1, dim=2)     # 4 × [batch, d_in, 1, d_state]
    dBu = deltaB_u.split(1, dim=2)       # 4 × [batch, d_in, 1, d_state]
    C_t = C.split(1, dim=1)              # 4 × [batch, 1, d_state]

    # Squeeze out the size-1 seq dim from each slice
    dA  = [t.squeeze(2) for t in dA]    # 4 × [batch, d_in, d_state]
    dBu = [t.squeeze(2) for t in dBu]   # 4 × [batch, d_in, d_state]
    # C_t[i]: [batch, 1, d_state] — keep the dim for broadcasting with h below

    # Unrolled recurrence: h[t] = dA[t] * h[t-1] + dBu[t], h[-1] = 0
    h0 = dBu[0]                          # [batch, d_in, d_state]
    h1 = dA[1] * h0 + dBu[1]
    h2 = dA[2] * h1 + dBu[2]
    h3 = dA[3] * h2 + dBu[3]

    # y[t] = sum_k h[t, :, k] * C[t, k]  →  [batch, d_in]
    # h[t]: [batch, d_in, d_state], C_t[t]: [batch, 1, d_state] → broadcasts over d_in
    y0 = (h0 * C_t[0]).sum(-1)           # [batch, d_in]
    y1 = (h1 * C_t[1]).sum(-1)
    y2 = (h2 * C_t[2]).sum(-1)
    y3 = (h3 * C_t[3]).sum(-1)

    # Stack outputs along the seq_len dim
    scan_output = torch.stack([y0, y1, y2, y3], dim=-1)                            # [batch, d_in, seq_len]

    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(h3.to(dtype))

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
    return contextualized_states
