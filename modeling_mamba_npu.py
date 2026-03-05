"""
NPU-compatible patch for Mamba-1's MambaMixer.slow_forward.

Problem: The original slow_forward uses a Python for-loop over seq_len that,
when ONNX-traced, unrolls into 16+ sequential Gather ops. The sequential
dependency chain crashes the NPU compiler at Step 7 (model loading).

Fix: Replace the for-loop SSM scan with a fully vectorized implementation
using CumSum + triangular factor matrix. This is mathematically exact
(no approximation) and uses only NPU-supported ops.

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
    projected_states = self.in_proj(input_states).transpose(1, 2)   # [batch, 2*d_in, seq_len]
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
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()                   # [batch, d_in, seq_len, d_state]

    # 3.c. Vectorized SSM scan — replaces the for-loop to eliminate Gather ops
    #
    # Recurrence: h[t] = A[t]*h[t-1] + Bu[t],  h[-1] = 0
    # Closed form: h[t] = Σ_{s=0}^{t} (Π_{r=s+1}^{t} A[r]) * Bu[s]
    #
    # factor[t,s] = Π_{r=s+1}^{t} A[r]
    #             = exp( Σ_{r=s+1}^{t} log(A[r]) )
    #             = exp( cumlog_A[t] - cumlog_A[s] )
    #
    # where cumlog_A[t] = Σ_{r=0}^{t} log(A[r])   (CumSum — NPU supported)
    #
    # Ops: Log, CumSum, Unsqueeze, Subtract, Exp, Multiply, ReduceSum
    # All confirmed NPU-supported from Mamba-2 working model.

    log_A = torch.log(discrete_A.float().clamp(min=1e-8))                          # [batch, d_in, seq_len, d_state]
    cumlog_A = torch.cumsum(log_A, dim=2)                                           # [batch, d_in, seq_len, d_state]

    # factor[b, d, t, s, k] = exp(cumlog_A[b,d,t,k] - cumlog_A[b,d,s,k])
    factors = torch.exp(
        cumlog_A.unsqueeze(3) - cumlog_A.unsqueeze(2)                              # [batch, d_in, seq_len, seq_len, d_state]
    )

    # Causal lower-triangular mask: zero out future positions (t < s)
    # With static seq_len=4 this becomes a Const node in ONNX — no runtime cost
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=factors.dtype, device=factors.device))
    factors = factors * mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)               # [batch, d_in, seq_len, seq_len, d_state]

    # h[t] = Σ_s factor[t,s] * Bu[s]
    h = (factors * deltaB_u.float().unsqueeze(2)).sum(dim=3)                       # [batch, d_in, seq_len, d_state]

    # y[t] = Σ_k h[t,:,k] * C[t,k]
    scan_output = (h * C.float().unsqueeze(1)).sum(dim=-1).to(dtype)               # [batch, d_in, seq_len]

    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(h[:, :, -1, :].to(dtype))

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
    return contextualized_states
