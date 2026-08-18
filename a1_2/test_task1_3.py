import torch

from A2_skeleton import (
    A2ModelConfig,
    A2Attention,
    A2RotaryEmbedding,
)

config = A2ModelConfig(
    hidden_size=256,
    num_attention_heads=8,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)

attention = A2Attention(config)

x = torch.randn(
    2, 10, 256
)

rotary_emb = A2RotaryEmbedding(config)

rope_rotations = rotary_emb(x)

y = attention(
    x,
    rope_rotations
)

print("Input: ", x.shape)
print("Output:", y.shape)

assert y.shape == x.shape

print("Task 1.3 passed!")