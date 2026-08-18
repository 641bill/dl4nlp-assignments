import torch

from A2_skeleton import (
    A2ModelConfig,
    A2DecoderLayer,
    A2RotaryEmbedding,
)


config = A2ModelConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=2,
    rope_theta=10000.0,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)

# Create one Transformer decoder layer
layer = A2DecoderLayer(config)

# Fake hidden states:
# batch_size = 2
# sequence_length = 10
# hidden_size = 256
x = torch.randn(2, 10, 256)

# Generate RoPE rotations for this sequence length
rotary_emb = A2RotaryEmbedding(config)
rope_rotations = rotary_emb(x)

# Forward pass
y = layer(x, rope_rotations)

print("Input shape: ", x.shape)
print("Output shape:", y.shape)

# A Transformer decoder layer must preserve shape
assert y.shape == x.shape

# Make sure the result contains valid numbers
assert torch.isfinite(y).all()

print("Task 1.4 sanity check passed!")