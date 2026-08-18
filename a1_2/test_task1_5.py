import torch

from A2_skeleton import (
    A2ModelConfig,
    A2Transformer,
)


config = A2ModelConfig(
    vocab_size=10_000,
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=2,
    rope_theta=10000.0,
    hidden_act="silu",
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
)

model = A2Transformer(config)


# --------------------------------------------------
# Test 1: forward pass without labels
# --------------------------------------------------

# Fake token IDs
# B = 2
# N = 10
input_ids = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 10),
)

output = model(input_ids)

print("Input IDs shape:", input_ids.shape)
print("Logits shape:   ", output.logits.shape)

# Expected:
# input_ids: (B, N)
# logits:    (B, N, V)
assert output.logits.shape == (
    2,
    10,
    config.vocab_size,
)

assert torch.isfinite(output.logits).all()

print("Forward pass passed!")


# --------------------------------------------------
# Test 2: forward pass with labels
# --------------------------------------------------

labels = input_ids.clone()

output_with_loss = model(
    input_ids=input_ids,
    labels=labels,
)

print("Loss:", output_with_loss.loss)

assert output_with_loss.loss is not None
assert torch.isfinite(output_with_loss.loss)

print("Loss computation passed!")


# --------------------------------------------------
# Test 3: backward pass
# --------------------------------------------------

output_with_loss.loss.backward()

# Check that at least some parameters received gradients
params_with_grad = [
    p
    for p in model.parameters()
    if p.grad is not None
]

print(
    "Parameters with gradients:",
    len(params_with_grad),
)

assert len(params_with_grad) > 0

print("Backward pass passed!")


print("\nTask 1.5 sanity check passed!")