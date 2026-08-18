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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# Fake batch
input_ids = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 32)
)

labels = input_ids.clone()


# -------------------------
# Forward
# -------------------------

output = model(
    input_ids=input_ids,
    labels=labels
)

loss = output.loss

print("Logits shape:", output.logits.shape)
print("Loss:", loss.item())

assert output.logits.shape == (
    2,
    32,
    config.vocab_size
)

assert torch.isfinite(loss)


# -------------------------
# Backward
# -------------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("Backward + optimizer step passed!")

print("\nTask 2.1 smoke test passed!")