import torch
from A2_skeleton import A2ModelConfig, A2MLP


config = A2ModelConfig(
    hidden_size=256,
    intermediate_size=1024,
    hidden_act="silu"
)

mlp = A2MLP(config)

x = torch.randn(2, 10, 256)

y = mlp(x)

print("Input shape: ", x.shape)
print("Output shape:", y.shape)

assert y.shape == x.shape

print("Task 1.1 sanity check passed!")