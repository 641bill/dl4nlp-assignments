import torch
from torch import nn
from A2_skeleton import A2ModelConfig

config = A2ModelConfig(
    hidden_size=256,
    rms_norm_eps=1e-6,
)

norm = nn.RMSNorm(
    config.hidden_size,
    eps=config.rms_norm_eps,
    elementwise_affine=True,
)

x = torch.randn(2, 10, 256)
y = norm(x)

print("Input: ", x.shape)
print("Output:", y.shape)

assert y.shape == x.shape

print("Task 1.2 passed!")