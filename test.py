import torch

from models.unet import UNet

model = UNet()

dummy_input = torch.randn(
    8,
    3,
    256,
    256
)

output = model(dummy_input)

print("Input :", dummy_input.shape)
print("Output:", output.shape)