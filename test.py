# import torch

# from losses.dice_loss import DiceLoss


# loss_fn = DiceLoss()

# predictions = torch.randn(
#     8,
#     1,
#     256,
#     256
# )

# targets = torch.randint(
#     low=0,
#     high=2,
#     size=(8, 1, 256, 256)
# ).float()

# loss = loss_fn(
#     predictions,
#     targets
# )

# print("Loss:", loss.item())


import torch

from models.unet import UNet

model = UNet()

images = torch.randn(
    8,
    3,
    256,
    256
)

output = model(images)

print(output.shape)