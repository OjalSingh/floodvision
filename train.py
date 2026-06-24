from torch.utils.data import DataLoader
from datasets.flood_dataset import FloodDataset

dataset = FloodDataset(
    image_dir="data/images",
    mask_dir="data/masks"
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, masks = next(iter(train_loader))

print(images.shape)
print(masks.shape)