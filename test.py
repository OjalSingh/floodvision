
from datasets.flood_dataset import FloodDataset

dataset = FloodDataset(
    image_dir="data/images",
    mask_dir="data/masks",
    image_size=(256, 256)
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

print("Image dtype:", image.dtype)
print("Mask dtype:", mask.dtype)

print("Mask unique values:", mask.unique())
