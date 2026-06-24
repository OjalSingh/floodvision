
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from datasets.flood_dataset import FloodDataset


IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42


def create_dataloaders():
    if not Path(IMAGE_DIR).exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not Path(MASK_DIR).exists():
        raise FileNotFoundError(
            f"Mask directory not found: {MASK_DIR}"
        )

    dataset = FloodDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE
    )

    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError(
            f"Dataset contains only {dataset_size} sample(s). "
            f"At least 2 samples are required for train-validation split."
        )

    train_size = int(TRAIN_SPLIT * dataset_size)
    val_size = dataset_size - train_size

    if train_size == 0:
        raise ValueError(
            "Train set size is 0. Adjust TRAIN_SPLIT."
        )

    if val_size == 0:
        raise ValueError(
            "Validation set size is 0. Adjust TRAIN_SPLIT."
        )

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_dataset, val_dataset


def verify_dataloaders(train_loader, val_loader):
    try:
        train_images, train_masks = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError(
            "Training DataLoader returned no batches."
        )

    try:
        val_images, val_masks = next(iter(val_loader))
    except StopIteration:
        raise RuntimeError(
            "Validation DataLoader returned no batches."
        )

    #error handling

    if train_images.ndim != 4:
        raise ValueError(
            f"Expected training images to have 4 dimensions, "
            f"got {train_images.shape}"
        )

    if train_masks.ndim != 4:
        raise ValueError(
            f"Expected training masks to have 4 dimensions, "
            f"got {train_masks.shape}"
        )

    if val_images.ndim != 4:
        raise ValueError(
            f"Expected validation images to have 4 dimensions, "
            f"got {val_images.shape}"
        )

    if val_masks.ndim != 4:
        raise ValueError(
            f"Expected validation masks to have 4 dimensions, "
            f"got {val_masks.shape}"
        )

    if train_images.shape[1] != 3:
        raise ValueError(
            f"Expected 3 image channels, got "
            f"{train_images.shape[1]}"
        )

    if train_masks.shape[1] != 1:
        raise ValueError(
            f"Expected 1 mask channel, got "
            f"{train_masks.shape[1]}"
        )

    print("\nTraining Batch")
    print("Images:", train_images.shape)
    print("Masks :", train_masks.shape)

    print("\nValidation Batch")
    print("Images:", val_images.shape)
    print("Masks :", val_masks.shape)


def main():
    train_loader, val_loader, train_dataset, val_dataset = (
        create_dataloaders()
    )

    print(f"Total Samples      : {len(train_dataset) + len(val_dataset)}")
    print(f"Training Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}")

    verify_dataloaders(train_loader, val_loader)


if __name__ == "__main__":
    main()
