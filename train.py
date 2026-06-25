import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from datasets.flood_dataset import FloodDataset
from torch import optim
from losses.dice_loss import DiceLoss
from models.unet import UNet


EPOCHS = 20
LEARNING_RATE = 3e-4
MODEL_SAVE_PATH = "best_model.pth"
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
IMAGE_SIZE = (128, 128)
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

def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0

    for images, masks in dataloader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        print("Starting training batch...")
        predictions = model(images)
        print("Forward pass done")

        if predictions.shape != masks.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} "
                f"does not match mask shape {masks.shape}"
            )

        loss = loss_fn(
            predictions,
            masks
        )
        print("Starting backward pass")
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print("Loss computed")


    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    return epoch_loss

def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device
):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            if predictions.shape != masks.shape:
                raise ValueError(
                    f"Prediction shape {predictions.shape} "
                    f"does not match mask shape {masks.shape}"
                )

            loss = loss_fn(
                predictions,
                masks
            )

            running_loss += loss.item()

    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    return epoch_loss

def save_model(
    model,
    path
):
    torch.save(
        model.state_dict(),
        path
    )


def main():

    train_loader, val_loader, train_dataset, val_dataset = (
        create_dataloaders()
    )

    print(
        f"Total Samples      : "
        f"{len(train_dataset) + len(val_dataset)}"
    )

    print(
        f"Training Samples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Samples : "
        f"{len(val_dataset)}"
    )

    verify_dataloaders(
        train_loader,
        val_loader
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing Device: {device}")

    model = UNet().to(device)
    print(sum(p.numel() for p in model.parameters()))

    loss_fn = DiceLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_model(
                model,
                MODEL_SAVE_PATH
            )

            print(
                f"Best model saved "
                f"(Val Loss: {val_loss:.4f})"
            )

    print(
        "\nTraining Complete."
    )

if __name__ == "__main__":
    main()
