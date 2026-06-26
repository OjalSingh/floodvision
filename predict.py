from pathlib import Path
import numpy as np
import torch

from models.unet import UNet
from train import create_dataloaders

import matplotlib.pyplot as plt
from pathlib import Path


MODEL_PATH = "best_model.pth"


def load_model():
    """
    Loads the trained U-Net model and prepares it for inference.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = Path(MODEL_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    model = UNet()

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load checkpoint: {checkpoint_path}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a valid PyTorch state dictionary."
        )

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint is incompatible with the current U-Net architecture."
        ) from error

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"Checkpoint : {checkpoint_path.resolve()}")
    print(f"Device     : {device}")

    return model, device


def load_validation_loader():
    """
    Creates and returns the validation DataLoader.
    """

    try:
        (
            _,
            val_loader,
            _,
            val_dataset
        ) = create_dataloaders()

    except Exception as error:
        raise RuntimeError(
            "Failed to create validation DataLoader."
        ) from error

    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    print(
        f"Validation Samples : {len(val_dataset)}"
    )

    return val_loader

def run_inference(
    model,
    images,
    masks,
    device
):
    """
    Runs inference on one validation sample.
    """

    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        predictions = model(images)

    if predictions.ndim != 4:
        raise ValueError(
            f"Expected prediction tensor with 4 dimensions, got {predictions.shape}"
        )

    if predictions.shape != masks.shape:
        raise ValueError(
            "Prediction-mask dimension mismatch.\n"
            f"Predictions: {predictions.shape}\n"
            f"Masks      : {masks.shape}"
        )

    predictions = (predictions > 0.5).float()

    print("\nInference Successful")
    print("--------------------")
    print(f"Input Shape      : {images.shape}")
    print(f"Ground Truth     : {masks.shape}")
    print(f"Prediction Shape : {predictions.shape}")
    print(f"Prediction Values: {torch.unique(predictions)}")

    return images, masks, predictions

def prepare_visualization(
    images,
    masks,
    predictions
):
    """
    Converts tensors into NumPy arrays suitable for visualization.
    """

    if not isinstance(images, torch.Tensor):
        raise TypeError(
            "Images must be a torch.Tensor."
        )

    if not isinstance(masks, torch.Tensor):
        raise TypeError(
            "Masks must be a torch.Tensor."
        )

    if not isinstance(predictions, torch.Tensor):
        raise TypeError(
            "Predictions must be a torch.Tensor."
        )

    if (
        images.shape[0] != 1
        or masks.shape[0] != 1
        or predictions.shape[0] != 1
    ):
        raise ValueError(
            "Expected a batch size of 1."
        )

    image = (
        images[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    mask = (
        masks[0, 0]
        .cpu()
        .numpy()
    )

    prediction = (
        predictions[0, 0]
        .cpu()
        .numpy()
    )

    if image.ndim != 3:
        raise ValueError(
            f"Expected RGB image, got {image.shape}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got {mask.shape}"
        )

    if prediction.ndim != 2:
        raise ValueError(
            f"Expected 2D prediction, got {prediction.shape}"
        )

    print("\nVisualization Ready")
    print("-------------------")
    print(f"Image Shape      : {image.shape}")
    print(f"Mask Shape       : {mask.shape}")
    print(f"Prediction Shape : {prediction.shape}")

    return (
        image,
        mask,
        prediction
    )

def save_prediction_visualization(
    image,
    mask,
    prediction,
    output_directory="predictions",
    filename="prediction_001.png"
):
    """
    Saves a side-by-side visualization of the
    original image, ground truth mask and prediction.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected RGB image, got {image.shape}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got {mask.shape}"
        )

    if prediction.ndim != 2:
        raise ValueError(
            f"Expected 2D prediction, got {prediction.shape}"
        )

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_directory / filename

    figure = plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(
        mask,
        cmap="gray"
    )
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(
        prediction,
        cmap="gray"
    )
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()

    try:
        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to save visualization to {output_path}"
        ) from error
    finally:
        plt.close(figure)

    print("\nVisualization Saved")
    print("-------------------")
    print(output_path.resolve())


def generate_prediction_gallery(
    model,
    val_loader,
    device,
    max_predictions=5
):
    """
    Generates and saves prediction visualizations.
    """

    if max_predictions <= 0:
        raise ValueError(
            "max_predictions must be greater than zero."
        )

    saved = 0

    model.eval()

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            predictions = (
                predictions > 0.5
            ).float()

            batch_size = images.shape[0]

            for index in range(batch_size):

                image = (
                    images[index]
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )

                mask = (
                    masks[index, 0]
                    .cpu()
                    .numpy()
                )

                prediction = (
                    predictions[index, 0]
                    .cpu()
                    .numpy()
                )

                save_prediction_visualization(
                    image,
                    mask,
                    prediction,
                    filename=(
                        f"prediction_{saved + 1:03d}.png"
                    )
                )

                saved += 1

                if saved >= max_predictions:

                    print(
                        f"\nSaved {saved} prediction(s)."
                    )

                    return

# def main():

#     model, device = load_model()

#     images, masks = load_validation_sample()

#     images, masks, predictions = run_inference(
#         model,
#         images,
#         masks,
#         device
#     )

#     image, mask, prediction = prepare_visualization(
#         images,
#         masks,
#         predictions
#     )

#     save_prediction_visualization(
#         image,
#         mask,
#         prediction
#     )


def main():

    model, device = load_model()

    val_loader = load_validation_loader()

    generate_prediction_gallery(
        model,
        val_loader,
        device,
        max_predictions=15
    )


if __name__ == "__main__":
    main()