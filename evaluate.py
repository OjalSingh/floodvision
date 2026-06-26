from pathlib import Path
import torch
from models.unet import UNet
from train import create_dataloaders


MODEL_PATH = "best_model.pth"


def load_model():
    """
    Loads the trained U-Net model from disk and prepares it for inference.
    Returns:
        model (torch.nn.Module): Loaded model in evaluation mode.
        device (torch.device): Device on which the model is loaded.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet()

    checkpoint_path = Path(MODEL_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path.resolve()}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}'."
        ) from error

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a valid PyTorch state dictionary."
        )

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint does not match the current U-Net architecture."
        ) from error

    model.to(device)
    model.eval()

    print("\nModel loaded successfully.")
    print(f"Checkpoint : {checkpoint_path.resolve()}")
    print(f"Device     : {device}")

    return model, device

def load_validation_dataloader():
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


def verify_inference(
    model,
    val_loader,
    device
):
    """
    Runs inference on a single validation batch to verify that the
    trained model produces valid segmentation masks.
    """

    try:
        images, masks = next(iter(val_loader))
    except StopIteration:
        raise RuntimeError(
            "Validation DataLoader returned no batches."
        )

    if images.ndim != 4:
        raise ValueError(
            f"Expected image batch to have 4 dimensions, "
            f"got {images.shape}"
        )

    if masks.ndim != 4:
        raise ValueError(
            f"Expected mask batch to have 4 dimensions, "
            f"got {masks.shape}"
        )

    if images.shape[1] != 3:
        raise ValueError(
            f"Expected RGB images with 3 channels, "
            f"got {images.shape[1]}"
        )

    if masks.shape[1] != 1:
        raise ValueError(
            f"Expected masks with 1 channel, "
            f"got {masks.shape[1]}"
        )

    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        predictions = model(images)

    if predictions.ndim != 4:
        raise ValueError(
            f"Model output should have 4 dimensions, "
            f"got {predictions.shape}"
        )

    if predictions.shape != masks.shape:
        raise ValueError(
            "Prediction-mask dimension mismatch.\n"
            f"Predictions: {predictions.shape}\n"
            f"Masks      : {masks.shape}"
        )

    binary_predictions = (
        predictions > 0.5
    ).float()

    unique_values = torch.unique(binary_predictions)

    print("\nInference Verification")
    print("----------------------")
    print(f"Images Shape       : {images.shape}")
    print(f"Masks Shape        : {masks.shape}")
    print(f"Predictions Shape  : {predictions.shape}")
    print(f"Prediction Dtype   : {predictions.dtype}")
    print(f"Binary Mask Values : {unique_values}")

    return binary_predictions, masks

def dice_score(
    predictions: torch.Tensor,
    masks: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Computes the Dice Score between predicted masks and ground truth masks.

    Args:
        predictions: Binary prediction tensor of shape
                     (B, 1, H, W)
        masks: Ground truth tensor of shape
               (B, 1, H, W)
        smooth: Small constant to avoid division by zero.

    Returns:
        Dice Score as a float.
    """

    if not isinstance(predictions, torch.Tensor):
        raise TypeError(
            "Predictions must be a torch.Tensor."
        )

    if not isinstance(masks, torch.Tensor):
        raise TypeError(
            "Masks must be a torch.Tensor."
        )

    if predictions.ndim != 4:
        raise ValueError(
            f"Predictions must have 4 dimensions, "
            f"got {predictions.shape}"
        )

    if masks.ndim != 4:
        raise ValueError(
            f"Masks must have 4 dimensions, "
            f"got {masks.shape}"
        )

    if predictions.shape != masks.shape:
        raise ValueError(
            "Prediction-mask shape mismatch.\n"
            f"Predictions: {predictions.shape}\n"
            f"Masks      : {masks.shape}"
        )

    if predictions.dtype != torch.float32:
        predictions = predictions.float()

    if masks.dtype != torch.float32:
        masks = masks.float()

    predictions = predictions.contiguous().view(-1)
    masks = masks.contiguous().view(-1)

    intersection = torch.sum(
        predictions * masks
    )

    prediction_area = torch.sum(predictions)
    mask_area = torch.sum(masks)

    dice = (
        (2.0 * intersection + smooth)
        /
        (prediction_area + mask_area + smooth)
    )

    if torch.isnan(dice):
        raise RuntimeError(
            "Dice Score computation resulted in NaN."
        )

    if torch.isinf(dice):
        raise RuntimeError(
            "Dice Score computation resulted in Inf."
        )

    return dice.item()


def evaluate_dice(
    model,
    val_loader,
    device
):
    """
    Evaluates the trained model on the complete validation dataset
    and returns the average Dice Score.
    """

    if model is None:
        raise ValueError(
            "Model cannot be None."
        )

    model.eval()

    total_dice = 0.0
    total_batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            if images.shape[0] != masks.shape[0]:
                raise ValueError(
                    "Batch size mismatch.\n"
                    f"Images: {images.shape}\n"
                    f"Masks : {masks.shape}"
                )

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            if predictions.shape != masks.shape:
                raise ValueError(
                    "Prediction-mask shape mismatch.\n"
                    f"Predictions: {predictions.shape}\n"
                    f"Masks      : {masks.shape}"
                )

            predictions = (
                predictions > 0.5
            ).float()

            batch_dice = dice_score(
                predictions,
                masks
            )

            total_dice += batch_dice
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    average_dice = (
        total_dice /
        total_batches
    )

    return average_dice


# def main():
#     model, device = load_model()

#     print("\nModel Summary")
#     print(model.__class__.__name__)
#     print(f"Running on: {device}")


# def main():

#     model, device = load_model()

#     val_loader = load_validation_dataloader()

#     print("\nEvaluation setup complete.")

#     print(
#         f"Validation Batches : {len(val_loader)}"
#     )


# def main():

#     model, device = load_model()

#     val_loader = load_validation_dataloader()

#     verify_inference(
#         model,
#         val_loader,
#         device
#     )

# def main():

#     model, device = load_model()

#     val_loader = load_validation_dataloader()

#     predictions, masks = verify_inference(
#         model,
#         val_loader,
#         device
#     )

#     score = dice_score(
#         predictions,
#         masks
#     )

#     print(f"\nDice Score: {score:.4f}")


def main():

    model, device = load_model()

    val_loader = load_validation_dataloader()

    average_dice = evaluate_dice(
        model,
        val_loader,
        device
    )

    print("\nEvaluation Complete")
    print("-------------------")
    print(
        f"Average Dice Score : "
        f"{average_dice:.4f}"
    )


if __name__ == "__main__":
    main()