from typing import Union

import torch


def intersection_over_union(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) score for binary segmentation.

    Args:
        predictions: Model predictions after sigmoid.
                     Shape: [B, 1, H, W]
        targets: Ground-truth masks.
                 Shape: [B, 1, H, W]
        smooth: Small constant to avoid division by zero.

    Returns:
        Mean IoU score for the batch.
    """

    if not isinstance(predictions, torch.Tensor):
        raise TypeError(
            f"Predictions must be torch.Tensor, got {type(predictions)}"
        )

    if not isinstance(targets, torch.Tensor):
        raise TypeError(
            f"Targets must be torch.Tensor, got {type(targets)}"
        )

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch.\n"
            f"Predictions: {predictions.shape}\n"
            f"Targets: {targets.shape}"
        )

    if predictions.ndim != 4:
        raise ValueError(
            f"Expected predictions with shape [B,1,H,W], "
            f"got {predictions.shape}"
        )

    predictions = (predictions > 0.5).float()

    predictions = predictions.contiguous().view(
        predictions.size(0),
        -1
    )

    targets = targets.contiguous().view(
        targets.size(0),
        -1
    )

    intersection = (predictions * targets).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()