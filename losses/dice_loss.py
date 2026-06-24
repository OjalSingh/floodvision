import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()

        if smooth <= 0:
            raise ValueError(
                f"smooth must be positive. Got {smooth}"
            )

        self.smooth = smooth

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        if not isinstance(predictions, torch.Tensor):
            raise TypeError(
                "predictions must be a torch.Tensor"
            )

        if not isinstance(targets, torch.Tensor):
            raise TypeError(
                "targets must be a torch.Tensor"
            )

        if predictions.ndim != 4:
            raise ValueError(
                f"Expected predictions to have shape "
                f"(B,C,H,W). Got {predictions.shape}"
            )

        if targets.ndim != 4:
            raise ValueError(
                f"Expected targets to have shape "
                f"(B,C,H,W). Got {targets.shape}"
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch.\n"
                f"Predictions: {predictions.shape}\n"
                f"Targets: {targets.shape}"
            )

        if predictions.shape[1] != 1:
            raise ValueError(
                f"Expected prediction channel size of 1. "
                f"Got {predictions.shape[1]}"
            )

        if targets.shape[1] != 1:
            raise ValueError(
                f"Expected target channel size of 1. "
                f"Got {targets.shape[1]}"
            )

        predictions = torch.sigmoid(predictions)

        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (predictions * targets).sum()

        dice_score = (
            (2.0 * intersection + self.smooth)
            /
            (
                predictions.sum()
                + targets.sum()
                + self.smooth
            )
        )

        loss = 1.0 - dice_score

        return loss