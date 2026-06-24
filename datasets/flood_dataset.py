
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FloodDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: tuple[int, int] = (256, 256)
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}"
            )

        self.image_paths = sorted(
            [
                file
                for file in self.image_dir.iterdir()
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        self.mask_paths = sorted(
            [
                file
                for file in self.mask_dir.iterdir()
                if file.suffix.lower() == ".png"
            ]
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No image files found in {self.image_dir}"
            )

        if len(self.mask_paths) == 0:
            raise ValueError(
                f"No mask files found in {self.mask_dir}"
            )

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image count ({len(self.image_paths)}) "
                f"does not match mask count ({len(self.mask_paths)})"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(
                f"Failed to load image: {image_path}"
            )

        if mask is None:
            raise ValueError(
                f"Failed to load mask: {mask_path}"
            )

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]

        if image_height != mask_height or image_width != mask_width:
            raise ValueError(
                f"Dimension mismatch:\n"
                f"Image: {image_path.name} -> "
                f"{image_height}x{image_width}\n"
                f"Mask: {mask_path.name} -> "
                f"{mask_height}x{mask_width}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )

        mask = (mask > 127).astype(np.float32)

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
