
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
        self.valid_pairs = []
        self.invalid_pairs = []

        for image_path, mask_path in zip(
            self.image_paths,
            self.mask_paths
        ):
            image = cv2.imread(str(image_path))
            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                self.invalid_pairs.append(
                    (
                        image_path.name,
                        "Image could not be loaded"
                    )
                )
                continue

            if mask is None:
                self.invalid_pairs.append(
                    (
                        mask_path.name,
                        "Mask could not be loaded"
                    )
                )
                continue

            image_height, image_width = image.shape[:2]
            mask_height, mask_width = mask.shape[:2]

            if (
                image_height != mask_height
                or image_width != mask_width
            ):
                self.invalid_pairs.append(
                    (
                        image_path.name,
                        f"Image={image_height}x{image_width}, "
                        f"Mask={mask_height}x{mask_width}"
                    )
                )
                continue
            
            self.valid_pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

        if len(self.valid_pairs) == 0:
            raise ValueError(
                "No valid image-mask pairs found."
            )

        print(
            f"Valid samples: {len(self.valid_pairs)}"
        )

        print(
            f"Invalid samples removed: "
            f"{len(self.invalid_pairs)}"
        )
        
        

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.valid_pairs[idx]

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
