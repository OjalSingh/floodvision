from pathlib import Path

import cv2


IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")


def analyze_dataset():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR.resolve()}"
        )

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Mask directory not found: {MASK_DIR.resolve()}"
        )

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    mask_paths = sorted(
        MASK_DIR.glob("*.png")
    )

    if len(image_paths) == 0:
        raise ValueError(
            "No images found."
        )

    if len(mask_paths) == 0:
        raise ValueError(
            "No masks found."
        )

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            "Number of images and masks do not match."
        )

    valid_samples = 0
    invalid_samples = []

    smallest_height = float("inf")
    smallest_width = float("inf")

    largest_height = 0
    largest_width = 0

    flood_pixels = 0
    total_pixels = 0

    for image_path, mask_path in zip(
        image_paths,
        mask_paths
    ):
        image = cv2.imread(
            str(image_path)
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            invalid_samples.append(
                (
                    image_path.name,
                    "Image could not be loaded."
                )
            )
            continue

        if mask is None:
            invalid_samples.append(
                (
                    mask_path.name,
                    "Mask could not be loaded."
                )
            )
            continue

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]

        if (
            image_height != mask_height
            or image_width != mask_width
        ):
            invalid_samples.append(
                (
                    image_path.stem,
                    (
                        image_height,
                        image_width
                    ),
                    (
                        mask_height,
                        mask_width
                    )
                )
            )
            continue

        smallest_height = min(
            smallest_height,
            image_height
        )

        smallest_width = min(
            smallest_width,
            image_width
        )

        largest_height = max(
            largest_height,
            image_height
        )

        largest_width = max(
            largest_width,
            image_width
        )

        binary_mask = (
            mask > 127
        )

        flood_pixels += binary_mask.sum()
        total_pixels += binary_mask.size

        valid_samples += 1

    if valid_samples == 0:
        raise RuntimeError(
            "No valid image-mask pairs found."
        )

    flood_ratio = (
        flood_pixels / total_pixels
    ) * 100

    print("\nDataset Analysis")
    print("----------------")
    print(
        f"Total Samples     : {len(image_paths)}"
    )
    print(
        f"Valid Samples     : {valid_samples}"
    )
    print(
        f"Invalid Samples   : {len(invalid_samples)}"
    )

    print("\nImage Resolution")
    print("----------------")
    print(
        f"Smallest Image    : "
        f"{smallest_height} x {smallest_width}"
    )
    print(
        f"Largest Image     : "
        f"{largest_height} x {largest_width}"
    )

    print("\nFlood Statistics")
    print("----------------")
    print(
        f"Flood Pixel Ratio : "
        f"{flood_ratio:.2f}%"
    )

    if invalid_samples:
        print("\nInvalid Samples")
        print("----------------")

        for sample in invalid_samples:
            print(sample)


def main():
    analyze_dataset()


if __name__ == "__main__":
    main()
