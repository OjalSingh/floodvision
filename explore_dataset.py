# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from pathlib import Path

# image_dir = Path("data/images")
# mask_dir = Path("data/masks")

# images = sorted(image_dir.glob("*"))
# masks = sorted(mask_dir.glob("*"))

# # image_shapes = set()
# # mask_shapes = set()

# # for img_path, mask_path in zip(images, masks):
# #     img = cv2.imread(str(img_path))
# #     mask = cv2.imread(str(mask_path), 0)

# #     image_shapes.add(img.shape)
# #     mask_shapes.add(mask.shape)

# # print("Image Shapes:", image_shapes)
# # print("Mask Shapes:", mask_shapes)


# for i in range(10):
#     mask = cv2.imread(str(masks[i]), 0)

#     print(
#         f"Mask {i}:",
#         np.min(mask),
#         np.max(mask),
#         len(np.unique(mask))
#     )


# mask = cv2.imread(str(masks[0]), 0)

# plt.imshow(mask, cmap="gray")
# plt.colorbar()
# plt.show()

# mask = cv2.imread(str(masks[0]), 0)

# print(np.unique(mask))


import cv2
from pathlib import Path

mask_dir = Path("data/masks")
mask_paths = list(mask_dir.glob("*"))

flood_pixels = 0
total_pixels = 0

for path in mask_paths:
    mask = cv2.imread(str(path), 0)

    mask = (mask > 127)

    flood_pixels += mask.sum()
    total_pixels += mask.size

print("Flood Ratio:", flood_pixels / total_pixels)