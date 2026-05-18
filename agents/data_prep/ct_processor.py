import numpy as np
from PIL import Image


def load_and_slice(
    npy_path: str,
    bbox: list[int],
    max_slices: int = 85,
    image_size: int = 896,
) -> list[Image.Image]:
    """
    Load a 3D CT volume, crop to bbox, sample ≤max_slices axial slices,
    resize each to image_size × image_size, return as list of PIL Images.

    bbox format: [z_min, y_min, x_min, z_max, y_max, x_max]
    """
    volume = np.load(npy_path)  # [Z, H, W]

    z1, y1, x1, z2, y2, x2 = bbox
    crop = volume[z1:z2, y1:y2, x1:x2]  # [Z', H', W']

    Z = crop.shape[0]
    indices = np.linspace(0, Z - 1, min(Z, max_slices), dtype=int)
    slices = crop[indices]  # [N, H', W']

    pil_slices = []
    for s in slices:
        # Normalise HU → [0, 255]
        s_clipped = np.clip(s, -1000, 400)
        s_norm = ((s_clipped + 1000) / 1400 * 255).astype(np.uint8)
        img = Image.fromarray(s_norm, mode="L").convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        pil_slices.append(img)

    return pil_slices
