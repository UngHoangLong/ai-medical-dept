import numpy as np
from PIL import Image


def load_and_slice(npy_path: str, max_slices: int) -> list[Image.Image]:
    """
    Load full CT volume, sample max_slices evenly, return as PIL RGB images.
    HU normalized [-1000, 400] → [0, 255].
    MedGemma processor handles resizing internally.
    """
    volume = np.load(npy_path)  # [Z, H, W]
    Z = volume.shape[0]
    indices = np.linspace(0, Z - 1, min(Z, max_slices), dtype=int)

    pil_slices = []
    for idx in indices:
        s = volume[idx]
        s_norm = ((np.clip(s, -1000, 400) + 1000) / 1400 * 255).astype(np.uint8)
        pil_slices.append(Image.fromarray(s_norm, mode="L").convert("RGB"))

    return pil_slices
