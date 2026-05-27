import numpy as np
from PIL import Image


def _hu_window(s: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return ((np.clip(s, wmin, wmax) - wmin) / (wmax - wmin) * 255).astype(np.uint8)


def load_and_slice(npy_path: str, max_slices: int) -> list[Image.Image]:
    """
    Load full CT volume, sample max_slices evenly, return as PIL RGB images.
    3-channel HU windowing per MedGemma technical report (arXiv):
      R: bone and lung  [-1225, 1025]  (WW=2250, WL=-100)
      G: soft tissue    [ -135,  215]  (WW= 350, WL=  40)
      B: brain          [    0,   80]  (WW=  80, WL=  40)
    MedGemma processor handles resizing internally.
    """
    volume = np.load(npy_path).astype(np.float32)  # [Z, H, W]
    Z = volume.shape[0]
    indices = np.linspace(0, Z - 1, min(Z, max_slices), dtype=int)

    pil_slices = []
    for idx in indices:
        s = volume[idx]
        r = _hu_window(s, -1225.0, 1025.0)
        g = _hu_window(s,  -135.0,  215.0)
        b = _hu_window(s,     0.0,   80.0)
        pil_slices.append(Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB"))

    return pil_slices
