from pathlib import Path

import numpy as np
from PIL import Image


def _hu_window(s: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return ((np.clip(s, wmin, wmax) - wmin) / (wmax - wmin) * 255).astype(np.uint8)


def volume_to_slices(volume: np.ndarray, max_slices: int) -> list[Image.Image]:
    """
    Sample max_slices evenly từ volume HU [Z, H, W], trả về PIL RGB images.
    3-channel HU windowing theo MedGemma 1.5 technical report:
      R: bone and lung  [-1024, 1024]  (WW=2048, WL=0)
      G: soft tissue    [ -135,  215]  (WW= 350, WL=40)
      B: brain          [    0,   80]  (WW=  80, WL=40)
    MedGemma processor handles resizing to 896x896 internally.
    """
    volume = volume.astype(np.float32)
    Z = volume.shape[0]
    indices = np.linspace(0, Z - 1, min(Z, max_slices), dtype=int)

    pil_slices = []
    for idx in indices:
        s = volume[idx]
        r = _hu_window(s, -1024.0, 1024.0)
        g = _hu_window(s,  -135.0,  215.0)
        b = _hu_window(s,     0.0,   80.0)
        pil_slices.append(Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB"))

    return pil_slices


def load_and_slice(npy_path: str, max_slices: int) -> list[Image.Image]:
    """Load full CT volume từ .npy, sample max_slices evenly → PIL RGB images."""
    return volume_to_slices(np.load(npy_path), max_slices)


# ---------------------------------------------------------------------------
# DICOM series → HU volume (khớp logic dicom_dir_to_npy trong build_npy_hub.py)
# ---------------------------------------------------------------------------

def _dicom_sort_key(ds):
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return int(getattr(ds, "InstanceNumber", 0))


def dicom_dir_to_volume(dicom_dir: str | Path) -> np.ndarray:
    """
    Đọc 1 thư mục chứa các file .dcm của 1 CT series, trả về volume HU
    dạng [Z, H, W] (float16) — đã sort đúng thứ tự không gian + áp dụng
    RescaleSlope/RescaleIntercept để ra giá trị Hounsfield Unit thật.
    """
    import pydicom

    files = []
    for p in Path(dicom_dir).rglob("*"):
        if p.is_file():
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if hasattr(ds, "PixelData"):
                    files.append(ds)
            except Exception:
                pass
    if not files:
        raise RuntimeError(f"No DICOM slices in {dicom_dir}")

    files = sorted(files, key=_dicom_sort_key)
    slices = []
    for ds in files:
        arr       = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append((arr * slope + intercept).astype(np.float16))

    return np.stack(slices, axis=0)


def dicom_dir_to_slices(dicom_dir: str | Path, max_slices: int) -> list[Image.Image]:
    """Convert thẳng 1 thư mục DICOM series → PIL RGB slices (gộp 2 bước)."""
    return volume_to_slices(dicom_dir_to_volume(dicom_dir), max_slices)
