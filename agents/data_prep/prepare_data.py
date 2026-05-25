"""
DataPrepAgent — on-the-fly batch data preparation.

Mỗi khi training cần 1 batch:
  1. Nhận list records (từ JSON) gồm keys, pids, clinical_data, questions
  2. Kiểm tra NPY cache local — download + convert những file còn thiếu
  3. Trả về list dict {slices, prompt, answer} sẵn sàng đưa vào model
  4. Tự evict NPY cũ khi cache vượt max_cache_gb

DICOM luôn bị xóa ngay sau khi convert xong → chỉ giữ NPY cache.
"""

import random
import time
from pathlib import Path

import numpy as np
import pydicom
from idc_index import IDCClient

from .clinical_text import clinical_to_text
from .ct_processor import load_and_slice


# ---------------------------------------------------------------------------
# DICOM → NPY (internal)
# ---------------------------------------------------------------------------

def _sort_key(ds):
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return int(getattr(ds, "InstanceNumber", 0))


def _dicom_dir_to_npy(dicom_dir: Path, npy_out: Path) -> None:
    files = []
    for p in dicom_dir.rglob("*"):
        if p.is_file():
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if hasattr(ds, "PixelData"):
                    files.append(ds)
            except Exception:
                pass

    if not files:
        raise RuntimeError(f"No DICOM slices in {dicom_dir}")

    files = sorted(files, key=_sort_key)
    slices = []
    for ds in files:
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(arr * slope + intercept)

    npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_out), np.stack(slices, axis=0))


# ---------------------------------------------------------------------------
# DataPrepAgent
# ---------------------------------------------------------------------------

class DataPrepAgent:
    """
    Args:
        cache_dir:     local directory to store cached NPY files
        max_cache_gb:  evict oldest NPY files when cache exceeds this limit (from configs/model.yaml)
        max_slices:    max CT slices passed to MedGemma (from configs/model.yaml)
    """

    def __init__(
        self,
        cache_dir: str,
        max_slices: int,
        max_cache_gb: float,
    ):
        self.cache_dir    = Path(cache_dir)
        self.max_cache_gb = max_cache_gb
        self.max_slices   = max_slices
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None   # lazy init — IDCClient takes a few seconds to load
        self._df     = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_batch(self, records: list[dict]) -> list[dict]:
        """
        Args:
            records: list of raw JSON records, each must have:
                     keys, pids, clinical_data, questions, (answer computed by label_fn externally)

        Returns:
            list of dicts: {slices, prompt, answer, pid}
            Records that fail to download/convert are skipped with a warning.
        """
        # Step 1: ensure NPY exists for every record
        self._ensure_npys(records)

        # Step 2: build output batch
        results = []
        for rec in records:
            npy_path = self._npy_path(rec["keys"], str(rec.get("pids", "unknown")))
            if not npy_path.exists():
                print(f"[DataPrepAgent] SKIP {rec['keys']} — NPY not available")
                continue
            try:
                slices   = load_and_slice(str(npy_path), self.max_slices)
                clinical = clinical_to_text(rec.get("clinical_data", {}))
                question = rec.get("_question") or random.choice(rec["questions"])
                results.append({
                    "slices":  slices,
                    "prompt":  f"{clinical}\nQuestion: {question}",
                    "answer":  rec.get("_answer", ""),
                    "pid":     str(rec.get("pids", "")),
                })
            except Exception as e:
                print(f"[DataPrepAgent] ERROR loading {rec['keys']}: {e}")

        # Step 3: evict if over budget
        self._evict_if_needed()

        return results

    def get_npy_path(self, series_uid: str, pid: str) -> Path | None:
        """Return cached NPY path for a single series, downloading if needed."""
        npy_path = self._npy_path(series_uid, pid)
        if not npy_path.exists():
            self._download_and_convert(series_uid, pid)
        return npy_path if npy_path.exists() else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _npy_path(self, series_uid: str, pid: str) -> Path:
        return self.cache_dir / pid / f"{series_uid}.npy"

    def _ensure_npys(self, records: list[dict]) -> None:
        missing = [
            r for r in records
            if not self._npy_path(r["keys"], str(r.get("pids", "unknown"))).exists()
        ]
        if not missing:
            return

        self._init_client()
        for rec in missing:
            self._download_and_convert(rec["keys"], str(rec.get("pids", "unknown")))

    def _init_client(self) -> None:
        if self._client is None:
            print("[DataPrepAgent] Initializing IDC client...")
            self._client = IDCClient()
            self._df     = self._client.index

    def _download_and_convert(self, series_uid: str, pid: str) -> None:
        npy_out  = self._npy_path(series_uid, pid)
        dcm_dir  = self.cache_dir / "_dicom_tmp" / pid / series_uid

        # Check series exists on IDC and is CT
        match = self._df[self._df["SeriesInstanceUID"] == series_uid]
        if match.empty or match["Modality"].values[0] != "CT":
            print(f"[DataPrepAgent] SKIP {series_uid} — not found on IDC or not CT")
            return

        # Download
        dcm_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(dcm_dir),
            )
        except Exception as e:
            print(f"[DataPrepAgent] Download FAILED {series_uid}: {e}")
            return

        # Convert
        try:
            _dicom_dir_to_npy(dcm_dir, npy_out)
        except Exception as e:
            print(f"[DataPrepAgent] Convert FAILED {series_uid}: {e}")
        finally:
            # Always delete DICOM regardless of success — save disk space
            import shutil
            shutil.rmtree(dcm_dir, ignore_errors=True)

    def _evict_if_needed(self) -> None:
        npy_files = sorted(
            self.cache_dir.rglob("*.npy"),
            key=lambda p: p.stat().st_atime,   # oldest access first
        )
        total_gb = sum(p.stat().st_size for p in npy_files) / 1e9
        while total_gb > self.max_cache_gb and npy_files:
            oldest = npy_files.pop(0)
            freed  = oldest.stat().st_size / 1e9
            oldest.unlink()
            total_gb -= freed
            print(f"[DataPrepAgent] Evicted {oldest.name} ({freed:.2f} GB freed)")
