"""
DataPrepAgent — tải NPY từ HuggingFace Hub, convert HU→RGB, build training batch.

Flow:
  1. Nhận list joined records (labels đã merge) + formatter
  2. Download NPY từ HF Hub (auto cache local)
  3. Convert HU→RGB qua ct_processor
  4. Gọi formatter.format_sample() → prompt + answer
  5. Trả về list dict {slices, prompt, answer, pid}
  6. Evict NPY cũ nếu cache vượt max_cache_gb
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

from .ct_processor import load_and_slice

NPY_REPO_ID   = "UngLong/openm3chest-npy-v2"
NPY_REPO_TYPE = "dataset"


class DataPrepAgent:
    """
    Args:
        cache_dir:    local dir để lưu NPY cache
        max_slices:   số slices tối đa đưa vào MedGemma (85 theo paper)
        max_cache_gb: evict NPY cũ khi cache vượt giới hạn này
    """

    def __init__(self, cache_dir: str, max_slices: int, max_cache_gb: float):
        self.cache_dir    = Path(cache_dir)
        self.max_slices   = max_slices
        self.max_cache_gb = max_cache_gb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch(self, records: list[dict], formatter) -> list[dict]:
        """
        Args:
            records:   list joined records, mỗi record có: keys, pids, clinical_data, labels
            formatter: instance có method format_sample(record) → {prompt, answer, keys, pids}

        Returns:
            list of {slices, prompt, answer, pid}
        """
        results = []
        for rec in records:
            npy_path = self._get_npy(rec["keys"], str(rec.get("pids", "unknown")))
            if npy_path is None:
                print(f"[DataPrepAgent] SKIP {rec['keys']} — NPY not available")
                continue
            try:
                formatted = formatter.format_sample(rec)
                slices    = load_and_slice(npy_path, self.max_slices)
                results.append({
                    "slices": slices,
                    "prompt": formatted["prompt"],
                    "answer": formatted["answer"],
                    "pid":    formatted["pids"],
                })
            except Exception as e:
                print(f"[DataPrepAgent] ERROR {rec['keys']}: {e}")

        self._evict_if_needed()
        return results

    def get_npy_path(self, series_uid: str, pid: str) -> str | None:
        return self._get_npy(series_uid, pid)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_npy(self, series_uid: str, pid: str) -> str | None:
        try:
            return hf_hub_download(
                repo_id    = NPY_REPO_ID,
                filename   = f"{pid}/{series_uid}.npy",
                repo_type  = NPY_REPO_TYPE,
                local_dir  = str(self.cache_dir),
            )
        except Exception as e:
            print(f"[DataPrepAgent] Download FAILED {series_uid}: {e}")
            return None

    def _evict_if_needed(self) -> None:
        npy_files = sorted(
            self.cache_dir.rglob("*.npy"),
            key=lambda p: p.stat().st_atime,
        )
        total_gb = sum(p.stat().st_size for p in npy_files) / 1e9
        while total_gb > self.max_cache_gb and npy_files:
            oldest   = npy_files.pop(0)
            freed    = oldest.stat().st_size / 1e9
            oldest.unlink()
            total_gb -= freed
            print(f"[DataPrepAgent] Evicted {oldest.name} ({freed:.2f} GB freed)")
