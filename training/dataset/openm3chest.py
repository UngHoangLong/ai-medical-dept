import random
from typing import Callable

from torch.utils.data import Dataset

from agents.data_prep.prepare_data import DataPrepAgent


class OpenM3ChestDataset(Dataset):
    """
    Streaming instruction-tuning dataset cho 1 task.

    records: list of dicts từ HuggingFace load_dataset() hoặc local JSON
    Mỗi record cần có: keys, pids, clinical_data, questions, labels

    Mỗi __getitem__:
      - Gọi DataPrepAgent để đảm bảo NPY tồn tại (download nếu chưa có)
      - Random pick 1 trong 10 câu hỏi
      - Build prompt = clinical text + question
      - Trả về {slices, prompt, answer, pid}
    """

    def __init__(
        self,
        records,                    # HuggingFace Dataset hoặc list[dict]
        label_fn: Callable,         # callable(record) -> str
        data_prep: DataPrepAgent,
    ):
        self.records   = records
        self.label_fn  = label_fn
        self.data_prep = data_prep

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record     = dict(self.records[idx])   # HF Dataset trả về dict-like, convert để an toàn
        series_uid = record["keys"]
        pid        = str(record.get("pids", "unknown"))

        record["_answer"]   = self.label_fn(record)
        record["_question"] = random.choice(record["questions"])

        results = self.data_prep.prepare_batch([record])
        if not results:
            raise RuntimeError(f"NPY not available for series {series_uid} (pid={pid})")

        return results[0]
