import json
from pathlib import Path

from torch.utils.data import Dataset

from agents.data_prep.clinical_text import clinical_to_text
from agents.data_prep.ct_processor import load_and_slice


class OpenM3ChestDataset(Dataset):
    """
    Instruction-tuning dataset for one OpenM3Chest task file.
    Each item: (list[PIL.Image], question_text, answer_text)
    """

    def __init__(
        self,
        json_path: str,
        task_name: str,
        question: str,
        label_fn,           # callable(record) -> str
        bbox_key: str = "lung_bboxes",
        max_slices: int = 85,
        image_size: int = 896,
    ):
        self.records   = json.loads(Path(json_path).read_text())
        self.task_name = task_name
        self.question  = question
        self.label_fn  = label_fn
        self.bbox_key  = bbox_key
        self.max_slices = max_slices
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        slices = load_and_slice(
            npy_path=record["npy_path"],
            bbox=record[self.bbox_key],
            max_slices=self.max_slices,
            image_size=self.image_size,
        )

        clinical_text = clinical_to_text(record.get("clinical_data", {}), self.question)
        answer = self.label_fn(record)

        return {
            "slices": slices,
            "clinical_text": clinical_text,
            "answer": answer,
            "pid": record.get("pids", ""),
        }
