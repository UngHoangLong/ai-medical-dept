"""
Test DataPrepAgent với data từ HuggingFace Hub — không cần JSON local.

Dùng split "test" của chest_abn_54 để lấy vài records,
trỏ cache_dir vào thư mục NPY local đã có sẵn.
Nếu NPY chưa có → DataPrepAgent tự download từ IDC.

Usage (chạy từ thư mục ai-medical-dept/):
    python scripts/test_data_prep_local.py \
        --npy-dir  /Users/macbook/Documents/Demo_AI/data/npy_ct_images_test \
        --n-samples 3
"""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset

from agents.data_prep.prepare_data import DataPrepAgent


def label_fn(record: dict) -> str:
    return {0: "No.", 1: "Yes."}.get(int(record["labels"]), str(record["labels"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir",   required=True, help="Thư mục cache NPY local")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--model-cfg", default="configs/model.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.model_cfg).read_text())

    # Load records từ HuggingFace Hub
    print("Loading dataset từ HuggingFace Hub...")
    hf_ds   = load_dataset("UngLong/openm3chest-labels", "chest_abn_54")
    records = [dict(hf_ds["test"][i]) for i in range(args.n_samples)]
    print(f"Lấy {len(records)} records từ split 'test'\n")

    # DataPrepAgent — trỏ cache_dir vào NPY local
    data_prep = DataPrepAgent(
        cache_dir    = args.npy_dir,
        max_slices   = cfg["data_prep"]["max_slices"],
        max_cache_gb = cfg["data_prep"]["max_cache_gb"],
    )

    # Gắn answer
    for r in records:
        r["_answer"] = label_fn(r)

    # Chạy pipeline
    results = data_prep.prepare_batch(records)

    # In kết quả
    print(f"{'='*60}")
    print(f"Kết quả: {len(results)}/{len(records)} records thành công\n")

    for i, res in enumerate(results):
        print(f"--- Sample {i+1} ---")
        print(f"PID    : {res['pid']}")
        print(f"Slices : {len(res['slices'])} ảnh, size={res['slices'][0].size}")
        print(f"Prompt :\n{res['prompt']}")
        print(f"Answer : {res['answer']}")
        print()


if __name__ == "__main__":
    main()
