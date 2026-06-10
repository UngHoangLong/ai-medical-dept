"""
build_labels_subset.py — Filter labels theo subset keys và push lên HuggingFace.

Đọc subset_manifest.json (output của select_subset.py), filter từng task file,
push mỗi task thành 1 config riêng trong HuggingFace dataset.

Schema giữ nguyên như openm3chest-labels:
  keys, pids, clinical_data, questions, labels

Usage:
    python scripts/build_labels_subset.py \\
        --data-json-dir data/data_json \\
        --subset-dir    data/subset_keys \\
        --hf-repo       UngLong/openm3chest-labels-v2
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


# Default answer_dict for tasks missing it in source data
DEFAULT_ANSWER_DICT = {
    "CVD_mortality":    {"0": "Low risk", "1": "High risk"},
    "lung_cancer_risk": {"0": "No cancer within follow-up", "1": "Cancer within follow-up"},
}


# task → agent
TASK_AGENT = {
    "chest_abn_54":      "radiology",
    "chest_abn_55":      "radiology",
    "chest_abn_56":      "radiology",
    "chest_abn_57":      "radiology",
    "chest_abn_58":      "radiology",
    "chest_abn_59":      "radiology",
    "chest_abn_61":      "radiology",
    "nodule_presence":   "radiology",
    "nodule_attenuation":"radiology",
    "nodule_location":   "radiology",
    "nodule_margin":     "radiology",
    "nodule_size":       "radiology",
    "CVD_diagnosis":     "cardiology",
    "CVD_mortality":     "cardiology",
    "lung_cancer_risk":  "oncology",
}


def load_keys(subset_dir: Path, agent: str, pool: str | None = None) -> set[str]:
    if agent == "radiology":
        if pool == "a":
            fname = "radiology_pool_a_keys.txt"
        elif pool == "b":
            fname = "radiology_pool_b_keys.txt"
        else:
            # Union of both pools
            keys_a = load_keys(subset_dir, "radiology", "a")
            keys_b = load_keys(subset_dir, "radiology", "b")
            return keys_a | keys_b
    else:
        fname = f"{agent}_keys.txt"
    return set((subset_dir / fname).read_text().splitlines())


def normalize_labels(labels, task: str):
    """Serialize complex labels (e.g. lung_cancer_risk dict) to JSON string."""
    if isinstance(labels, (dict, list)):
        return json.dumps(labels)
    return labels


def filter_task(
    data_json_dir: Path,
    task: str,
    allowed_keys: set[str],
    split: str = "trainval",
) -> list[dict]:
    path = data_json_dir / f"{task}_{split}.json"
    if not path.exists():
        print(f"  [WARN] Not found: {path.name}")
        return []

    records = json.load(open(path))
    filtered = []
    for r in records:
        key = str(r.get("keys", ""))
        if key not in allowed_keys:
            continue
        answer_dict = r.get("answer_dict") or DEFAULT_ANSWER_DICT.get(task, {})
        filtered.append({
            "keys":          key,
            "pids":          str(r.get("pids", "unknown")),
            "clinical_data": json.dumps(r["clinical_data"]) if isinstance(r.get("clinical_data"), (dict, list)) else str(r.get("clinical_data", "")),
            "questions":     r.get("questions", []),
            "labels":        normalize_labels(r.get("labels"), task),
            "answer_dict":   json.dumps(answer_dict),
        })
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json-dir", required=True)
    parser.add_argument("--subset-dir",    required=True)
    parser.add_argument("--hf-repo",       required=True, help="vd: UngLong/openm3chest-labels-v2")
    args = parser.parse_args()

    data_dir   = Path(args.data_json_dir)
    subset_dir = Path(args.subset_dir)

    # Load manifest
    manifest = json.load(open(subset_dir / "subset_manifest.json"))

    # Load key sets per agent
    radio_keys    = load_keys(subset_dir, "radiology")   # Pool A ∪ Pool B
    cardio_keys   = load_keys(subset_dir, "cardiology")
    oncology_keys = load_keys(subset_dir, "oncology")

    agent_keys = {
        "radiology":  radio_keys,
        "cardiology": cardio_keys,
        "oncology":   oncology_keys,
    }

    print(f"Keys loaded:")
    for agent, keys in agent_keys.items():
        print(f"  {agent}: {len(keys):,} scans")

    # Create HF repo
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True, private=True)
    print(f"\nHuggingFace repo: {args.hf_repo}")

    # Process each task
    all_tasks = list(TASK_AGENT.keys())
    total_records = 0

    for task in all_tasks:
        agent  = TASK_AGENT[task]
        keys   = agent_keys[agent]

        print(f"\n[{task}] agent={agent}, keys={len(keys):,}")
        train_records = filter_task(data_dir, task, keys, split="trainval")
        test_records  = filter_task(data_dir, task, keys, split="test")

        if not train_records:
            print(f"  [SKIP] No records after filtering.")
            continue

        # test split: dùng toàn bộ test set (không filter theo subset)
        # để evaluation vẫn đại diện cho full distribution
        all_test_records = filter_task(data_dir, task, set(), split="test")
        if not all_test_records:
            # fallback: không filter test set gì cả
            path = data_dir / f"{task}_test.json"
            if path.exists():
                raw = json.load(open(path))
                all_test_records = [{
                    "keys":          str(r.get("keys", "")),
                    "pids":          str(r.get("pids", "unknown")),
                    "clinical_data": json.dumps(r["clinical_data"]) if isinstance(r.get("clinical_data"), (dict, list)) else str(r.get("clinical_data", "")),
                    "questions":     r.get("questions", []),
                    "labels":        normalize_labels(r.get("labels"), task),
                    "answer_dict":   json.dumps(r.get("answer_dict") or DEFAULT_ANSWER_DICT.get(task, {})),
                } for r in raw]

        ds = DatasetDict({
            "train": Dataset.from_list(train_records),
            "test":  Dataset.from_list(all_test_records) if all_test_records else Dataset.from_list([]),
        })

        print(f"  train: {len(ds['train']):,} records  |  test: {len(ds['test']):,} records")
        total_records += len(ds["train"])

        ds.push_to_hub(
            repo_id     = args.hf_repo,
            config_name = task,
            private     = True,
        )
        print(f"  ✓ Pushed config: {task}")

    print(f"\n{'='*50}")
    print(f"Total train records pushed: {total_records:,}")
    print(f"Hub: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
