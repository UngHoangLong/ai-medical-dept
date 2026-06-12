"""
Stratified scan selection for each agent's LoRA training subset.

Radiology uses 2 pools:
  Pool A (200 scans, all 12 tasks): scans present in ALL 12 tasks → nodule_presence=1
  Pool B (150 scans, 8 tasks):      scans in nodule_presence=0, dùng cho chest_abn + nodule_presence
  → Sequential inference: Bước 1 (screening) dùng cả A+B; Bước 2 (nodule detail) chỉ dùng A.

Stratification:
  - Binary tasks: ~50/50 positive/negative
  - Categorical tasks: tất cả classes đều có mặt

Output (in --output-dir):
  radiology_pool_a_keys.txt / _pid_map.json   (200 scans, 12 tasks)
  radiology_pool_b_keys.txt / _pid_map.json   (150 scans, 8 tasks)
  cardiology_keys.txt       / _pid_map.json
  oncology_keys.txt         / _pid_map.json
  subset_manifest.json      — task list per pool, dùng cho build_labels_subset.py

Usage:
    python scripts/select_subset.py \\
        --data-json-dir data/data_json \\
        --output-dir    data/subset_keys \\
        --n-radio-a     200 \\
        --n-radio-b     150 \\
        --n-cardiology  1500 \\
        --n-oncology    2000
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

RADIO_TASKS_A = [
    "chest_abn_54", "chest_abn_55", "chest_abn_56", "chest_abn_57",
    "chest_abn_58", "chest_abn_59", "chest_abn_61",
    "nodule_presence",
    "nodule_attenuation", "nodule_location", "nodule_margin", "nodule_size",
]
RADIO_TASKS_B = [
    "chest_abn_54", "chest_abn_55", "chest_abn_56", "chest_abn_57",
    "chest_abn_58", "chest_abn_59", "chest_abn_61",
    "nodule_presence",
]
RADIO_BINARY      = {
    "chest_abn_54", "chest_abn_55", "chest_abn_56", "chest_abn_57",
    "chest_abn_58", "chest_abn_59", "chest_abn_61", "nodule_presence",
}
RADIO_CATEGORICAL = {"nodule_attenuation", "nodule_location", "nodule_margin", "nodule_size"}

NODULE_CHAR_TASKS = {"nodule_attenuation", "nodule_location", "nodule_margin", "nodule_size"}

CARDIOLOGY_TASKS  = ["CVD_diagnosis", "CVD_mortality"]
CARDIOLOGY_BINARY = {"CVD_diagnosis", "CVD_mortality"}

ONCOLOGY_TASKS    = ["lung_cancer_risk"]
ONCOLOGY_BINARY   = {"lung_cancer_risk"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_task(data_json_dir: Path, task: str) -> dict[str, dict]:
    """Load trainval JSON → {key: {labels, pids}}"""
    path = data_json_dir / f"{task}_trainval.json"
    result = {}
    for r in json.load(open(path)):
        key = str(r["keys"])
        result[key] = {"labels": r["labels"], "pids": str(r.get("pids", "unknown"))}
    return result


def scalar_label(labels, task: str) -> int:
    if task == "lung_cancer_risk":
        return int(labels["y"]) if isinstance(labels, dict) else 0
    return int(labels)


def make_pid_map(keys: list[str], ref_task_data: dict[str, dict]) -> dict[str, str]:
    return {k: ref_task_data[k]["pids"] for k in keys if k in ref_task_data}


# ---------------------------------------------------------------------------
# Stratified sampler (generic)
# ---------------------------------------------------------------------------

def stratified_sample(
    task_data:         dict[str, dict[str, dict]],
    binary_tasks:      set[str],
    categorical_tasks: set[str],
    n_target:          int,
    restrict_to:       Optional[set] = None,
    seed:              int = 42,
) -> list[str]:
    """
    restrict_to: if provided, only consider these scan keys (used for Pool A).
    """
    rng   = random.Random(seed)
    tasks = list(task_data.keys())

    # Common keys
    common = set(task_data[tasks[0]].keys())
    for t in tasks[1:]:
        common &= set(task_data[t].keys())
    if restrict_to is not None:
        common &= restrict_to
    common = list(common)
    print(f"  Eligible scans: {len(common):,}")

    key_lbl = {k: {t: scalar_label(task_data[t][k]["labels"], t) for t in tasks} for k in common}

    # Binary report
    bin_in = [t for t in binary_tasks if t in task_data]
    print("  Binary positive rates:")
    for t in bin_in:
        pos = sum(1 for k in common if key_lbl[k][t] == 1)
        print(f"    {t}: {pos:,}/{len(common):,} ({100*pos/max(len(common),1):.1f}%)")

    # Categorical class maps
    cat_in = [t for t in categorical_tasks if t in task_data]
    cat_class_keys: dict[str, dict[int, list[str]]] = {}
    for t in cat_in:
        cm: dict[int, list[str]] = defaultdict(list)
        for k in common:
            cm[key_lbl[k][t]].append(k)
        cat_class_keys[t] = dict(cm)
        print(f"  {t} classes: { {c: len(v) for c, v in sorted(cm.items())} }")

    # Step 1: Force categorical coverage
    total_cls    = sum(len(v) for v in cat_class_keys.values())
    per_cls_quota = max(2, n_target // max(total_cls, 1))
    forced: set[str] = set()
    for t, cm in cat_class_keys.items():
        for cls, cands in sorted(cm.items()):
            forced.update(rng.sample(cands, min(per_cls_quota, len(cands))))
    print(f"  Forced (categorical): {len(forced)}")

    # Step 2: Balance binary in remaining quota
    remaining = max(0, n_target - len(forced))
    pool      = [k for k in common if k not in forced]
    positive  = list({k for k in pool if any(key_lbl[k][t] == 1 for t in bin_in)})
    negative  = [k for k in pool if k not in set(positive)]
    half      = remaining // 2
    sel_pos   = rng.sample(positive, min(half, len(positive)))
    sel_neg   = rng.sample(negative, min(remaining - len(sel_pos), len(negative)))

    selected = list(forced) + sel_pos + sel_neg
    if len(selected) > n_target:
        selected = rng.sample(selected, n_target)

    print(f"  Selected: {len(selected):,} scans")
    return selected


# ---------------------------------------------------------------------------
# Radiology Pool B: nodule_presence=0 scans
# ---------------------------------------------------------------------------

def select_radio_pool_b(
    task_data_all: dict[str, dict[str, dict]],
    pool_a_keys:   set[str],
    n_target:      int,
    seed:          int = 42,
) -> list[str]:
    """
    Pool B = scans in nodule_presence (label=0) that are also in chest_abn tasks.
    Excludes Pool A scans. Stratifies on chest_abn binary tasks.
    """
    rng = random.Random(seed)

    nodule_pres = task_data_all["nodule_presence"]
    nodule_char_keys = set()
    for t in NODULE_CHAR_TASKS:
        if t in task_data_all:
            nodule_char_keys |= set(task_data_all[t].keys())

    # nodule_presence=0: in nodule_presence file, NOT in characterization tasks, NOT in Pool A
    neg_nodule_keys = {
        k for k, v in nodule_pres.items()
        if scalar_label(v["labels"], "nodule_presence") == 0
        and k not in nodule_char_keys
        and k not in pool_a_keys
    }
    print(f"\n  Pool B — nodule_presence=0 scans (excl. Pool A): {len(neg_nodule_keys):,}")

    # Must also be in ALL chest_abn tasks
    chest_tasks = [t for t in RADIO_TASKS_B if t.startswith("chest_abn")]
    eligible = neg_nodule_keys.copy()
    for t in chest_tasks:
        if t in task_data_all:
            eligible &= set(task_data_all[t].keys())
    eligible = list(eligible)
    print(f"  Pool B — also in all chest_abn tasks: {len(eligible):,}")

    # Build binary label map for chest_abn tasks
    bin_in = [t for t in RADIO_TASKS_B if t in task_data_all and t != "nodule_presence"]
    key_lbl = {k: {t: scalar_label(task_data_all[t][k]["labels"], t) for t in bin_in} for k in eligible}

    print("  Pool B binary positive rates:")
    for t in bin_in:
        pos = sum(1 for k in eligible if key_lbl[k][t] == 1)
        print(f"    {t}: {pos:,}/{len(eligible):,} ({100*pos/max(len(eligible),1):.1f}%)")

    # Balance binary tasks
    positive = list({k for k in eligible if any(key_lbl[k][t] == 1 for t in bin_in)})
    negative = [k for k in eligible if k not in set(positive)]
    half     = n_target // 2
    sel_pos  = rng.sample(positive, min(half, len(positive)))
    sel_neg  = rng.sample(negative, min(n_target - len(sel_pos), len(negative)))

    selected = sel_pos + sel_neg
    if len(selected) > n_target:
        selected = rng.sample(selected, n_target)
    print(f"  Pool B selected: {len(selected):,} scans")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json-dir", required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--n-radio-a",     type=int, default=200)
    parser.add_argument("--n-radio-b",     type=int, default=150)
    parser.add_argument("--n-cardiology",  type=int, default=1500)
    parser.add_argument("--n-oncology",    type=int, default=2000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_json_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Radiology ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RADIOLOGY")
    print(f"{'='*55}")

    print("\n  Loading all radiology task files...")
    all_radio_tasks = list(dict.fromkeys(RADIO_TASKS_A + RADIO_TASKS_B))
    task_data_radio = {t: load_task(data_dir, t) for t in all_radio_tasks if (data_dir / f"{t}_trainval.json").exists()}

    # Pool A: scans in all 12 tasks
    print(f"\n  [Pool A] target {args.n_radio_a} scans — all 12 tasks")
    task_data_a = {t: task_data_radio[t] for t in RADIO_TASKS_A if t in task_data_radio}
    pool_a = stratified_sample(task_data_a, RADIO_BINARY, RADIO_CATEGORICAL, args.n_radio_a, seed=args.seed)

    # Pool B: nodule_presence=0, chest_abn tasks only
    print(f"\n  [Pool B] target {args.n_radio_b} scans — chest_abn + nodule_presence (label=0)")
    pool_b = select_radio_pool_b(task_data_radio, set(pool_a), args.n_radio_b, seed=args.seed)

    # Save Pool A
    ref_a = RADIO_TASKS_A[0]
    (out_dir / "radiology_pool_a_keys.txt").write_text("\n".join(sorted(pool_a)))
    json.dump(make_pid_map(pool_a, task_data_radio[ref_a]), open(out_dir / "radiology_pool_a_pid_map.json", "w"), indent=2)

    # Save Pool B
    ref_b = RADIO_TASKS_B[0]
    (out_dir / "radiology_pool_b_keys.txt").write_text("\n".join(sorted(pool_b)))
    json.dump(make_pid_map(pool_b, task_data_radio[ref_b]), open(out_dir / "radiology_pool_b_pid_map.json", "w"), indent=2)

    # Balance report
    print(f"\n  Pool A label balance:")
    for t in RADIO_TASKS_A:
        if t not in task_data_radio: continue
        c = Counter(scalar_label(task_data_radio[t][k]["labels"], t) for k in pool_a if k in task_data_radio[t])
        print(f"    {t}: {dict(sorted(c.items()))}")

    print(f"\n  Pool B label balance:")
    for t in RADIO_TASKS_B:
        if t not in task_data_radio: continue
        c = Counter(scalar_label(task_data_radio[t][k]["labels"], t) for k in pool_b if k in task_data_radio[t])
        print(f"    {t}: {dict(sorted(c.items()))}")

    # ── Cardiology ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CARDIOLOGY  —  target {args.n_cardiology} scans")
    print(f"{'='*55}")
    td_cardio = {t: load_task(data_dir, t) for t in CARDIOLOGY_TASKS if (data_dir / f"{t}_trainval.json").exists()}
    cardio = stratified_sample(td_cardio, CARDIOLOGY_BINARY, set(), args.n_cardiology, seed=args.seed)
    (out_dir / "cardiology_keys.txt").write_text("\n".join(sorted(cardio)))
    json.dump(make_pid_map(cardio, td_cardio[CARDIOLOGY_TASKS[0]]), open(out_dir / "cardiology_pid_map.json", "w"), indent=2)
    print(f"\n  Label balance:")
    for t in CARDIOLOGY_TASKS:
        if t not in td_cardio: continue
        c = Counter(scalar_label(td_cardio[t][k]["labels"], t) for k in cardio if k in td_cardio[t])
        print(f"    {t}: {dict(sorted(c.items()))}")

    # ── Oncology ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ONCOLOGY  —  target {args.n_oncology} scans")
    print(f"{'='*55}")
    td_onco = {t: load_task(data_dir, t) for t in ONCOLOGY_TASKS if (data_dir / f"{t}_trainval.json").exists()}
    onco = stratified_sample(td_onco, ONCOLOGY_BINARY, set(), args.n_oncology, seed=args.seed)
    (out_dir / "oncology_keys.txt").write_text("\n".join(sorted(onco)))
    json.dump(make_pid_map(onco, td_onco[ONCOLOGY_TASKS[0]]), open(out_dir / "oncology_pid_map.json", "w"), indent=2)
    print(f"\n  Label balance:")
    for t in ONCOLOGY_TASKS:
        if t not in td_onco: continue
        c = Counter(scalar_label(td_onco[t][k]["labels"], t) for k in onco if k in td_onco[t])
        print(f"    {t}: {dict(sorted(c.items()))}")

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "radiology": {
            "pool_a": {"keys_file": "radiology_pool_a_keys.txt", "tasks": RADIO_TASKS_A,  "n": len(pool_a)},
            "pool_b": {"keys_file": "radiology_pool_b_keys.txt", "tasks": RADIO_TASKS_B,  "n": len(pool_b)},
        },
        "cardiology": {"keys_file": "cardiology_keys.txt", "tasks": CARDIOLOGY_TASKS, "n": len(cardio)},
        "oncology":   {"keys_file": "oncology_keys.txt",   "tasks": ONCOLOGY_TASKS,   "n": len(onco)},
    }
    json.dump(manifest, open(out_dir / "subset_manifest.json", "w"), indent=2)
    print(f"\n  → subset_manifest.json")

    # Summary
    total_radio  = len(pool_a) * len(RADIO_TASKS_A) + len(pool_b) * len(RADIO_TASKS_B)
    total_cardio = len(cardio) * len(CARDIOLOGY_TASKS)
    total_onco   = len(onco)   * len(ONCOLOGY_TASKS)
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Radiology  Pool A: {len(pool_a):>4} scans × 12 tasks = {len(pool_a)*12:,} samples")
    print(f"  Radiology  Pool B: {len(pool_b):>4} scans ×  8 tasks = {len(pool_b)*8:,} samples")
    print(f"  Radiology  total : {total_radio:,} samples")
    print(f"  Cardiology total : {total_cardio:,} samples")
    print(f"  Oncology   total : {total_onco:,} samples")
    unique_scans = len(set(pool_a) | set(pool_b) | set(cardio) | set(onco))
    print(f"\n  Unique scans to upload: ~{unique_scans:,}")
    print(f"  Est. NPY storage      : ~{unique_scans * 86 / 1024:.1f} GB")
    print("\nDone.")


if __name__ == "__main__":
    main()
