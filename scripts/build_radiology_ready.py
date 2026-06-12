"""
build_radiology_ready.py — Build ready dataset cho Radiology (screening + detail),
chọn subset CÂN BẰNG HƠN từ các scan ĐÃ CÓ NPY trên npy-v2 (0 conversion mới).

Chạy LOCAL (cần data_json + checkpoint trên máy này).

Flow:
  1. Pool NPY-ready = scan có đủ nhãn + đã upload NPY (theo checkpoint)
  2. Screening: lấy HẾT rare positive + fill negative → SCREEN_N scan
  3. Detail: inner join 4 nodule task (Pool A), trong pool NPY-ready
  4. Build prompt/response JSON (giống prepare-dataset-radiology.ipynb)
  5. Push ready dataset lên HF (hoặc --dry-run để xem trước)

Usage:
    python scripts/build_radiology_ready.py --dry-run
    python scripts/build_radiology_ready.py --push --out-repo UngLong/radiology-ready-v2
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

DATA_JSON   = Path("/Users/macbook/Documents/Demo_AI/data/data_json")
CHECKPOINT  = Path("/Users/macbook/Documents/Demo_AI/ai-medical-dept/scripts/npy_upload_checkpoint.txt")

SCREENING = ["chest_abn_54", "chest_abn_55", "chest_abn_56", "chest_abn_57",
             "chest_abn_58", "chest_abn_59", "chest_abn_61", "nodule_presence"]
DETAIL    = ["nodule_location", "nodule_attenuation", "nodule_margin", "nodule_size"]
RARE      = ["chest_abn_54", "chest_abn_55", "chest_abn_56", "chest_abn_57", "chest_abn_58"]


# ---------------------------------------------------------------------------
# clinical_to_text (bản có _valid, đồng bộ với notebook)
# ---------------------------------------------------------------------------

def _valid(v):
    if v is None:
        return False
    if isinstance(v, str):
        return v not in ("", "None")
    if isinstance(v, (int, float)):
        return v > 0
    return True


def clinical_to_text(cd) -> str:
    if isinstance(cd, str):
        cd = json.loads(cd)
    demo    = cd.get("demo", {}) or {}
    smoking = cd.get("smoking", {}) or {}
    disease = cd.get("disease_his", {}) or {}
    cancer  = cd.get("cancer_his", {}) or {}
    fam     = cd.get("fam_lc", {}) or {}
    parts = []
    if demo:
        tokens = []
        age, gender = demo.get("age"), (demo.get("gender", "") if _valid(demo.get("gender")) else "")
        race, ethnic, educat = demo.get("race"), demo.get("ethnic"), demo.get("educat")
        height, weight = demo.get("height"), demo.get("weight")
        if _valid(age):    tokens.append(f"{int(age)}-year-old {gender}".strip())
        if _valid(race):   tokens.append(race)
        if _valid(ethnic) and ethnic != "Neither Hispanic nor Latino": tokens.append(ethnic)
        if _valid(height) and _valid(weight):
            bmi = round(weight * 0.453592 / ((height * 0.0254) ** 2), 1)
            tokens.append(f"height {int(height)}in, weight {int(weight)}lbs (BMI {bmi})")
        if _valid(educat): tokens.append(f"education: {educat}")
        if tokens: parts.append("Patient: " + ", ".join(tokens) + ".")
    status = smoking.get("cigsmok", "")
    if _valid(status) and status != "Never":
        smk = f"Smoking: {status}"
        if _valid(smoking.get("pkyr")):     smk += f", {int(smoking['pkyr'])} pack-years"
        if _valid(smoking.get("smokeage")): smk += f", started age {int(smoking['smokeage'])}"
        if _valid(smoking.get("smokeday")): smk += f", {int(smoking['smokeday'])} cigarettes/day"
        if _valid(smoking.get("smokeyr")):  smk += f", {int(smoking['smokeyr'])} years"
        if status == "Former" and _valid(smoking.get("age_quit")): smk += f", quit age {int(smoking['age_quit'])}"
        if smoking.get("smokelive") == "Yes": smk += ", lives with smoker"
        if smoking.get("smokework") == "Yes": smk += ", works with smoker"
        if smoking.get("cigar") == "Yes":     smk += ", cigar smoker"
        if smoking.get("pipe") == "Yes":      smk += ", pipe smoker"
        parts.append(smk + ".")
    if disease:
        items = [f"{n} (onset age {int(a)})" if _valid(a) else f"{n} (onset age unknown)" for n, a in disease.items()]
        parts.append(f"Medical history: {', '.join(items)}.")
    if cancer:
        items = [f"{n} (age {int(a)})" if _valid(a) else n for n, a in cancer.items()]
        parts.append(f"Cancer history: {', '.join(items)}.")
    if fam:
        parts.append(f"Family history of lung cancer: {', '.join(str(k) for k in fam.keys())}.")
    return " ".join(parts) if parts else "No clinical data available."


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_task(task):
    """Trả về dict[key] = record."""
    return {str(r["keys"]): r for r in json.load(open(DATA_JSON / f"{task}_trainval.json"))}


def build_rows(keys, task_list, task_data, task_type, rng):
    rows = []
    for k in keys:
        ref = task_data[task_list[0]][k]
        clinical = clinical_to_text(ref["clinical_data"])
        q_lines, ans_obj = [], {}
        for t in task_list:
            r = task_data[t][k]
            q = rng.choice(r["questions"])
            ad = json.loads(r["answer_dict"]) if isinstance(r.get("answer_dict"), str) else r.get("answer_dict", {})
            allowed = " | ".join(ad.values())
            q_lines.append(f'- "{t}": {q} (allowed: {allowed})')
            ans_obj[t] = ad.get(str(r["labels"]), "Unknown")
        keys_str = ", ".join(f'"{t}"' for t in task_list)
        prompt = (
            "You are a helpful radiology assistant. Analyze the provided chest CT scan "
            "slices together with the patient's clinical record.\n\n"
            f"[PATIENT CLINICAL RECORD]\n{clinical}\n\n"
            "[QUESTIONS]\nFor each item below, choose exactly one of its allowed values:\n"
            + "\n".join(q_lines) +
            "\n\n[OUTPUT FORMAT]\nRespond ONLY with a single JSON object (no extra text) "
            f"using exactly these keys: {{{keys_str}}}."
        )
        rows.append({
            "keys": k, "pids": str(ref.get("pids", "unknown")),
            "prompt": prompt, "response": json.dumps(ans_obj, ensure_ascii=False),
            "task_type": task_type,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-n", type=int, default=302)
    ap.add_argument("--detail-n", type=int, default=198)
    ap.add_argument("--out-repo", default="UngLong/radiology-ready-v2")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Load all task data
    task_data = {t: load_task(t) for t in SCREENING + DETAIL}

    uploaded = set(CHECKPOINT.read_text().splitlines()) if CHECKPOINT.exists() else set()

    # Pool NPY-ready
    all8 = set.intersection(*[set(task_data[t].keys()) for t in SCREENING])
    screen_pool = all8 & uploaded
    all_detail = set.intersection(*[set(task_data[t].keys()) for t in DETAIL])
    detail_pool = all_detail & uploaded

    # Screening: hết rare positive + fill
    rare_pos = {k for k in screen_pool for t in RARE if task_data[t][k]["labels"] == 1}
    fill = [k for k in screen_pool if k not in rare_pos]
    rng.shuffle(fill)
    screen_keys = list(rare_pos) + fill[:max(0, args.screen_n - len(rare_pos))]

    # Detail: sample DETAIL_N từ Pool A NPY-ready (cap theo budget)
    detail_all = sorted(detail_pool)
    rng.shuffle(detail_all)
    detail_keys = detail_all[:args.detail_n]

    print(f"NPY-ready: screening pool {len(screen_pool)}, detail pool {len(detail_pool)}")
    print(f"Chọn: screening {len(screen_keys)}, detail {len(detail_keys)} → tổng {len(screen_keys)+len(detail_keys)}")

    # Balance screening
    print("\nBalance screening đã chọn:")
    for t in SCREENING:
        yes = sum(1 for k in screen_keys if task_data[t][k]["labels"] == 1)
        print(f"  {t:16s}: {yes:>3} dương ({100*yes/len(screen_keys):.0f}%)")

    # Build rows
    rows = build_rows(screen_keys, SCREENING, task_data, "screening", rng) \
         + build_rows(detail_keys, DETAIL, task_data, "detail", rng)
    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"\nTổng rows: {len(df)}")

    if args.dry_run or not args.push:
        print("\n--- SCREENING SAMPLE ---")
        s = df[df.task_type == "screening"].iloc[0]
        print(s["prompt"]); print("RESPONSE:", s["response"])
        print("\n--- DETAIL SAMPLE ---")
        de = df[df.task_type == "detail"].iloc[0]
        print(de["prompt"]); print("RESPONSE:", de["response"])
        print("\n[dry-run] CHƯA push. Thêm --push để đẩy lên HF.")
        return

    from datasets import Dataset
    Dataset.from_pandas(df).push_to_hub(repo_id=args.out_repo, private=True)
    print(f"\n✓ Pushed: {args.out_repo}")


if __name__ == "__main__":
    main()
