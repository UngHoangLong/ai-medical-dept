"""
build_radiology_test.py — Build test set cho Radiology agent (~154 rows).

Lấy từ test split data_json (hoàn toàn tách biệt với training scans).
Format y hệt radiology-ready-v2 (prompt + response JSON, screening + detail).

Flow:
  1. Inner join 8 screening tasks test → tối đa 74 scans
  2. Sample detail scans (4 task) khớp phân phối training
  3. Check NPY availability → export danh sách cần upload
  4. Build prompt/response → push lên HF (hoặc --dry-run)

Usage:
    python scripts/build_radiology_test.py --dry-run
    python scripts/build_radiology_test.py --push --out-repo UngLong/radiology-test-v2
"""

import argparse, json, random
from pathlib import Path
from collections import Counter

import pandas as pd

DATA_JSON  = Path("/Users/macbook/Documents/Demo_AI/data/data_json")
CHECKPOINT = Path("/Users/macbook/Documents/Demo_AI/ai-medical-dept/scripts/npy_upload_checkpoint.txt")

SCREENING = ["chest_abn_54","chest_abn_55","chest_abn_56","chest_abn_57",
             "chest_abn_58","chest_abn_59","chest_abn_61","nodule_presence"]
DETAIL    = ["nodule_location","nodule_attenuation","nodule_margin","nodule_size"]
RARE      = ["chest_abn_54","chest_abn_55","chest_abn_56","chest_abn_57","chest_abn_58"]

# Target distribution từ training (radiology-ready-v2, 302 screening rows)
TRAIN_DETAIL_DIST = {
    "nodule_location":    {"Right Upper Lobe":46,"Left Upper Lobe":45,"Left Lower Lobe":41,"Right Lower Lobe":39,"Right Middle Lobe":27},
    "nodule_attenuation": {"Solid":144,"Ground Glass":33,"Others":21},
    "nodule_margin":      {"Smooth":118,"Poorly defined":40,"Spiculated (Stellate)":28,"Unable to determine":12},
    "nodule_size":        {"4~6mm":66,"<=4mm":42,"8~15mm":40,"6~8mm":33,"15~30mm":14,">30mm":3},
}


def _valid(v):
    if v is None: return False
    if isinstance(v, str): return v not in ("","None")
    if isinstance(v, (int,float)): return v > 0
    return True


def clinical_to_text(cd) -> str:
    if isinstance(cd, str): cd = json.loads(cd)
    demo=cd.get("demo",{}); smk=cd.get("smoking",{}); dis=cd.get("disease_his",{})
    can=cd.get("cancer_his",{}); fam=cd.get("fam_lc",{}); parts=[]
    if demo:
        tok=[]
        if _valid(demo.get("age")): tok.append(f"{int(demo['age'])}-year-old {demo.get('gender','') if _valid(demo.get('gender')) else ''}".strip())
        if _valid(demo.get("race")): tok.append(demo["race"])
        if _valid(demo.get("ethnic")) and demo["ethnic"]!="Neither Hispanic nor Latino": tok.append(demo["ethnic"])
        if _valid(demo.get("height")) and _valid(demo.get("weight")):
            bmi=round(demo["weight"]*0.453592/((demo["height"]*0.0254)**2),1)
            tok.append(f"height {int(demo['height'])}in, weight {int(demo['weight'])}lbs (BMI {bmi})")
        if _valid(demo.get("educat")): tok.append(f"education: {demo['educat']}")
        if tok: parts.append("Patient: "+", ".join(tok)+".")
    status=smk.get("cigsmok","")
    if _valid(status) and status!="Never":
        s=f"Smoking: {status}"
        if _valid(smk.get("pkyr")):     s+=f", {int(smk['pkyr'])} pack-years"
        if _valid(smk.get("smokeage")): s+=f", started age {int(smk['smokeage'])}"
        if _valid(smk.get("smokeday")): s+=f", {int(smk['smokeday'])} cigarettes/day"
        if _valid(smk.get("smokeyr")):  s+=f", {int(smk['smokeyr'])} years"
        if status=="Former" and _valid(smk.get("age_quit")): s+=f", quit age {int(smk['age_quit'])}"
        if smk.get("smokelive")=="Yes": s+=", lives with smoker"
        if smk.get("smokework")=="Yes": s+=", works with smoker"
        if smk.get("cigar")=="Yes": s+=", cigar smoker"
        if smk.get("pipe")=="Yes": s+=", pipe smoker"
        parts.append(s+".")
    if dis:
        items=[f"{n} (onset age {int(a)})" if _valid(a) else f"{n} (onset age unknown)" for n,a in dis.items()]
        parts.append(f"Medical history: {', '.join(items)}.")
    if can:
        items=[f"{n} (age {int(a)})" if _valid(a) else n for n,a in can.items()]
        parts.append(f"Cancer history: {', '.join(items)}.")
    if fam:
        parts.append(f"Family history of lung cancer: {', '.join(str(k) for k in fam.keys())}.")
    return " ".join(parts) if parts else "No clinical data available."


def build_rows(keys, task_list, task_data, task_type, rng):
    rows=[]
    for k in keys:
        ref=next(task_data[t][k] for t in task_list if k in task_data[t])
        clinical=clinical_to_text(ref["clinical_data"])
        q_lines, ans_obj = [], {}
        for t in task_list:
            r=task_data[t][k]
            q=rng.choice(r["questions"])
            ad=json.loads(r["answer_dict"]) if isinstance(r.get("answer_dict"),str) else r.get("answer_dict",{})
            allowed=" | ".join(ad.values())
            q_lines.append(f'- "{t}": {q} (allowed: {allowed})')
            ans_obj[t]=ad.get(str(r["labels"]),"Unknown")
        keys_str=", ".join(f'"{t}"' for t in task_list)
        prompt=(
            "You are a helpful radiology assistant. Analyze the provided chest CT scan "
            "slices together with the patient's clinical record.\n\n"
            f"[PATIENT CLINICAL RECORD]\n{clinical}\n\n"
            "[QUESTIONS]\nFor each item below, choose exactly one of its allowed values:\n"
            +"\n".join(q_lines)+
            "\n\n[OUTPUT FORMAT]\nRespond ONLY with a single JSON object (no extra text) "
            f"using exactly these keys: {{{keys_str}}}."
        )
        rows.append({"keys":k,"pids":str(ref.get("pids","unknown")),
                     "prompt":prompt,"response":json.dumps(ans_obj,ensure_ascii=False),
                     "task_type":task_type})
    return rows


def sample_detail_matching_train(detail_pool, task_data, target_n, rng):
    """Sample detail scans cố gắng khớp phân phối train."""
    # Ưu tiên lấy đủ mỗi class của từng task (stratified)
    pool = list(detail_pool); rng.shuffle(pool)
    selected = set()
    # Lấy đại diện cho mỗi class hiếm trước
    for t in ["nodule_size","nodule_attenuation","nodule_margin"]:
        class_keys = {}
        for k in pool:
            if k in task_data[t]:
                lbl = str(task_data[t][k]["labels"])
                ad = json.loads(task_data[t][k]["answer_dict"]) if isinstance(task_data[t][k].get("answer_dict"),str) else {}
                cls = ad.get(lbl, lbl)
                class_keys.setdefault(cls,[]).append(k)
        for cls, ks in class_keys.items():
            take = max(1, int(target_n * TRAIN_DETAIL_DIST.get(t,{}).get(cls,1) / 198))
            selected.update(rng.sample(ks, min(take, len(ks))))
    # Fill còn lại
    fill = [k for k in pool if k not in selected]
    rng.shuffle(fill)
    selected.update(fill[:max(0, target_n - len(selected))])
    return list(selected)[:target_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-n", type=int, default=80)
    ap.add_argument("--out-repo", default="UngLong/radiology-test-v2")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--export-npy-list", default="scripts/test_npy_needed.txt")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    task_data = {t: {str(r["keys"]): r for r in json.load(open(DATA_JSON/f"{t}_test.json"))}
                 for t in SCREENING+DETAIL}

    uploaded = set(CHECKPOINT.read_text().splitlines()) if CHECKPOINT.exists() else set()

    all8 = set.intersection(*[set(task_data[t].keys()) for t in SCREENING])
    all_detail = set.intersection(*[set(task_data[t].keys()) for t in DETAIL])

    screen_keys = list(all8)
    detail_keys = sample_detail_matching_train(all_detail, task_data, args.detail_n, rng)

    total = len(screen_keys) + len(detail_keys)
    print(f"Screening: {len(screen_keys)} | Detail: {len(detail_keys)} → Tổng: {total} rows")

    # NPY check
    all_keys = set(screen_keys) | set(detail_keys)
    have_npy = all_keys & uploaded
    need_npy = all_keys - uploaded
    print(f"Đã có NPY: {len(have_npy)} | CẦN upload: {len(need_npy)} scans (~{len(need_npy)*80/1000:.0f}GB)")

    # Export danh sách cần upload
    npy_lines = []
    for k in sorted(need_npy):
        rec = next((task_data[t][k] for t in SCREENING+DETAIL if k in task_data[t]), None)
        if rec:
            pid = str(rec.get("pids","unknown"))
            npy_lines.append(f"{pid}/{k}")
    Path(args.export_npy_list).write_text("\n".join(npy_lines))
    print(f"→ Exported NPY list: {args.export_npy_list} ({len(npy_lines)} scans)")

    # Balance check
    print(f"\n=== Balance screening ({len(screen_keys)} scans) ===")
    for t in SCREENING:
        yes = sum(1 for k in screen_keys if task_data[t].get(k,{}).get("labels")==1)
        print(f"  {t:16s}: {yes:>3}/{len(screen_keys)} ({100*yes/len(screen_keys):.0f}% Yes)")

    print(f"\n=== Balance detail ({len(detail_keys)} scans) ===")
    for t in DETAIL:
        cnt = Counter()
        for k in detail_keys:
            if k in task_data[t]:
                r=task_data[t][k]; ad=json.loads(r["answer_dict"]) if isinstance(r.get("answer_dict"),str) else {}
                cnt[ad.get(str(r["labels"]),str(r["labels"]))] += 1
        print(f"  {t}: {dict(cnt.most_common())}")

    if args.dry_run or not args.push:
        rows = build_rows(screen_keys[:1], SCREENING, task_data, "screening", rng)
        print(f"\n--- SCREENING SAMPLE ---\n{rows[0]['prompt']}\nRESPONSE: {rows[0]['response']}")
        print("\n[dry-run] Chưa push. Thêm --push để đẩy lên HF.")
        return

    rows = build_rows(screen_keys, SCREENING, task_data, "screening", rng) + \
           build_rows(detail_keys, DETAIL, task_data, "detail", rng)
    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    from datasets import Dataset, DatasetDict
    DatasetDict({"test": Dataset.from_pandas(df)}).push_to_hub(repo_id=args.out_repo, private=True)
    print(f"\n✓ Pushed: {args.out_repo} (split=test)")


if __name__ == "__main__":
    main()
