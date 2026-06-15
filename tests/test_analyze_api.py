"""
Integration test — gọi POST /api/v1/analyze với sample thật (pid=122833).

Yêu cầu trước khi chạy:
  1. Backend đang chạy:  uvicorn serving.backend.server:app --port 8000
  2. Modal container đã warm (modal run serving/modal/app.py::warmup)
  3. .env có MODAL_PIPELINE_URL, AWS_*, ...

Chạy:
    python tests/integration/test_analyze_api.py
    python tests/integration/test_analyze_api.py --url http://localhost:8001
    python tests/integration/test_analyze_api.py --url http://localhost:8001 --json
"""

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import httpx

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sample_122833"
BACKEND_URL_DEFAULT = "http://localhost:8000"

W = 72  # line width cho separator

TASK_LABEL = {
    "chest_abn_54": "atelectasis",
    "chest_abn_55": "pleural effusion",
    "chest_abn_56": "hilar/mediastinal mass",
    "chest_abn_57": "chest wall abnormality",
    "chest_abn_58": "consolidation",
    "chest_abn_59": "emphysema",
    "chest_abn_61": "fibrosis/honeycombing",
    "nodule_presence": "nodule present",
    "nodule_location": "location",
    "nodule_attenuation": "attenuation",
    "nodule_margin": "margin",
    "nodule_size": "size",
    "CVD_diagnosis": "CVD diagnosis",
    "CVD_mortality": "CVD mortality",
    "lung_cancer_risk": "lung cancer risk",
}


def sep(char="─"):
    print(char * W)


def header(title: str):
    print(f"\n{'━' * W}")
    print(f"  {title}")
    print("━" * W)


def wrap_text(text: str, indent: str = "    ", width: int = W) -> str:
    if not text:
        return f"{indent}(empty)"
    out = []
    for para in text.strip().split("\n"):
        line = para.strip()
        if not line:
            out.append("")
        else:
            out.append(textwrap.fill(line, width=width,
                                     initial_indent=indent,
                                     subsequent_indent=indent))
    return "\n".join(out)


def load_fixture():
    meta         = json.loads((FIXTURE_DIR / "meta.json").read_text())
    clinical     = (FIXTURE_DIR / "clinical_data.json").read_text()
    ground_truth = json.loads((FIXTURE_DIR / "ground_truth.json").read_text())
    dicom_bytes  = (FIXTURE_DIR / "dicom.zip").read_bytes()
    return meta, clinical, ground_truth, dicom_bytes


def print_results(result: dict):
    """In kết quả pipeline theo từng agent, có nhóm rõ ràng."""

    # ── Radiology ──────────────────────────────────────────────────────
    header("RADIOLOGY — Screening")
    screen = result.get("radiology", {}).get("screening", {}).get("answer", {})
    for k, v in screen.items():
        label = TASK_LABEL.get(k, k)
        print(f"  {k:<16}  {label:<26}  {v}")

    detail = result.get("radiology", {}).get("detail")
    if detail:
        header("RADIOLOGY — Nodule Detail")
        for k, v in detail.get("answer", {}).items():
            label = TASK_LABEL.get(k, k)
            print(f"  {k:<20}  {label:<16}  {v}")
    else:
        header("RADIOLOGY — Nodule Detail")
        print("  (skipped — no nodule detected)")

    # ── Cardiology ─────────────────────────────────────────────────────
    header("CARDIOLOGY")
    cardio = result.get("cardiology", {}).get("answer", {})
    for k, v in cardio.items():
        label = TASK_LABEL.get(k, k)
        print(f"  {label:<22}  {v}")

    # ── Oncology ───────────────────────────────────────────────────────
    header("ONCOLOGY")
    onco = result.get("oncology", {}).get("answer", {})
    for k, v in onco.items():
        label = TASK_LABEL.get(k, k)
        print(f"  {label:<22}  {v}")

    # ── Finding & Impression ───────────────────────────────────────────
    header("FINDING & IMPRESSION")
    fi = result.get("finding_impression", {})
    print("  Findings:")
    print(wrap_text(fi.get("findings") or "(empty)"))
    print("\n  Impression:")
    print(wrap_text(fi.get("impression") or "(empty)"))

    # ── Verification ───────────────────────────────────────────────────
    header("VERIFICATION — Agent 5 Part 1")
    analysis = result.get("verification", {}).get("analysis")
    if analysis:
        print(wrap_text(analysis))
    else:
        print("  (no analysis)")


def print_ground_truth(result: dict, gt: dict) -> list[str]:
    """In ground truth comparison theo nhóm, trả về list mismatch."""
    mismatches: list[str] = []

    def check_group(title: str, items: dict[str, str], answer: dict) -> int:
        correct = 0
        rows = []
        for k, expected in items.items():
            got = answer.get(k)
            ok = got == expected
            if ok:
                correct += 1
            else:
                mismatches.append(k)
            tick = "✓" if ok else "✗"
            label = TASK_LABEL.get(k, k)
            if ok:
                rows.append(f"  {tick}  {label:<28}  {got}")
            else:
                rows.append(f"  {tick}  {label:<28}  got={got!r:<30} expected={expected!r}")
        n = len(items)
        print(f"\n  {title}  [{correct}/{n}]")
        sep()
        for r in rows:
            print(r)
        return correct

    total_correct = 0

    total_correct += check_group(
        "Radiology — Screening",
        gt["radiology"]["screening"],
        result.get("radiology", {}).get("screening", {}).get("answer", {}),
    )

    detail_result = result.get("radiology", {}).get("detail")
    if detail_result:
        total_correct += check_group(
            "Radiology — Nodule Detail",
            gt["radiology"]["detail"],
            detail_result.get("answer", {}),
        )
    else:
        n = len(gt["radiology"]["detail"])
        print(f"\n  Radiology — Nodule Detail  [0/{n}]")
        sep()
        print("  (skipped — no nodule detected)")
        for k in gt["radiology"]["detail"]:
            mismatches.append(k)

    total_correct += check_group(
        "Cardiology",
        gt["cardiology"],
        result.get("cardiology", {}).get("answer", {}),
    )

    total_correct += check_group(
        "Oncology",
        gt["oncology"],
        result.get("oncology", {}).get("answer", {}),
    )

    return mismatches, total_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BACKEND_URL_DEFAULT)
    parser.add_argument("--json", action="store_true", help="Dump raw JSON response")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    endpoint = f"{base_url}/api/v1/analyze"

    print(f"fixture : {FIXTURE_DIR}")
    meta, clinical, ground_truth, dicom_bytes = load_fixture()
    print(f"pid     : {meta['pid']}")
    print(f"series  : {meta['series_uid'][:55]}...")
    print(f"zip     : {len(dicom_bytes)/1e6:.1f} MB")
    print(f"url     : {endpoint}")

    t0 = time.perf_counter()
    print("\nCalling API...")

    with httpx.Client(timeout=900.0) as client:
        resp = client.post(
            endpoint,
            data={
                "pid":           meta["pid"],
                "series_uid":    meta["series_uid"],
                "clinical_data": clinical,
            },
            files={
                "dicom_zip": ("dicom.zip", dicom_bytes, "application/zip"),
            },
        )

    elapsed = time.perf_counter() - t0
    print(f"HTTP {resp.status_code}  ({elapsed:.1f}s)")

    if resp.status_code != 200:
        print("\nERROR:", resp.text[:500])
        sys.exit(1)

    result = resp.json()

    if args.json:
        print("\n=== Raw JSON ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print_results(result)

    header("GROUND TRUTH COMPARISON")
    mismatches, total_correct = print_ground_truth(result, ground_truth)

    total_tasks = 8 + 4 + 2 + 1
    print(f"\n{'━' * W}")
    print(f"  SCORE: {total_correct}/{total_tasks}")
    if mismatches:
        print(f"  FAIL : {mismatches}")
    else:
        print("  All tasks match ground truth!")
    print("━" * W)


if __name__ == "__main__":
    main()
