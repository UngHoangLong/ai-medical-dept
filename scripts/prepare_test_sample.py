"""
Chuẩn bị sample test cho /api/v1/analyze.

Download DICOM của pid=122833 từ IDC, đóng gói thành .zip,
và lưu clinical_data + ground_truth vào tests/fixtures/sample_122833/.

Chạy 1 lần duy nhất:
    python scripts/prepare_test_sample.py
"""

import io
import json
import shutil
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Sample đã chọn — pid=122833 là sample duy nhất có ground truth cho cả 15 task
# ---------------------------------------------------------------------------

PID        = "122833"
SERIES_UID = "1.2.840.113654.2.55.307229896835702397101325832166669748691"

CLINICAL_DATA = {
    "demo": {
        "age": 63.0,
        "educat": "Post high school training, excluding college",
        "ethnic": "Neither Hispanic nor Latino",
        "gender": "Female",
        "height": 61.0,
        "race": "White",
        "weight": 100.0,
    },
    "smoking": {
        "age_quit": 51.0,
        "cigar": "No",
        "cigsmok": "Former",
        "pipe": "No",
        "pkyr": 90.0,
        "smokeage": 21.0,
        "smokeday": 60.0,
        "smokelive": "Yes",
        "smokework": "No",
        "smokeyr": 30.0,
    },
    "disease_his": {
        "Chronic bronchitis": 4.0,
        "COPD": 47.0,
        "Pneumonia": 36.0,
    },
    "cancer_his": {
        "Breast Cancer": 44.0,
    },
    "fam_lc": {
        "mother have lung cancer": 1.0,
    },
}

GROUND_TRUTH = {
    "radiology": {
        "screening": {
            "chest_abn_54": "No",           # atelectasis
            "chest_abn_55": "Yes",          # pleural effusion
            "chest_abn_56": "No",           # hilar/mediastinal mass
            "chest_abn_57": "No",           # chest wall abnormality
            "chest_abn_58": "No",           # consolidation
            "chest_abn_59": "Yes",          # emphysema
            "chest_abn_61": "Yes",          # fibrosis / honeycombing
            "nodule_presence": "Yes",
        },
        "detail": {
            "nodule_location":   "Left Upper Lobe",
            "nodule_attenuation": "Solid",
            "nodule_margin":     "Poorly defined",
            "nodule_size":       "6~8mm",
        },
    },
    "cardiology": {
        "CVD_diagnosis": "No",
        "CVD_mortality": "Low risk",
    },
    "oncology": {
        "lung_cancer_risk": "No cancer within follow-up",
    },
}

OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / f"sample_{PID}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    # 1. Lưu clinical_data.json
    (OUT_DIR / "clinical_data.json").write_text(
        json.dumps(CLINICAL_DATA, indent=2, ensure_ascii=False)
    )
    print("✓ clinical_data.json")

    # 2. Lưu ground_truth.json
    (OUT_DIR / "ground_truth.json").write_text(
        json.dumps(GROUND_TRUTH, indent=2, ensure_ascii=False)
    )
    print("✓ ground_truth.json")

    # 3. Lưu meta.json
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "pid": PID,
        "series_uid": SERIES_UID,
    }, indent=2))
    print("✓ meta.json")

    # 4. Download DICOM từ IDC nếu chưa có
    zip_path = OUT_DIR / "dicom.zip"
    if zip_path.exists():
        print(f"✓ dicom.zip đã có sẵn ({zip_path.stat().st_size / 1e6:.1f} MB) — bỏ qua download")
        return

    dcm_tmp = OUT_DIR / "_dicom_tmp"
    dcm_tmp.mkdir(exist_ok=True)

    print(f"Downloading DICOM series {SERIES_UID[:40]}... (có thể mất vài phút)")
    from idc_index import IDCClient
    client = IDCClient()
    client.download_dicom_series(
        seriesInstanceUID=SERIES_UID,
        downloadDir=str(dcm_tmp),
    )

    # Tìm tất cả .dcm files
    dcm_files = list(dcm_tmp.rglob("*.dcm"))
    if not dcm_files:
        # IDC đôi khi không có extension
        dcm_files = [p for p in dcm_tmp.rglob("*") if p.is_file()]

    print(f"  Downloaded {len(dcm_files)} files")

    # 5. Đóng gói thành .zip (flatten — không giữ cấu trúc thư mục sâu)
    print("Packing into dicom.zip...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(dcm_files):
            zf.write(f, arcname=f.name)

    shutil.rmtree(dcm_tmp)
    print(f"✓ dicom.zip  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nSample sẵn sàng tại: {OUT_DIR}")
    print("\nGọi API:")
    print(f"  pid        = {PID}")
    print(f"  series_uid = {SERIES_UID}")
    print(f"  dicom_zip  = tests/fixtures/sample_{PID}/dicom.zip")


if __name__ == "__main__":
    main()
