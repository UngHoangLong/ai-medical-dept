"""
clinical_to_text — chuyển structured clinical JSON (NLST) → đoạn văn tiếng Anh.

Dùng chung cho cả fine-tune lẫn inference. Bỏ qua các missing sentinel của NLST:
  - số   : -1.0          (height/weight/age_quit/onset age không có)
  - chuỗi : "None" / ""   (race/ethnic/educat/smoking flags không có)
"""

import json


def _valid(v) -> bool:
    """False nếu là missing sentinel (None, "", "None", hoặc số <= 0)."""
    if v is None:
        return False
    if isinstance(v, str):
        return v not in ("", "None")
    if isinstance(v, (int, float)):
        return v > 0
    return True


def clinical_to_text(clinical_data) -> str:
    """
    Convert structured clinical JSON → plain English paragraph.
    Covers: demographics (+BMI), smoking, disease history, cancer history, family history.
    Accepts dict hoặc JSON string.
    """
    if isinstance(clinical_data, str):
        clinical_data = json.loads(clinical_data)

    demo    = clinical_data.get("demo", {}) or {}
    smoking = clinical_data.get("smoking", {}) or {}
    disease = clinical_data.get("disease_his", {}) or {}
    cancer  = clinical_data.get("cancer_his", {}) or {}
    fam     = clinical_data.get("fam_lc", {}) or {}

    parts = []

    # --- Demographics ---
    if demo:
        tokens = []
        age    = demo.get("age")
        gender = demo.get("gender", "") if _valid(demo.get("gender")) else ""
        race   = demo.get("race")
        ethnic = demo.get("ethnic")
        educat = demo.get("educat")
        height = demo.get("height")
        weight = demo.get("weight")

        if _valid(age):
            tokens.append(f"{int(age)}-year-old {gender}".strip())
        if _valid(race):
            tokens.append(race)
        if _valid(ethnic) and ethnic != "Neither Hispanic nor Latino":
            tokens.append(ethnic)
        if _valid(height) and _valid(weight):
            bmi = round(weight * 0.453592 / ((height * 0.0254) ** 2), 1)
            tokens.append(f"height {int(height)}in, weight {int(weight)}lbs (BMI {bmi})")
        if _valid(educat):
            tokens.append(f"education: {educat}")
        if tokens:
            parts.append("Patient: " + ", ".join(tokens) + ".")

    # --- Smoking ---
    status = smoking.get("cigsmok", "")
    if _valid(status) and status != "Never":
        smk      = f"Smoking: {status}"
        pkyr     = smoking.get("pkyr")
        start    = smoking.get("smokeage")
        quit_age = smoking.get("age_quit")
        smokeday = smoking.get("smokeday")
        smokeyr  = smoking.get("smokeyr")

        if _valid(pkyr):
            smk += f", {int(pkyr)} pack-years"
        if _valid(start):
            smk += f", started age {int(start)}"
        if _valid(smokeday):
            smk += f", {int(smokeday)} cigarettes/day"
        if _valid(smokeyr):
            smk += f", {int(smokeyr)} years"
        if status == "Former" and _valid(quit_age):
            smk += f", quit age {int(quit_age)}"
        if smoking.get("smokelive") == "Yes":
            smk += ", lives with smoker"
        if smoking.get("smokework") == "Yes":
            smk += ", works with smoker"
        if smoking.get("cigar") == "Yes":
            smk += ", cigar smoker"
        if smoking.get("pipe") == "Yes":
            smk += ", pipe smoker"
        parts.append(smk + ".")

    # --- Disease history ---
    if disease:
        items = [
            f"{name} (onset age {int(age)})" if _valid(age) else f"{name} (onset age unknown)"
            for name, age in disease.items()
        ]
        parts.append(f"Medical history: {', '.join(items)}.")

    # --- Cancer history ---
    if cancer:
        items = [
            f"{name} (age {int(age)})" if _valid(age) else name
            for name, age in cancer.items()
        ]
        parts.append(f"Cancer history: {', '.join(items)}.")

    # --- Family lung cancer history ---
    if fam:
        relatives = ", ".join(str(k) for k in fam.keys())
        parts.append(f"Family history of lung cancer: {relatives}.")

    return " ".join(parts) if parts else "No clinical data available."
