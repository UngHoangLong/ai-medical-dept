def clinical_to_text(clinical_data: dict, question: str) -> str:
    demo    = clinical_data.get("demo", {})
    smoking = clinical_data.get("smoking", {})
    disease = clinical_data.get("disease_his", {})
    fam     = clinical_data.get("fam_lc", {})

    parts = []
    if demo:
        parts.append(
            f"Patient: {demo.get('age', '')} y/o {demo.get('gender', '')}, "
            f"Race: {demo.get('race', '')}, "
            f"{demo.get('height', '')}in / {demo.get('weight', '')}lbs."
        )
    if smoking.get("cigsmok"):
        parts.append(
            f"Smoking: {smoking['cigsmok']}, "
            f"{smoking.get('pkyr', '')} pack-years, "
            f"started age {smoking.get('smokeage', '')}."
        )
    if disease:
        items = ", ".join(f"{k} (age {int(v)})" for k, v in disease.items())
        parts.append(f"Disease history: {items}.")
    if fam:
        parts.append(f"Family history: {', '.join(fam.keys())}.")

    return " ".join(parts) + f"\nQuestion: {question}"
