def clinical_to_text(clinical_data: dict) -> str:
    """
    Convert structured clinical JSON → plain English paragraph.
    Covers: demographics, smoking history, disease history, cancer history, family history.
    """
    demo    = clinical_data.get("demo", {})
    smoking = clinical_data.get("smoking", {})
    disease = clinical_data.get("disease_his", {})
    cancer  = clinical_data.get("cancer_his", {})
    fam     = clinical_data.get("fam_lc", {})

    parts = []

    # Demographics
    if demo:
        age    = int(demo["age"]) if demo.get("age") else None
        gender = demo.get("gender", "")
        race   = demo.get("race", "")
        height = demo.get("height")
        weight = demo.get("weight")
        tokens = []
        if age:
            tokens.append(f"{age}-year-old {gender}")
        if race:
            tokens.append(race)
        if height and weight:
            bmi = round(weight * 0.453592 / ((height * 0.0254) ** 2), 1)
            tokens.append(f"height {int(height)}in, weight {int(weight)}lbs (BMI {bmi})")
        if tokens:
            parts.append("Patient: " + ", ".join(tokens) + ".")

    # Smoking
    status = smoking.get("cigsmok", "")
    if status and status != "Never":
        pkyr     = smoking.get("pkyr")
        start    = smoking.get("smokeage")
        quit_age = smoking.get("age_quit")
        smk = f"Smoking: {status}"
        if pkyr:
            smk += f", {int(pkyr)} pack-years"
        if start:
            smk += f", started age {int(start)}"
        if status == "Former" and quit_age and quit_age > 0:
            smk += f", quit age {int(quit_age)}"
        parts.append(smk + ".")

    # Disease history
    if disease:
        items = ", ".join(
            f"{name} (onset age {int(age)})" for name, age in disease.items()
        )
        parts.append(f"Medical history: {items}.")

    # Cancer history
    if cancer:
        items = ", ".join(str(k) for k in cancer.keys())
        parts.append(f"Cancer history: {items}.")

    # Family lung cancer history
    if fam:
        relatives = ", ".join(str(k) for k in fam.keys())
        parts.append(f"Family history of lung cancer: {relatives}.")

    return " ".join(parts) if parts else "No clinical data available."
