"""
Question and label-to-text mapping for all 7 Radiology tasks.
"""

TASK_CONFIG: dict[str, tuple[str, dict]] = {
    "chest_abn_54": (
        "Is atelectasis present in this chest CT?",
        {0: "No atelectasis detected.", 1: "Atelectasis detected."},
    ),
    "covid19": (
        "Does this CT show signs of COVID-19?",
        {0: "COVID-19 negative.", 1: "COVID-19 findings present."},
    ),
    "nodule_presence": (
        "Is a lung nodule present?",
        {0: "No lung nodule detected.", 1: "Lung nodule detected."},
    ),
    "nodule_size": (
        "What is the size category of the lung nodule?",
        {0: "No nodule.", 1: "Small nodule (<6mm).", 2: "Medium nodule (6–20mm).", 3: "Large nodule (>20mm)."},
    ),
    "nodule_attenuation": (
        "What is the attenuation type of the nodule?",
        {
            0: "No nodule.",
            1: "Solid nodule.",
            2: "Part-solid (subsolid) nodule.",
            3: "Ground-glass opacity (GGO).",
        },
    ),
    "nodule_margin": (
        "What is the margin characteristic of the nodule?",
        {
            0: "No nodule.",
            1: "Well-defined, smooth margin.",
            2: "Irregular or spiculated margin.",
        },
    ),
    "nodule_location": (
        "Where is the nodule located?",
        {},  # location labels are multi-class strings — handled separately
    ),
}


def label_to_text(task_name: str, label) -> str:
    """Convert raw label to answer text for instruction tuning."""
    _, label_map = TASK_CONFIG[task_name]
    if isinstance(label, int) and label_map:
        return label_map.get(label, str(label))
    return str(label)
