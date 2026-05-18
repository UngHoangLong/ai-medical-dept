"""
QLoRA fine-tuning for the Cardiology Agent.
Trains on CVD_diagnosis + CVD_mortality from OpenM3Chest.

Usage:
    python training/finetune_cardiology.py --config configs/training/cardiology.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from training.dataset.openm3chest import OpenM3ChestDataset

TASK_FILES = {
    "CVD_diagnosis":  ("Is cardiovascular disease (CVD) present?",  {0: "No CVD detected.", 1: "CVD detected."},  "lung_bboxes"),
    "CVD_mortality":  ("Is there risk of CVD-related mortality?",    {0: "Low CVD mortality risk.", 1: "High CVD mortality risk."}, "heart_bboxes"),
}


def make_label_fn(label_map: dict):
    def label_fn(record):
        return label_map.get(int(record["labels"]), str(record["labels"]))
    return label_fn


def main(cfg: dict):
    model_id   = os.getenv("BASE_MODEL_ID", "google/medgemma-1.5-4b-it")
    output_dir = cfg["output_dir"]
    data_root  = os.getenv("DATA_ROOT", cfg["data"]["data_root"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    model = prepare_model_for_kbit_training(model)
    lora_cfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    datasets = []
    for task_name, (question, label_map, bbox_key) in TASK_FILES.items():
        json_path = Path(data_root) / f"{task_name}_trainval.json"
        datasets.append(OpenM3ChestDataset(
            json_path=str(json_path),
            task_name=task_name,
            question=question,
            label_fn=make_label_fn(label_map),
            bbox_key=bbox_key,
            max_slices=cfg["data"]["max_slices"],
            image_size=cfg["data"]["image_size"],
        ))

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(datasets)

    t = cfg["training"]
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=t["num_epochs"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            learning_rate=t["learning_rate"],
            warmup_ratio=t["warmup_ratio"],
            lr_scheduler_type=t["lr_scheduler_type"],
            bf16=t["bf16"],
            save_steps=t["save_steps"],
            logging_steps=t["logging_steps"],
            report_to=t["report_to"],
        ),
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
