"""
QLoRA fine-tuning for the Radiology Agent.
Follows a two-stage approach: Detection first, then Recognition.
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from torch.utils.data import ConcatDataset

from training.dataset.openm3chest import OpenM3ChestDataset
from agents.radiology.task_config import TASK_CONFIG, label_to_text

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
    
    base_adapter = cfg.get("base_adapter")
    if base_adapter:
        from peft import PeftModel
        base_adapter_path = os.path.expandvars(base_adapter)
        print(f"Loading and unfreezing base adapter from: {base_adapter_path}")
        model = PeftModel.from_pretrained(model, base_adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type="CAUSAL_LM",
        ))

    datasets = []
    for task_file in cfg["data"]["task_files"]:
        task_name = task_file.replace("_trainval.json", "")
        if task_name not in TASK_CONFIG:
            continue
            
        question, _ = TASK_CONFIG[task_name]
        ds = OpenM3ChestDataset(
            json_path=str(Path(data_root) / task_file),
            task_name=task_name,
            question=question,
            label_fn=lambda r, tn=task_name: label_to_text(tn, r.get("labels", r.get("label"))),
            bbox_key="lung_bboxes",
            max_slices=cfg["data"]["max_slices"],
            image_size=cfg["data"]["image_size"],
        )
        
        # Filter out "No nodule" samples (label 0) for recognition tasks
        if task_name in ["nodule_size", "nodule_attenuation", "nodule_margin", "nodule_location"]:
            original_len = len(ds.records)
            ds.records = [
                r for r in ds.records 
                if str(r.get("labels", r.get("label", ""))) != "0"
            ]
            print(f"[{task_name}] Filtered 'No nodule' samples: {original_len} -> {len(ds.records)}")
            
        datasets.append(ds)

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
            report_to=t["report_to"],
        ),
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)