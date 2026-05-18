"""
Manages a single base MedGemma model with hot-swappable LoRA adapters.
Only one adapter is active at a time to save VRAM.
"""

import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


class LoraManager:
    def __init__(self, base_model, processor, adapter_paths: dict[str, Path]):
        self.base_model    = base_model
        self.processor     = processor
        self.adapter_paths = adapter_paths
        self._active: str | None = None
        self._peft_model: PeftModel | None = None

    @classmethod
    def from_config(cls, config_path: str) -> "LoraManager":
        cfg = yaml.safe_load(Path(config_path).read_text())
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_id = cfg["base_model_id"]
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
        adapter_paths = {
            agent: checkpoint_dir / rel_path
            for agent, rel_path in cfg["lora_adapters"].items()
        }
        return cls(model, processor, adapter_paths)

    def load(self, agent: str) -> PeftModel:
        """Load LoRA adapter for agent. Reuses cached model if already active."""
        if self._active == agent:
            return self._peft_model

        adapter_path = self.adapter_paths[agent]
        self._peft_model = PeftModel.from_pretrained(self.base_model, str(adapter_path))
        self._active = agent
        return self._peft_model
