from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass
class AgentOutput:
    text_answer: str
    hidden_state: torch.Tensor  # [D_model] — fed to Synthesis Agent


class BaseSpecialistAgent(ABC):
    """
    Base class for Radiology, Cardiology, Oncology agents.
    Each agent = MedGemma base (4-bit) + one LoRA adapter.
    """

    def __init__(self, base_model, processor, lora_adapter_path: Path):
        self.model = base_model
        self.processor = processor
        self.lora_adapter_path = lora_adapter_path

    @abstractmethod
    def build_prompt(self, slices: list[Image.Image], clinical_text: str) -> list[dict]:
        """Build the messages list for processor.apply_chat_template()."""

    @abstractmethod
    def parse_label(self, record: dict) -> str:
        """Map raw JSON label → text answer for training."""

    @torch.inference_mode()
    def run(self, slices: list[Image.Image], clinical_text: str) -> AgentOutput:
        messages = self.build_prompt(slices, clinical_text)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        generated_ids = output.sequences[0][input_len:]
        text_answer = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Last hidden state of the final generated token — shape [D_model]
        hidden_state = output.hidden_states[-1][-1][0, -1, :].float().cpu()

        return AgentOutput(text_answer=text_answer, hidden_state=hidden_state)
