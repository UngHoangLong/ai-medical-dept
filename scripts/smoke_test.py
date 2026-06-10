"""
Smoke test — chạy trên VastAI RTX 3090 để đo tài nguyên trước khi train full.

Kiểm tra 2 thứ:
  1. Fine-tune: 5 bước training với task chest_abn_54
  2. Inference: 1 sample với model gốc (chưa có LoRA)

In ra: VRAM peak, thời gian/step, thời gian inference.

Usage:
    python scripts/smoke_test.py \
        --data-json /data/labels/data_json \
        --cache-dir /data/npy_cache \
        --model-cfg configs/model.yaml \
        --train-steps 5
"""

import argparse
import os
import time
from pathlib import Path

import torch
import yaml


def get_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


SYSTEM_PROMPT = (
    "You are an expert radiologist assistant. "
    "Analyze the provided CT scan slices along with the patient's clinical information. "
    "Answer the question with Yes or No, then briefly explain the key CT findings "
    "that support your conclusion."
)


def build_messages(slices, prompt: str, answer: str | None = None) -> list[dict]:
    """Build MedGemma chat format: system prompt + PIL images + clinical text + question."""
    content = [{"type": "image", "image": img} for img in slices]
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": content},
    ]
    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        })
    return messages


def label_fn(record: dict) -> str:
    return {0: "No.", 1: "Yes."}.get(int(record["labels"]), str(record["labels"]))


# ---------------------------------------------------------------------------
# Fine-tune smoke test
# ---------------------------------------------------------------------------

def test_finetune(model, processor, dataset, steps: int):
    print(f"\n{'='*50}")
    print(f"[FINETUNE] Testing {steps} training steps...")
    print(f"{'='*50}")

    from unsloth import FastVisionModel

    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    reset_vram()

    step_times = []
    for step in range(steps):
        sample  = dataset[step % len(dataset)]
        slices  = sample["slices"]
        prompt  = sample["prompt"]
        answer  = sample["answer"]

        messages = build_messages(slices, prompt, answer)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        labels = inputs["input_ids"].clone()
        # Mask user turn — only compute loss on assistant answer
        labels[labels == processor.tokenizer.pad_token_id] = -100

        t0 = time.time()
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_times.append(time.time() - t0)

        print(f"  Step {step+1}/{steps}  loss={loss.item():.4f}  "
              f"time={step_times[-1]:.1f}s  VRAM={get_vram_gb():.2f}GB")

    print(f"\n[FINETUNE] Avg time/step : {sum(step_times)/len(step_times):.1f}s")
    print(f"[FINETUNE] Peak VRAM     : {get_vram_gb():.2f} GB")


# ---------------------------------------------------------------------------
# Inference smoke test
# ---------------------------------------------------------------------------

def test_inference(model, processor, dataset):
    print(f"\n{'='*50}")
    print("[INFERENCE] Testing 1 sample...")
    print(f"{'='*50}")

    model.eval()
    reset_vram()

    sample   = dataset[0]
    slices   = sample["slices"]
    prompt   = sample["prompt"]
    expected = sample["answer"]

    messages = build_messages(slices, prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[-1]
    answer = processor.decode(output[0][input_len:], skip_special_tokens=True)

    print(f"  Prompt    : {prompt[:120]}...")
    print(f"  Expected  : {expected}")
    print(f"  Got       : {answer}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"  Peak VRAM : {get_vram_gb():.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",      required=True, help="Local dir để cache NPY files")
    parser.add_argument("--model-cfg",      default="configs/model.yaml")
    parser.add_argument("--inference-only", action="store_true", help="Chỉ test inference, bỏ qua fine-tune")
    parser.add_argument("--train-steps",    type=int, default=5)
    args = parser.parse_args()

    cfg      = yaml.safe_load(Path(args.model_cfg).read_text())
    model_id = cfg["base_model_id"]

    # --- Load model ---
    print(f"\n[SETUP] Loading model: {model_id}")
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        dtype=None,
    )
    print(f"[SETUP] Model loaded. VRAM: {get_vram_gb():.2f} GB")

    # --- Load dataset từ HuggingFace Hub ---
    print("\n[SETUP] Loading dataset from HuggingFace Hub...")
    from datasets import load_dataset

    hf_ds = load_dataset("UngLong/openm3chest-labels", "chest_abn_54")
    print(f"[SETUP] chest_abn_54 — train: {len(hf_ds['train'])} rows, test: {len(hf_ds['test'])} rows")

    # --- Dataset ---
    from agents.data_prep.prepare_data import DataPrepAgent
    from training.dataset.openm3chest import OpenM3ChestDataset

    data_prep = DataPrepAgent(
        cache_dir    = args.cache_dir,
        max_slices   = cfg["data_prep"]["max_slices"],
        max_cache_gb = cfg["data_prep"]["max_cache_gb"],
    )
    # Dùng split "test" để test inference, "train" để test fine-tune
    dataset = OpenM3ChestDataset(
        records   = hf_ds["test"],
        label_fn  = label_fn,
        data_prep = data_prep,
    )
    print(f"[SETUP] Test split: {len(dataset)} records")

    # --- Run tests ---
    test_inference(model, processor, dataset)
    if not args.inference_only:
        test_finetune(model, processor, dataset, steps=args.train_steps)

    print("\n[DONE] Smoke test complete.")


if __name__ == "__main__":
    main()
