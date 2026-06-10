"""
Check MedGemma processor config — không cần PyTorch.

Usage:
    python scripts/check_processor.py
    python scripts/check_processor.py --token YOUR_HF_TOKEN
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    MODEL_ID = "unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit"

    print(f"Loading processor from: {MODEL_ID}\n")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=args.token)
    img_proc = processor.image_processor

    config = img_proc.to_dict()

    print("=" * 50)
    print("IMAGE PROCESSOR CONFIG")
    print("=" * 50)
    keys = ["do_resize", "size", "resample", "do_rescale",
            "rescale_factor", "do_normalize", "image_mean", "image_std"]
    for k in keys:
        if k in config:
            print(f"  {k:20s}: {config[k]}")

    print()
    print("=" * 50)
    print("KẾT LUẬN")
    print("=" * 50)
    size = config.get("size", {})
    do_resize = config.get("do_resize", False)

    if do_resize:
        print(f"  ✅ do_resize = True → processor TỰ resize về {size}")
        print(f"  ✅ ct_processor.py không cần resize thủ công")
    else:
        print(f"  ⚠️  do_resize = False → phải tự resize về {size}")

    h = size.get("height") or size.get("shortest_edge")
    if h == 896:
        print(f"  ✅ Target size = 896×896 — đúng với MedGemma 1.5 paper")
    elif h:
        print(f"  ⚠️  Target size = {h} — khác với paper (896×896)")

    check_chat_template(processor)


def check_chat_template(processor):
    print()
    print("=" * 50)
    print("CHAT TEMPLATE — system role xử lý như thế nào?")
    print("=" * 50)

    messages_with_system = [
        {"role": "system", "content": "You are a medical AI."},
        {"role": "user",   "content": "What is BMI?"},
    ]
    messages_no_system = [
        {"role": "user", "content": "You are a medical AI.\n\nWhat is BMI?"},
    ]

    out_with = processor.apply_chat_template(messages_with_system, tokenize=False)
    out_without = processor.apply_chat_template(messages_no_system, tokenize=False)

    print("\n[Có system role]:")
    print(repr(out_with))
    print("\n[Không có system role — gộp vào user]:")
    print(repr(out_without))

    if out_with == out_without:
        print("\n✅ GIỐNG NHAU — system role được gộp vào user turn")
        print("   → Rimine đúng: gộp system vào user không ảnh hưởng")
    else:
        print("\n⚠️  KHÁC NHAU — system role được xử lý riêng")
        print("   → Dùng system role riêng là đúng hơn")


if __name__ == "__main__":
    main()
