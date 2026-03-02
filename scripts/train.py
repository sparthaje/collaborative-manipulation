"""SFT training script for unified bimanual VLA.

Fine-tunes Cosmos-Reason2-2B with QLoRA on the BimanualVLADataset.
Based on models/example_sft.py (TRL + QLoRA pattern).

Usage:
    python -m scripts.train [--max_steps 100] [--batch_size 1] [--output_dir outputs/bimanual_vla]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig
from PIL import Image
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from trl import SFTConfig, SFTTrainer

from scripts.dataloader import BimanualVLADataset
from scripts.tokenize_actions import ActionTokenizer


MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
PIXELS_PER_TOKEN = 32**2
DEFAULT_DATASET = "data/grabber_picker_black_marker_20260226_211245"
DEFAULT_OUTPUT = "outputs/bimanual_vla"


def create_collate_fn(processor: Qwen3VLProcessor):
    """Create a collate function that applies the chat template.

    Each sample from the dataset is a list of message dicts.
    The collate function processes them with the VLM processor.
    """

    def collate_fn(examples: list[list[dict]]) -> dict:
        texts = []
        images_list = []

        for messages in examples:
            images = []
            processed_messages = []
            for msg in messages:
                new_msg = {"role": msg["role"], "content": []}
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_tensor = item["image"]
                        if isinstance(img_tensor, torch.Tensor):
                            if img_tensor.shape[0] == 3:
                                img_np = img_tensor.permute(1, 2, 0).numpy()
                            else:
                                img_np = img_tensor.numpy()
                            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                            pil_image = Image.fromarray(img_np)
                            images.append(pil_image)
                            new_msg["content"].append(
                                {"type": "image", "image": pil_image}
                            )
                        else:
                            images.append(item["image"])
                            new_msg["content"].append(item)
                    else:
                        new_msg["content"].append(item)
                processed_messages.append(new_msg)

            text = processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            images_list.append(images)

        batch = processor(
            text=texts,
            images=[img for imgs in images_list for img in imgs]
            if any(images_list)
            else None,
            return_tensors="pt",
            padding=True,
        )

        # For SFT, labels are the same as input_ids (shifted internally)
        batch["labels"] = batch["input_ids"].clone()

        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Train unified bimanual VLA with QLoRA")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument(
        "--min_vision_tokens",
        type=int,
        default=256,
        help="Min vision tokens per image (controls resolution)",
    )
    parser.add_argument(
        "--max_vision_tokens",
        type=int,
        default=1024,
        help="Max vision tokens per image (controls resolution)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    # 1. Load model with quantization
    print(f"Loading model: {MODEL_NAME}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME)

    # Limit vision tokens to control memory with 3 images per sample
    processor.image_processor.size = {
        "shortest_edge": args.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": args.max_vision_tokens * PIXELS_PER_TOKEN,
    }

    # 2. Add action tokens to tokenizer and resize embeddings
    print("Adding action tokens to tokenizer...")
    action_tokenizer = ActionTokenizer.from_stats_json(dataset_root / "meta" / "stats.json")
    num_added = action_tokenizer.register_with_tokenizer(processor.tokenizer)
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"Added {num_added} action tokens (total vocab: {len(processor.tokenizer)})")

    # 3. Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 4. Load dataset
    print("Loading dataset...")
    dataset = BimanualVLADataset(dataset_root)
    print(f"Dataset size: {len(dataset)} samples")

    # 5. Configure training
    training_args = SFTConfig(
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        optim="adamw_8bit",
        max_length=None,  # Don't truncate — image tokens would break
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        save_steps=50,
        save_total_limit=3,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 6. Create trainer with custom collate
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=create_collate_fn(processor),
        peft_config=peft_config,
    )

    # 7. Print GPU stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
        print(f"Memory reserved before training: {start_gpu_memory} GB")

    # 8. Train
    print(f"\nStarting training for {args.max_steps} steps...")
    trainer_stats = trainer.train()

    # 9. Print training stats
    print(f"\nTraining completed in {trainer_stats.metrics['train_runtime']:.1f} seconds")
    if torch.cuda.is_available():
        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        print(f"Peak GPU memory: {used_memory} GB")

    # 10. Save adapter and tokenizer
    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
