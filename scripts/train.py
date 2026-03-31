"""SFT training script for Siamese VLA fine-tuning.

Fine-tunes Cosmos-Reason2-2B with QLoRA on bimanual handoff datasets.
Uses TRL SFTTrainer with custom action tokens.

Usage:
    python -m scripts.train [--max_steps 100] [--batch_size 2] [--data_dir data]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from trl import SFTConfig, SFTTrainer

from model.dataset import build_combined_dataset, get_global_tokenizer


MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_BASE = "outputs/experiments"


def create_collate_fn(processor: Qwen3VLProcessor):
    """Create a collate function that applies the chat template.

    Each sample from the dataset is a list of message dicts.
    The collate function processes them with the VLM processor.
    Labels are masked so only the assistant response is trained on.
    """
    IGNORE_INDEX = -100

    def collate_fn(examples: list[list[dict]]) -> dict:
        texts = []
        images_list = []

        # Also build prompt-only versions (without assistant) to find the boundary
        prompt_texts = []

        for messages in examples:
            images = []
            processed_messages = []
            prompt_messages = []  # everything except the assistant turn
            for msg in messages:
                new_msg = {"role": msg["role"], "content": []}
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_tensor = item["image"]
                        if isinstance(img_tensor, torch.Tensor):
                            from PIL import Image
                            import numpy as np

                            if img_tensor.shape[0] == 3:
                                img_np = img_tensor.permute(1, 2, 0).numpy()
                            else:
                                img_np = img_tensor.numpy()
                            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                            pil_image = Image.fromarray(img_np)
                            images.append(pil_image)
                            new_msg["content"].append({"type": "image", "image": pil_image})
                        else:
                            images.append(item["image"])
                            new_msg["content"].append(item)
                    else:
                        new_msg["content"].append(item)
                processed_messages.append(new_msg)
                if msg["role"] != "assistant":
                    prompt_messages.append(new_msg)

            # Full text (system + user + assistant)
            text = processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            images_list.append(images)

            # Prompt-only text (system + user) with generation prompt
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(prompt_text)

        # Tokenize full texts with images
        flat_images = [img for imgs in images_list for img in imgs] if any(images_list) else None
        batch = processor(
            text=texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize prompt-only texts to find where the assistant response starts
        prompt_batch = processor(
            text=prompt_texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )

        # Build labels: mask everything before the assistant response with IGNORE_INDEX
        labels = batch["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        for i in range(labels.shape[0]):
            # Mask padding tokens
            labels[i, labels[i] == pad_token_id] = IGNORE_INDEX
            # Mask prompt tokens (everything before assistant response)
            prompt_len = (prompt_batch["attention_mask"][i] == 1).sum().item()
            labels[i, :prompt_len] = IGNORE_INDEX

        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Train Siamese VLA with QLoRA")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Parent directory containing all dataset subdirectories")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/experiments/run_<timestamp>)")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Resolve output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{DEFAULT_OUTPUT_BASE}/run_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 1. Load model with quantization
    print(f"Loading model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        quantization_config=bnb_config,
    )

    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME)

    # 2. Add action tokens to tokenizer and resize embeddings
    print("Adding action tokens to tokenizer...")
    action_tokenizer = get_global_tokenizer(data_dir)
    num_added = action_tokenizer.register_with_tokenizer(processor.tokenizer)
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"Added {num_added} action tokens (total vocab: {len(processor.tokenizer)})")

    # 3. Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )

    # 4. Load combined dataset
    print("Loading datasets...")
    dataset = build_combined_dataset(data_dir)
    print(f"Total dataset size: {len(dataset)} samples")

    # 5. Configure training
    training_args = SFTConfig(
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        optim="adamw_8bit",
        bf16=True,
        max_length=None,  # Critical: don't truncate image tokens
        output_dir=str(output_dir / "checkpoints"),
        logging_steps=args.logging_steps,
        report_to="none",  # Set to "tensorboard" if tensorboard is installed
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 6. Create trainer
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
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
        print(f"Memory reserved before training: {start_gpu_memory} GB")

    # 8. Train
    print(f"\nStarting training for {args.max_steps} steps...")
    trainer_stats = trainer.train()

    # 9. Print training stats
    print(f"\nTraining completed in {trainer_stats.metrics['train_runtime']:.1f} seconds")
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"Peak GPU memory: {used_memory} GB")

    # 10. Save adapter
    save_dir = output_dir / "checkpoints"
    print(f"Saving adapter to {save_dir}")
    trainer.save_model(str(save_dir))

    # Also save the tokenizer with action tokens
    processor.tokenizer.save_pretrained(str(save_dir))

    print("Training complete!")


if __name__ == "__main__":
    main()
