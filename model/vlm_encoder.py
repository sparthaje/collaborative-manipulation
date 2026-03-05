"""Frozen VLM encoder for extracting embeddings from fine-tuned Cosmos-Reason2-2B.

Loads the base model with a LoRA adapter, freezes all parameters, and provides
methods for encoding VLM chat messages into fixed-size embedding vectors by
extracting the last hidden state at the </think> token position.

Usage:
    python -m model.vlm_encoder --adapter_dir outputs/experiments/run_*/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import PeftModel
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from model.dataset import get_global_tokenizer


MODEL_NAME = "nvidia/Cosmos-Reason2-2B"


class FrozenVLMEncoder:
    """Frozen VLM encoder that extracts hidden-state embeddings at the </think> token.

    Loads the base Cosmos-Reason2-2B model with a LoRA adapter, registers action
    tokens, and freezes all parameters. The encode() method runs a forward pass
    and returns the last-layer hidden state at the </think> token position.
    """

    def __init__(
        self,
        adapter_dir: str | Path,
        data_dir: str | Path = "data",
        device: str = "auto",
    ) -> None:
        adapter_dir = Path(adapter_dir)
        data_dir = Path(data_dir)

        # Load base model
        print(f"Loading base model: {MODEL_NAME}")
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map=device,
            attn_implementation="sdpa",
        )

        # Load processor
        processor = transformers.Qwen3VLProcessor.from_pretrained(MODEL_NAME)

        # Load global action tokenizer and register
        action_tokenizer = get_global_tokenizer(data_dir)
        action_tokenizer.register_with_tokenizer(processor.tokenizer)
        model.resize_token_embeddings(len(processor.tokenizer))

        # Load LoRA adapter
        print(f"Loading adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.processor = processor
        self.action_tokenizer = action_tokenizer

    def _collate_messages(self, messages_list: list[list[dict]]) -> dict:
        """Convert a list of chat message lists into a batched processor output.

        Handles converting torch.Tensor images to PIL Images and applying the
        chat template via the processor.

        Args:
            messages_list: List of B chat message lists, each in the format
                returned by SiameseVLADataset.__getitem__.

        Returns:
            Batched processor output dict ready for model forward pass.
        """
        texts = []
        images_list = []

        for messages in messages_list:
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
                            new_msg["content"].append({"type": "image", "image": pil_image})
                        else:
                            images.append(item["image"])
                            new_msg["content"].append(item)
                    else:
                        new_msg["content"].append(item)
                processed_messages.append(new_msg)

            text = self.processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            images_list.append(images)

        flat_images = [img for imgs in images_list for img in imgs] if any(images_list) else None
        batch = self.processor(
            text=texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )
        return batch

    def encode(self, messages_list: list[list[dict]]) -> Tensor:
        """Encode a batch of chat message lists into fixed-size embeddings.

        Runs a forward pass (not generate) with output_hidden_states=True,
        then extracts the last hidden state at the </think> token position
        for each sequence in the batch.

        Args:
            messages_list: List of B chat message lists, each in the format
                returned by SiameseVLADataset.__getitem__.

        Returns:
            Tensor of shape (B, 2048) containing the embedding for each sample.
        """
        batch = self._collate_messages(messages_list)

        # Move inputs to model device
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                **batch,
                output_hidden_states=True,
            )

        # Last layer hidden states: (B, seq_len, 2048)
        last_hidden = outputs.hidden_states[-1]
        input_ids = batch["input_ids"]

        # Find </think> token position in each sequence
        think_end_id = self.processor.tokenizer.convert_tokens_to_ids("</think>")

        B = input_ids.shape[0]
        embeddings = []
        for i in range(B):
            # Find all positions where </think> appears
            positions = (input_ids[i] == think_end_id).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                # Fallback: use the last non-padding token
                attention_mask = batch["attention_mask"][i]
                last_pos = attention_mask.sum().item() - 1
                embeddings.append(last_hidden[i, last_pos, :])
            else:
                # Use the last occurrence of </think>
                pos = positions[-1].item()
                embeddings.append(last_hidden[i, pos, :])

        return torch.stack(embeddings, dim=0)

    def precompute_and_cache(
        self,
        dataset,
        cache_dir: str | Path,
        batch_size: int = 4,
    ) -> None:
        """Precompute VLM embeddings for all samples in a BimanualDiffusionDataset.

        Iterates through every sample in the dataset, extracts VLM messages for
        both arms, runs encode(), and saves the resulting embeddings to disk.

        Args:
            dataset: A BimanualDiffusionDataset instance. Must have attributes:
                - dataset_root (Path): root directory with a .name attribute
                - Each sample should provide VLM messages for left and right arms.
            cache_dir: Directory to save cached embedding files.
            batch_size: Number of samples to process at once.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = dataset.dataset_root.name
        num_samples = len(dataset)

        # Collect all sample indices and their metadata
        samples_to_process = []
        for idx in range(num_samples):
            ep_idx, t_start = dataset._index_table[idx]
            filename = f"{dataset_name}_ep{ep_idx:04d}_t{t_start:04d}.pt"
            filepath = cache_dir / filename

            if filepath.exists():
                continue

            samples_to_process.append({
                "idx": idx,
                "ep_idx": ep_idx,
                "t_start": t_start,
                "filepath": filepath,
            })

        if not samples_to_process:
            print(f"All {num_samples} samples already cached in {cache_dir}")
            return

        print(f"Processing {len(samples_to_process)} samples "
              f"({num_samples - len(samples_to_process)} already cached)")

        # Process in batches
        for batch_start in tqdm(
            range(0, len(samples_to_process), batch_size),
            desc=f"Encoding {dataset_name}",
            total=(len(samples_to_process) + batch_size - 1) // batch_size,
        ):
            batch_samples = samples_to_process[batch_start:batch_start + batch_size]

            # Fetch samples and collect left and right messages for the batch
            left_messages_batch = []
            right_messages_batch = []
            for s in batch_samples:
                sample = dataset[s["idx"]]
                left_messages_batch.append(sample["left_vlm_messages"])
                right_messages_batch.append(sample["right_vlm_messages"])

            # Encode both arms in a single forward pass
            combined = left_messages_batch + right_messages_batch
            all_embeddings = self.encode(combined)  # (2B, 2048)
            left_embeddings = all_embeddings[:len(left_messages_batch)]
            right_embeddings = all_embeddings[len(left_messages_batch):]

            # Save each sample's embeddings
            for i, sample_info in enumerate(batch_samples):
                torch.save(
                    {
                        "left_embedding": left_embeddings[i].cpu(),
                        "right_embedding": right_embeddings[i].cpu(),
                    },
                    sample_info["filepath"],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FrozenVLMEncoder loading")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Parent directory containing all dataset subdirectories",
    )
    args = parser.parse_args()

    encoder = FrozenVLMEncoder(
        adapter_dir=args.adapter_dir,
        data_dir=args.data_dir,
    )

    device = next(encoder.model.parameters()).device
    print(f"Model loaded on device: {device}")
    print(f"Processor vocab size: {len(encoder.processor.tokenizer)}")
    print(f"Action tokenizer bins: {encoder.action_tokenizer.num_bins}")
