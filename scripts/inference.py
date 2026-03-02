"""Inference script for the unified bimanual VLA model.

Loads base model + LoRA adapter and generates 12-DoF action chunks
from 3 camera images + joint history.

Usage:
    # Run on a sample from the training dataset (for validation)
    python -m scripts.inference --adapter_dir outputs/bimanual_vla --episode 0 --frame 50

    # Run on raw image files
    python -m scripts.inference --adapter_dir outputs/bimanual_vla \
        --left_wrist path/to/left.jpg --right_wrist path/to/right.jpg --overhead path/to/top.jpg \
        --action_history "10,20,30,40,50,5,-30,20,10,40,-150,5;..."
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import PeftModel
from PIL import Image

from scripts.tokenize_actions import ActionTokenizer


MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
DEFAULT_DATASET = "data/grabber_picker_black_marker_20260226_211245"
PIXELS_PER_TOKEN = 32**2
NUM_JOINTS_PER_ARM = 6
NUM_JOINTS_TOTAL = 12
ACTION_CHUNK_LEN = 8
ACTION_HISTORY_LEN = 4

SYSTEM_PROMPT = (
    "You control both arms of a bimanual robot system. "
    "You receive three camera views (left wrist, right wrist, overhead) "
    "and recent joint history for all 12 joints (6 per arm). "
    "Output the next 8 timesteps of joint commands for both arms as 96 action tokens."
)

USER_PROMPT_TEMPLATE = (
    "Task: {task_description}.\n"
    "Here are your three camera views and recent joint history "
    "(12 joints per step: left arm then right arm):\n"
    "{action_history}\n"
    "Output the next 8 timesteps of joint commands (96 tokens: 8 steps x 12 joints)."
)


def load_model(
    adapter_dir: str | Path,
    dataset_root: str | Path = DEFAULT_DATASET,
    device: str = "auto",
) -> tuple:
    """Load base model + LoRA adapter + processor with action tokens.

    Returns:
        (model, processor, action_tokenizer)
    """
    adapter_dir = Path(adapter_dir)
    dataset_root = Path(dataset_root)

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

    # Register action tokens (must match training)
    stats_path = dataset_root / "meta" / "stats.json"
    action_tokenizer = ActionTokenizer.from_stats_json(stats_path)
    action_tokenizer.register_with_tokenizer(processor.tokenizer)
    model.resize_token_embeddings(len(processor.tokenizer))

    # Load LoRA adapter
    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # Set vision token budget
    processor.image_processor.size = {
        "shortest_edge": 256 * PIXELS_PER_TOKEN,
        "longest_edge": 1024 * PIXELS_PER_TOKEN,
    }

    return model, processor, action_tokenizer


def format_action_history_12d(
    history: np.ndarray,
    action_tokenizer: ActionTokenizer,
) -> str:
    """Format 4-step action history (12 joints) as tokenized text.

    Args:
        history: (4, 12) array of joint values (both arms).

    Returns:
        Formatted text block with left | right separation.
    """
    lines = []
    for i in range(history.shape[0]):
        label = f"t-{history.shape[0] - i}"
        left_tokens = action_tokenizer.encode_action(history[i, :NUM_JOINTS_PER_ARM])
        right_tokens = action_tokenizer.encode_action(history[i, NUM_JOINTS_PER_ARM:])
        left_names = action_tokenizer.token_ids_to_names(left_tokens)
        right_names = action_tokenizer.token_ids_to_names(right_tokens)
        lines.append(f"{label}: {' '.join(left_names)} | {' '.join(right_names)}")
    return "\n".join(lines)


def predict(
    model,
    processor,
    action_tokenizer: ActionTokenizer,
    left_wrist_image: Image.Image,
    right_wrist_image: Image.Image,
    overhead_image: Image.Image,
    action_history: np.ndarray,
    task_description: str = "Hand off the black marker (grabber to picker)",
    max_new_tokens: int = 150,
) -> np.ndarray:
    """Run inference to predict the next action chunk for both arms.

    Args:
        model: The fine-tuned model.
        processor: The VLM processor.
        action_tokenizer: ActionTokenizer instance.
        left_wrist_image: PIL Image from left wrist camera.
        right_wrist_image: PIL Image from right wrist camera.
        overhead_image: PIL Image from overhead camera.
        action_history: (4, 12) array of recent joint values (both arms).
        task_description: Text description of the task.
        max_new_tokens: Max tokens to generate.

    Returns:
        (8, 12) array of predicted joint commands (degrees) for both arms.
    """
    # Format action history
    history_text = format_action_history_12d(action_history, action_tokenizer)

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        action_history=history_text,
    )

    # Build messages with 3 images
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": left_wrist_image},
                {"type": "image", "image": right_wrist_image},
                {"type": "image", "image": overhead_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]

    # Parse action tokens from output
    action_token_pattern = re.compile(r"<action_j\d+_b\d+>")
    token_names = action_token_pattern.findall(output_text)

    expected_tokens = ACTION_CHUNK_LEN * NUM_JOINTS_TOTAL  # 96
    if len(token_names) < expected_tokens:
        print(f"Warning: Expected {expected_tokens} action tokens, got {len(token_names)}")
        print(f"Raw output: {output_text[:500]}")
        if len(token_names) == 0:
            return np.zeros((ACTION_CHUNK_LEN, NUM_JOINTS_TOTAL))

    # Take first 96 tokens (8 timesteps x 12 joints)
    token_names = token_names[:expected_tokens]
    token_ids = action_tokenizer.token_names_to_ids(token_names)

    # Decode: each timestep has 12 tokens (left 6 + right 6)
    # Both halves use the same 6-joint tokenizer
    result = np.zeros((ACTION_CHUNK_LEN, NUM_JOINTS_TOTAL))
    for t in range(ACTION_CHUNK_LEN):
        start = t * NUM_JOINTS_TOTAL
        left_ids = token_ids[start : start + NUM_JOINTS_PER_ARM]
        right_ids = token_ids[start + NUM_JOINTS_PER_ARM : start + NUM_JOINTS_TOTAL]
        result[t, :NUM_JOINTS_PER_ARM] = action_tokenizer.decode_action(left_ids)
        result[t, NUM_JOINTS_PER_ARM:] = action_tokenizer.decode_action(right_ids)

    return result


def run_on_dataset_sample(
    model, processor, action_tokenizer, dataset_root, episode, frame
):
    """Run inference on a sample from the training dataset."""
    from scripts.dataloader import BimanualVLADataset

    ds = BimanualVLADataset(dataset_root)

    # Find the matching sample
    target = None
    for i, (ep_idx, t_4fps) in enumerate(ds._index_table):
        if ep_idx == episode and t_4fps == frame:
            target = i
            break

    if target is None:
        print(f"Sample not found: episode={episode}, frame={frame}")
        return None

    # Get the sample
    sample = ds[target]
    ep_data = ds._episode_data[episode]
    frame_indices_4fps = ep_data["frame_indices_4fps"]

    # Extract 3 images
    images = []
    for img_idx in range(3):
        image_tensor = sample[1]["content"][img_idx]["image"]
        img_np = image_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))

    # Extract action history (12 joints)
    history_4fps_indices = [max(0, frame - ACTION_HISTORY_LEN + i) for i in range(ACTION_HISTORY_LEN)]
    history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
    action_history = ep_data["actions"][history_30fps]  # (4, 12)

    # Get ground truth action chunk (12 joints)
    chunk_4fps_indices = list(range(frame, frame + ACTION_CHUNK_LEN))
    chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
    gt_chunk = ep_data["actions"][chunk_30fps]  # (8, 12)

    # Run prediction
    aug = ds.augmentations[episode]
    predicted = predict(
        model,
        processor,
        action_tokenizer,
        images[0],
        images[1],
        images[2],
        action_history,
        task_description=aug["task_description"],
    )

    return predicted, gt_chunk


def main():
    parser = argparse.ArgumentParser(description="Unified Bimanual VLA Inference")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET)

    # Dataset sample mode
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--frame", type=int, default=None)

    # Raw image mode
    parser.add_argument("--left_wrist", type=str, default=None, help="Path to left wrist image")
    parser.add_argument("--right_wrist", type=str, default=None, help="Path to right wrist image")
    parser.add_argument("--overhead", type=str, default=None, help="Path to overhead image")
    parser.add_argument(
        "--action_history",
        type=str,
        default=None,
        help="Semicolon-separated rows of 12 comma-separated joint values",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Hand off the black marker (grabber to picker)",
    )
    args = parser.parse_args()

    model, processor, action_tokenizer = load_model(args.adapter_dir, args.dataset_root)

    if args.episode is not None and args.frame is not None:
        # Dataset sample mode
        result = run_on_dataset_sample(
            model,
            processor,
            action_tokenizer,
            args.dataset_root,
            args.episode,
            args.frame,
        )
        if result is not None:
            predicted, gt = result
            print("\nPredicted action chunk (8x12):")
            print("  Left arm (cols 0-5) | Right arm (cols 6-11)")
            print(predicted.round(2))
            print("\nGround truth action chunk (8x12):")
            print(gt.round(2))
            print(f"\nMean absolute error: {np.abs(predicted - gt).mean():.2f} degrees")
            print(f"  Left arm MAE:  {np.abs(predicted[:, :6] - gt[:, :6]).mean():.2f} degrees")
            print(f"  Right arm MAE: {np.abs(predicted[:, 6:] - gt[:, 6:]).mean():.2f} degrees")

    elif args.left_wrist and args.right_wrist and args.overhead:
        # Raw image mode
        left_img = Image.open(args.left_wrist)
        right_img = Image.open(args.right_wrist)
        overhead_img = Image.open(args.overhead)

        if args.action_history:
            rows = args.action_history.split(";")
            history = np.array([[float(v) for v in r.split(",")] for r in rows])
        else:
            history = np.zeros((ACTION_HISTORY_LEN, NUM_JOINTS_TOTAL))

        predicted = predict(
            model,
            processor,
            action_tokenizer,
            left_img,
            right_img,
            overhead_img,
            history,
            task_description=args.task,
        )
        print("\nPredicted action chunk (8x12):")
        print("  Left arm (cols 0-5) | Right arm (cols 6-11)")
        print(predicted.round(2))

    else:
        print("Provide either --episode/--frame or all three image paths for inference.")
        print("  Dataset mode: --episode 0 --frame 50")
        print("  Image mode:   --left_wrist l.jpg --right_wrist r.jpg --overhead o.jpg")


if __name__ == "__main__":
    main()
