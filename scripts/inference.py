"""Inference script for the Siamese VLA model.

Loads base model + LoRA adapter and generates action chunks
from a wrist camera image + action history.

Usage:
    # Run on a sample from the training dataset (for validation)
    python -m scripts.inference --adapter_dir outputs/siamese_vla --episode 0 --frame 50 --arm left

    # Run on a raw image file
    python -m scripts.inference --adapter_dir outputs/siamese_vla --image path/to/img.jpg --arm left \
        --action_history "10.0,20.0,30.0,40.0,50.0,5.0;11.0,21.0,31.0,41.0,51.0,6.0;..."
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
DEFAULT_DATASET = "data/grabber_picker_black_marker_20260228_150311"

SYSTEM_PROMPTS = {
    "left": (
        "You are controlling the LEFT arm of a two-arm robot system. "
        "You receive your wrist camera image and your recent action history. "
        "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 action tokens."
    ),
    "right": (
        "You are controlling the RIGHT arm of a two-arm robot system. "
        "You receive your wrist camera image and your recent action history. "
        "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 action tokens."
    ),
}

USER_PROMPT_TEMPLATES = {
    "left": (
        "You are the LEFT arm. The task is: {task_description}. "
        "Here is your current wrist camera image and your last 4 actions:\n"
        "{action_history}\n"
        "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."
    ),
    "right": (
        "You are the RIGHT arm. The task is: {task_description}. "
        "Here is your current wrist camera image and your last 4 actions:\n"
        "{action_history}\n"
        "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."
    ),
}

PIXELS_PER_TOKEN = 32**2


def load_model(
    adapter_dir: str | Path,
    device: str = "auto",
) -> tuple:
    """Load base model + LoRA adapter + processor with action tokens.

    Returns:
        (model, processor, action_tokenizer)
    """
    adapter_dir = Path(adapter_dir)

    # Load base model
    print(f"Loading base model: {MODEL_NAME}")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa",
    )

    # Load processor and action tokenizer
    processor = transformers.Qwen3VLProcessor.from_pretrained(MODEL_NAME)

    # Load action tokenizer from stats and register tokens
    stats_path = Path(DEFAULT_DATASET) / "meta" / "stats.json"
    action_tokenizer = ActionTokenizer.from_stats_json(stats_path)
    action_tokenizer.register_with_tokenizer(processor.tokenizer)
    model.resize_token_embeddings(len(processor.tokenizer))

    # Load LoRA adapter
    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # Set vision token budget
    min_vision_tokens = 256
    max_vision_tokens = 1024  # Reduced for inference speed
    processor.image_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }

    return model, processor, action_tokenizer


def format_action_history(
    history: np.ndarray,
    action_tokenizer: ActionTokenizer,
) -> str:
    """Format 4-step action history as tokenized text.

    Args:
        history: (4, 6) array of joint values.
        action_tokenizer: ActionTokenizer instance.

    Returns:
        Formatted text block.
    """
    lines = []
    for i in range(history.shape[0]):
        label = f"t-{history.shape[0] - i}"
        token_ids = action_tokenizer.encode_action(history[i])
        token_names = action_tokenizer.token_ids_to_names(token_ids)
        lines.append(f"{label}: {' '.join(token_names)}")
    return "\n".join(lines)


def predict(
    model,
    processor,
    action_tokenizer: ActionTokenizer,
    image: Image.Image,
    action_history: np.ndarray,
    arm: str,
    task_description: str = "Hand off the black marker (grabber to picker)",
    max_new_tokens: int = 128,
) -> np.ndarray:
    """Run inference to predict the next action chunk.

    Args:
        model: The fine-tuned model.
        processor: The VLM processor.
        action_tokenizer: ActionTokenizer instance.
        image: PIL Image from wrist camera.
        action_history: (4, 6) array of recent joint values.
        arm: "left" or "right".
        task_description: Text description of the task.
        max_new_tokens: Max tokens to generate.

    Returns:
        (8, 6) array of predicted joint commands (degrees).
    """
    # Format action history
    history_text = format_action_history(action_history, action_tokenizer)

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATES[arm].format(
        task_description=task_description,
        action_history=history_text,
    )

    # Build messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPTS[arm]}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
        out_ids[len(in_ids):]
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

    if len(token_names) < 48:
        print(f"Warning: Expected 48 action tokens, got {len(token_names)}")
        print(f"Raw output: {output_text[:500]}")
        # Pad with zeros if insufficient tokens
        if len(token_names) == 0:
            return np.zeros((8, 6))

    # Take first 48 tokens (8 timesteps * 6 joints)
    token_names = token_names[:48]
    token_ids = action_tokenizer.token_names_to_ids(token_names)
    action_chunk = action_tokenizer.decode_action_chunk(token_ids)

    return action_chunk


def run_on_dataset_sample(
    model, processor, action_tokenizer, dataset_root, episode, frame, arm
):
    """Run inference on a sample from the training dataset."""
    from scripts.dataloader import SiameseVLADataset

    ds = SiameseVLADataset(dataset_root)

    # Find the matching sample in the dataset
    target = None
    for i, (ep_idx, t_4fps, sample_arm) in enumerate(ds._index_table):
        if ep_idx == episode and t_4fps == frame and sample_arm == arm:
            target = i
            break

    if target is None:
        print(f"Sample not found: episode={episode}, frame={frame}, arm={arm}")
        return None

    # Get the sample for ground truth
    sample = ds[target]
    ep_data = ds._episode_data[episode]
    frame_indices_4fps = ep_data["frame_indices_4fps"]

    # Extract image
    image_tensor = sample[1]["content"][0]["image"]
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)

    # Extract action history
    history_4fps_indices = [max(0, frame - 4 + i) for i in range(4)]
    history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
    arm_slice = slice(0, 6) if arm == "left" else slice(6, 12)
    action_history = ep_data["actions"][history_30fps, arm_slice]

    # Get ground truth action chunk
    chunk_4fps_indices = list(range(frame, frame + 8))
    chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
    gt_chunk = ep_data["actions"][chunk_30fps, arm_slice]

    # Run prediction
    aug = ds.augmentations[episode]
    predicted = predict(
        model, processor, action_tokenizer,
        pil_image, action_history, arm,
        task_description=aug["task_description"],
    )

    return predicted, gt_chunk


def main():
    parser = argparse.ArgumentParser(description="Siamese VLA Inference")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--arm", type=str, required=True, choices=["left", "right"])

    # Dataset sample mode
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--frame", type=int, default=None)

    # Raw image mode
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument(
        "--action_history", type=str, default=None,
        help="Semicolon-separated rows of 6 comma-separated joint values"
    )
    parser.add_argument("--task", type=str,
                        default="Hand off the black marker (grabber to picker)")
    args = parser.parse_args()

    model, processor, action_tokenizer = load_model(args.adapter_dir)

    if args.episode is not None and args.frame is not None:
        # Dataset sample mode
        result = run_on_dataset_sample(
            model, processor, action_tokenizer,
            args.dataset_root, args.episode, args.frame, args.arm,
        )
        if result is not None:
            predicted, gt = result
            print("\nPredicted action chunk (8x6):")
            print(predicted.round(2))
            print("\nGround truth action chunk (8x6):")
            print(gt.round(2))
            print(f"\nMean absolute error: {np.abs(predicted - gt).mean():.2f} degrees")

    elif args.image is not None:
        # Raw image mode
        pil_image = Image.open(args.image)

        if args.action_history:
            rows = args.action_history.split(";")
            history = np.array([[float(v) for v in r.split(",")] for r in rows])
        else:
            history = np.zeros((4, 6))

        predicted = predict(
            model, processor, action_tokenizer,
            pil_image, history, args.arm,
            task_description=args.task,
        )
        print("\nPredicted action chunk (8x6):")
        print(predicted.round(2))

    else:
        print("Provide either --episode/--frame or --image for inference.")


if __name__ == "__main__":
    main()
