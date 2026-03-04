"""Evaluation script for the Siamese VLA model.

Samples random (episode, timestep) pairs from the combined dataset,
runs inference for both left and right arms at each timestep, and
generates a Markdown report with images, predictions, and ground truth.

Usage:
    python -m scripts.inference --adapter_dir outputs/experiments/run_*/checkpoints
    python -m scripts.inference --adapter_dir outputs/experiments/run_*/checkpoints --num_samples 20 --seed 123
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from peft import PeftModel
from PIL import Image

from model.dataset import get_global_tokenizer
from model.tokenizer import ActionTokenizer


MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
DEFAULT_DATA_DIR = "data"

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
    data_dir: str | Path = DEFAULT_DATA_DIR,
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

    # Load global action tokenizer from all datasets and register tokens
    action_tokenizer = get_global_tokenizer(data_dir)
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
    system_prompt: str | None = None,
    user_prompt: str | None = None,
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
        task_description: Text description of the task (used only if user_prompt is None).
        system_prompt: Explicit system prompt. If None, uses hardcoded fallback.
        user_prompt: Explicit user prompt (already formatted). If None, uses hardcoded template.
        max_new_tokens: Max tokens to generate.

    Returns:
        Tuple of ((8, 6) array of predicted joint commands in degrees,
                  inference time in seconds).
    """
    # Format action history
    history_text = format_action_history(action_history, action_tokenizer)

    # Use explicit prompts or fall back to hardcoded ones
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPTS[arm]
    if user_prompt is None:
        user_prompt = USER_PROMPT_TEMPLATES[arm].format(
            task_description=task_description,
            action_history=history_text,
        )

    # Build messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
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

    # Generate (timed)
    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    inference_time = time.perf_counter() - t0

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

    # Parse action tokens from output, validating joint ordering.
    # The model may corrupt some tokens (e.g. emit "VII" instead of
    # <action_j3_b...>). We greedily assemble complete 6-joint timesteps
    # by requiring joints j0-j5 in order and skipping misordered tokens.
    action_token_pattern = re.compile(r"<action_j(\d+)_b\d+>")
    raw_matches = list(action_token_pattern.finditer(output_text))

    valid_names: list[str] = []
    expected_joint = 0
    for m in raw_matches:
        joint_idx = int(m.group(1))
        if joint_idx == expected_joint:
            valid_names.append(m.group(0))
            expected_joint = (expected_joint + 1) % 6
        else:
            # Misaligned token — try to start a new timestep if this is j0
            if joint_idx == 0:
                # Discard partial timestep
                leftover = len(valid_names) % 6
                if leftover:
                    valid_names = valid_names[:-leftover]
                valid_names.append(m.group(0))
                expected_joint = 1
            # Otherwise skip the corrupted token

    # Discard any trailing partial timestep
    leftover = len(valid_names) % 6
    if leftover:
        valid_names = valid_names[:-leftover]

    num_valid = len(valid_names)
    if num_valid < 48:
        print(f"Warning: Expected 48 action tokens, got {num_valid} valid")
        print(f"Raw output: {output_text[:500]}")
        if num_valid == 0:
            return np.zeros((8, 6)), inference_time

    # Take up to 48 tokens (8 timesteps * 6 joints)
    valid_names = valid_names[:48]
    token_ids = action_tokenizer.token_names_to_ids(valid_names)
    action_chunk = action_tokenizer.decode_action_chunk(token_ids)

    # Pad to 8 timesteps if we got fewer
    if action_chunk.shape[0] < 8:
        padded = np.zeros((8, 6))
        padded[: action_chunk.shape[0]] = action_chunk
        action_chunk = padded

    return action_chunk, inference_time, output_text


def array_to_md_table(arr: np.ndarray) -> str:
    """Convert an (T, 6) array to a Markdown table with timestep rows and joint columns."""
    header = "| t | j0 | j1 | j2 | j3 | j4 | j5 |"
    sep = "|---|------|------|------|------|------|------|"
    rows = [header, sep]
    for t in range(arr.shape[0]):
        vals = " | ".join(f"{arr[t, j]:.2f}" for j in range(arr.shape[1]))
        rows.append(f"| {t} | {vals} |")
    return "\n".join(rows)


def plot_joint_comparison(
    gt: np.ndarray,
    predicted: np.ndarray,
    arm: str,
    save_path: Path,
) -> None:
    """Plot ground truth vs predicted joint angles over time.

    Args:
        gt: (8, 6) ground truth joint angles.
        predicted: (8, 6) predicted joint angles.
        arm: "left" or "right".
        save_path: Path to save the figure.
    """
    timesteps = np.arange(gt.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{arm.capitalize()} Arm — Joint Angles: Expected vs Predicted", fontsize=14)

    for j, ax in enumerate(axes.flat):
        ax.plot(timesteps, gt[:, j], "o-", label="Expected", color="tab:blue")
        ax.plot(timesteps, predicted[:, j], "s--", label="Predicted", color="tab:orange")
        ax.set_title(f"Joint {j}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def evaluate_samples(
    model,
    processor,
    action_tokenizer: ActionTokenizer,
    data_dir: Path,
    output_dir: Path,
    num_samples: int = 10,
    seed: int = 42,
) -> None:
    """Sample random (episode, timestep) pairs and run inference for both arms.

    Saves images, numpy arrays, and a Markdown report to output_dir.
    """
    from model.dataset import (
        ACTION_CHUNK_LEN,
        ACTION_HISTORY_LEN,
        build_combined_dataset,
    )

    print(f"\nBuilding combined dataset from {data_dir}...")
    combined = build_combined_dataset(data_dir)

    # Collect unique (dataset_idx, ep_idx, t_4fps) pairs from the concat dataset.
    # The index table has pairs: (ep, t, "left") at even indices, (ep, t, "right") at odd.
    # We iterate the sub-datasets to track which dataset each sample came from.
    unique_pairs: list[tuple[int, int, int, int]] = []  # (ds_idx, ep_idx, t_4fps, concat_left_idx)
    seen: set[tuple[int, int, int]] = set()

    offset = 0
    for ds_idx, ds in enumerate(combined.datasets):
        for local_idx, (ep_idx, t_4fps, arm) in enumerate(ds._index_table):
            if arm == "left":
                key = (ds_idx, ep_idx, t_4fps)
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append((ds_idx, ep_idx, t_4fps, offset + local_idx))
        offset += len(ds)

    print(f"Found {len(unique_pairs)} unique (episode, timestep) pairs")

    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(len(unique_pairs), size=min(num_samples, len(unique_pairs)), replace=False)
    sample_indices.sort()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sample_num, pair_idx in enumerate(sample_indices):
        ds_idx, ep_idx, t_4fps, concat_left_idx = unique_pairs[pair_idx]
        ds = combined.datasets[ds_idx]
        ep_data = ds._episode_data[ep_idx]
        aug = ds.augmentations[ep_idx]
        frame_indices_4fps = ep_data["frame_indices_4fps"]

        dataset_name = ds.dataset_root.name
        print(f"\n--- Sample {sample_num:02d}: episode {ep_idx}, frame {t_4fps} ({dataset_name}) ---")

        sample_dir = output_dir / f"sample_{sample_num:02d}"
        sample_dir.mkdir(exist_ok=True)

        arm_results = {}
        for arm in ("left", "right"):
            # Load image
            current_30fps = frame_indices_4fps[t_4fps]
            image_tensor = ds._load_frame(ep_idx, current_30fps, arm)
            img_np = image_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Save image
            img_path = sample_dir / f"{arm}_wrist.png"
            pil_image.save(img_path)

            # Action history
            history_4fps_indices = [
                max(0, t_4fps - ACTION_HISTORY_LEN + i)
                for i in range(ACTION_HISTORY_LEN)
            ]
            history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
            arm_slice = slice(0, 6) if arm == "left" else slice(6, 12)
            action_history = ep_data["actions"][history_30fps, arm_slice]

            # Ground truth chunk
            chunk_4fps_indices = list(range(t_4fps, t_4fps + ACTION_CHUNK_LEN))
            chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
            gt_chunk = ep_data["actions"][chunk_30fps, arm_slice]

            # Build prompts from augmentation (matching training)
            arm_key = f"{arm}_arm"
            history_text = format_action_history(action_history, action_tokenizer)
            sys_prompt = aug[arm_key]["system_prompt"]
            usr_prompt = aug[arm_key]["user_prompt_template"].format(
                task_description=aug["task_description"],
                action_history=history_text,
            )

            # Run inference
            predicted, inf_time, raw_output = predict(
                model, processor, action_tokenizer,
                pil_image, action_history, arm,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
            )

            # Build expected token string from ground truth
            expected_tokens = []
            for t_step in range(gt_chunk.shape[0]):
                token_ids = action_tokenizer.encode_action(gt_chunk[t_step])
                token_names = action_tokenizer.token_ids_to_names(token_ids)
                expected_tokens.extend(token_names)
            expected_token_str = " ".join(expected_tokens)

            # Save numpy arrays and joint plot
            np.save(sample_dir / f"{arm}_predicted.npy", predicted)
            np.save(sample_dir / f"{arm}_gt.npy", gt_chunk)
            plot_joint_comparison(
                gt_chunk, predicted, arm,
                sample_dir / f"{arm}_joints.png",
            )

            mae = float(np.abs(predicted - gt_chunk).mean())
            print(f"  {arm:5s} MAE: {mae:.2f} degrees  ({inf_time:.2f}s)")

            arm_results[arm] = {
                "predicted": predicted,
                "gt": gt_chunk,
                "mae": mae,
                "inference_time": inf_time,
                "user_prompt": usr_prompt,
                "raw_output": raw_output,
                "expected_tokens": expected_token_str,
            }

        results.append({
            "sample_num": sample_num,
            "ep_idx": ep_idx,
            "t_4fps": t_4fps,
            "dataset_name": dataset_name,
            "task_description": aug["task_description"],
            "arms": arm_results,
        })

    # Generate report
    _write_report(results, output_dir)
    print(f"\nReport written to {output_dir / 'report.md'}")


def _write_report(results: list[dict], output_dir: Path) -> None:
    """Write the Markdown report summarizing all samples."""
    from datetime import datetime

    lines = [
        "# Siamese VLA Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {len(results)}",
        "",
        "---",
        "",
    ]

    for r in results:
        n = r["sample_num"]
        lines.append(f"## Sample {n:02d} — Episode {r['ep_idx']}, Frame {r['t_4fps']} (dataset: {r['dataset_name']})")
        lines.append("")
        lines.append(f"**Task:** {r['task_description']}")
        lines.append("")

        for arm in ("left", "right"):
            arm_data = r["arms"][arm]
            lines.append(f"### {arm.capitalize()} Arm")
            lines.append("")
            lines.append("**Input Image:**")
            lines.append(f"![{arm} wrist](sample_{n:02d}/{arm}_wrist.png)")
            lines.append("")
            lines.append("**Prompt (user message):**")
            for prompt_line in arm_data["user_prompt"].split("\n"):
                lines.append(f"{prompt_line}")
            lines.append("")
            lines.append("**Expected tokens:**")
            lines.append(f"```")
            lines.append(arm_data["expected_tokens"])
            lines.append(f"```")
            lines.append("")
            lines.append("**Raw model output:**")
            lines.append(f"```")
            lines.append(arm_data["raw_output"])
            lines.append(f"```")
            lines.append("")
            lines.append("**Ground Truth (8x6):**")
            lines.append(array_to_md_table(arm_data["gt"]))
            lines.append("")
            lines.append("**Predicted (8x6):**")
            lines.append(array_to_md_table(arm_data["predicted"]))
            lines.append("")
            lines.append(f"**Joint Angle Comparison:**")
            lines.append(f"![{arm} joints](sample_{n:02d}/{arm}_joints.png)")
            lines.append("")
            lines.append(f"**MAE:** {arm_data['mae']:.2f} degrees")
            lines.append(f"**Inference time:** {arm_data['inference_time']:.2f}s")
            lines.append("")

        # Comparison table
        left_mae = r["arms"]["left"]["mae"]
        right_mae = r["arms"]["right"]["mae"]
        left_time = r["arms"]["left"]["inference_time"]
        right_time = r["arms"]["right"]["inference_time"]
        lines.append("### Left vs Right Comparison")
        lines.append("")
        lines.append("| Metric | Left Arm | Right Arm |")
        lines.append("|--------|----------|-----------|")
        lines.append(f"| MAE    | {left_mae:.2f} deg | {right_mae:.2f} deg |")
        lines.append(f"| Inference time | {left_time:.2f}s | {right_time:.2f}s |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Sample | Dataset | Episode | Frame | Left MAE | Right MAE | Left Time | Right Time |")
    lines.append("|--------|---------|---------|-------|----------|-----------|-----------|------------|")
    for r in results:
        n = r["sample_num"]
        left_mae = r["arms"]["left"]["mae"]
        right_mae = r["arms"]["right"]["mae"]
        left_time = r["arms"]["left"]["inference_time"]
        right_time = r["arms"]["right"]["inference_time"]
        lines.append(
            f"| {n:02d} | {r['dataset_name']} | {r['ep_idx']} | {r['t_4fps']} "
            f"| {left_mae:.2f} deg | {right_mae:.2f} deg "
            f"| {left_time:.2f}s | {right_time:.2f}s |"
        )

    avg_left = np.mean([r["arms"]["left"]["mae"] for r in results])
    avg_right = np.mean([r["arms"]["right"]["mae"] for r in results])
    avg_left_time = np.mean([r["arms"]["left"]["inference_time"] for r in results])
    avg_right_time = np.mean([r["arms"]["right"]["inference_time"] for r in results])
    lines.append(
        f"| **Avg** | | | | **{avg_left:.2f} deg** | **{avg_right:.2f} deg** "
        f"| **{avg_left_time:.2f}s** | **{avg_right_time:.2f}s** |"
    )
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Siamese VLA Evaluation")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Parent directory containing all dataset subdirectories")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of random (episode, timestep) pairs to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_samples",
                        help="Directory to write results and report")
    args = parser.parse_args()

    model, processor, action_tokenizer = load_model(args.adapter_dir, args.data_dir)

    evaluate_samples(
        model, processor, action_tokenizer,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
