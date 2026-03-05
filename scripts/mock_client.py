"""Mock client that replays dataset episodes through the inference server.

Loads images and joint states from a recorded dataset, sends them to the
server's /predict endpoint, and compares predicted actions against ground truth.

Usage:
    python -m scripts.mock_client \
        --server http://localhost:5000 \
        --data_dir data \
        --dataset picker_grabber_black_marker_20260226_204853 \
        --episode 0 \
        --num_steps 10
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from model.dataset import (
    ACTION_CHUNK_LEN,
    ACTION_HISTORY_LEN,
    DATASET_FPS,
    SiameseVLADataset,
)
from model.tokenizer import ActionTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Mock client for testing inference server.")
    parser.add_argument("--server", required=True, help='Server URL, e.g. "http://localhost:5000".')
    parser.add_argument("--data_dir", default="data", help="Parent directory with datasets.")
    parser.add_argument("--dataset", default=None, help="Dataset name (subfolder of data_dir). Uses first found if omitted.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of 4fps steps to run (default: all valid).")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-step output.")
    return parser.parse_args()


def tensor_to_jpeg(image_tensor: torch.Tensor) -> bytes:
    """Convert CHW [0,1] tensor to JPEG bytes."""
    img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Find dataset
    if args.dataset:
        dataset_root = data_dir / args.dataset
    else:
        dataset_dirs = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and (p / "meta" / "info.json").exists()
        )
        if not dataset_dirs:
            raise FileNotFoundError(f"No datasets found under {data_dir}")
        dataset_root = dataset_dirs[0]

    print(f"Loading dataset: {dataset_root.name}")

    # Build global tokenizer
    all_stats = sorted(data_dir.glob("*/meta/stats.json"))
    action_tokenizer = ActionTokenizer.from_multiple_stats(all_stats)

    ds = SiameseVLADataset(
        dataset_root,
        action_tokenizer=action_tokenizer,
    )

    # Find the requested episode
    ep_idx = args.episode
    if ep_idx not in ds._episode_data:
        available = sorted(ds._episode_data.keys())
        raise ValueError(f"Episode {ep_idx} not found. Available: {available}")

    ep_data = ds._episode_data[ep_idx]
    aug = ds.augmentations[ep_idx]
    frame_indices_4fps = ep_data["frame_indices_4fps"]
    num_valid = len(frame_indices_4fps) - ACTION_CHUNK_LEN
    task = aug["task"]

    num_steps = min(args.num_steps, num_valid) if args.num_steps else num_valid
    print(f"Episode {ep_idx}: {num_valid} valid steps, running {num_steps}")
    print(f"Task: {task}")

    predict_url = f"{args.server.rstrip('/')}/predict"

    # Check server health
    try:
        resp = requests.post(f"{args.server.rstrip('/')}/health", timeout=5)
        health = resp.json()
        print(f"Server status: {health.get('status')}, device: {health.get('device')}")
    except requests.RequestException as e:
        print(f"Warning: health check failed: {e}")

    # Initialize action history with the initial position
    arm_slice_left = slice(0, 6)
    arm_slice_right = slice(6, 12)

    initial_actions = ep_data["actions"][0]  # (12,)
    left_history = np.tile(initial_actions[arm_slice_left], (4, 1))
    right_history = np.tile(initial_actions[arm_slice_right], (4, 1))

    errors_left = []
    errors_right = []

    print(f"\nStarting mock inference loop ({num_steps} steps)...")
    print("-" * 70)

    for step in range(num_steps):
        t_4fps = step
        current_30fps = frame_indices_4fps[t_4fps]

        # Load images from dataset
        left_image = ds._load_frame(ep_idx, current_30fps, "left")
        right_image = ds._load_frame(ep_idx, current_30fps, "right")

        left_jpeg = tensor_to_jpeg(left_image)
        right_jpeg = tensor_to_jpeg(right_image)

        # Build ground truth action history from dataset
        history_4fps_indices = [max(0, t_4fps - ACTION_HISTORY_LEN + i) for i in range(ACTION_HISTORY_LEN)]
        history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
        gt_left_history = ep_data["actions"][history_30fps][:, arm_slice_left]
        gt_right_history = ep_data["actions"][history_30fps][:, arm_slice_right]

        # Ground truth future actions
        chunk_4fps_indices = list(range(t_4fps, t_4fps + ACTION_CHUNK_LEN))
        chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
        gt_left_chunk = ep_data["actions"][chunk_30fps][:, arm_slice_left]  # (8, 6)
        gt_right_chunk = ep_data["actions"][chunk_30fps][:, arm_slice_right]  # (8, 6)

        # Send to server (use ground truth history for consistency)
        files = {
            "left_image": ("left.jpg", left_jpeg, "image/jpeg"),
            "right_image": ("right.jpg", right_jpeg, "image/jpeg"),
        }
        data = {
            "task": task,
            "left_history": json.dumps(gt_left_history.tolist()),
            "right_history": json.dumps(gt_right_history.tolist()),
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(predict_url, files=files, data=data, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Step {step}: server error: {e}")
            continue
        elapsed = time.perf_counter() - t0

        result = resp.json()
        pred_actions = np.array(result["actions"])  # (2, 8, 6)
        cot = result.get("cot", ["", ""])

        pred_left = pred_actions[0]   # (8, 6)
        pred_right = pred_actions[1]  # (8, 6)

        # Compute MAE against ground truth
        mae_left = np.mean(np.abs(pred_left - gt_left_chunk))
        mae_right = np.mean(np.abs(pred_right - gt_right_chunk))
        errors_left.append(mae_left)
        errors_right.append(mae_right)

        if args.verbose:
            print(f"Step {step:3d} | t_30fps={current_30fps:5d} | "
                  f"MAE L={mae_left:6.2f}° R={mae_right:6.2f}° | "
                  f"server={elapsed:.2f}s")
            if cot[0]:
                print(f"         CoT L: {cot[0][:80]}")
            if cot[1]:
                print(f"         CoT R: {cot[1][:80]}")
        else:
            print(f"Step {step:3d}/{num_steps} | MAE L={mae_left:6.2f}° R={mae_right:6.2f}° | {elapsed:.2f}s")

    print("-" * 70)
    if errors_left:
        print(f"Mean MAE  Left: {np.mean(errors_left):.2f}°  Right: {np.mean(errors_right):.2f}°")
        print(f"Std  MAE  Left: {np.std(errors_left):.2f}°  Right: {np.std(errors_right):.2f}°")
    print("Done.")


if __name__ == "__main__":
    main()
