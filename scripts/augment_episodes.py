"""Episode augmentation script for unified bimanual VLA fine-tuning.

Reads episode metadata and generates per-episode JSON files containing
prompt templates and 4 FPS frame indices for training.

Usage:
    python scripts/augment_episodes.py [dataset_root]
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


DATASET_FPS = 30
TARGET_FPS = 4

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

COT_TEMPLATE = "<think>\n</think>"


def compute_4fps_indices(num_frames_30fps: int) -> list[int]:
    """Compute 30 FPS frame indices corresponding to 4 FPS sampling.

    For each 4 FPS index i, the 30 FPS frame is round(i * 30 / 4).
    """
    num_frames_4fps = math.ceil(num_frames_30fps * TARGET_FPS / DATASET_FPS)
    indices = []
    for i in range(num_frames_4fps):
        frame_30fps = round(i * DATASET_FPS / TARGET_FPS)
        frame_30fps = min(frame_30fps, num_frames_30fps - 1)
        indices.append(frame_30fps)
    return indices


def parse_task_description(task_name: str) -> str:
    """Parse semicolon-separated task name into readable description.

    'grabber;picker;black_marker' -> 'Hand off the black marker (grabber to picker)'
    """
    parts = task_name.split(";")
    if len(parts) == 3:
        role1, role2, obj = parts
        obj_readable = obj.replace("_", " ")
        return f"Hand off the {obj_readable} ({role1} to {role2})"
    return task_name.replace(";", " ").replace("_", " ")


def generate_augmentation(dataset_root: str | Path) -> list[dict]:
    """Generate augmentation JSON files for all episodes.

    Args:
        dataset_root: Path to the LeRobot dataset root.

    Returns:
        List of augmentation dicts (one per episode).
    """
    dataset_root = Path(dataset_root)

    # Load task name
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_name = str(tasks_df.index[0])
    task_description = parse_task_description(task_name)

    # Load episode metadata
    episodes_df = pd.read_parquet(
        dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    # Create output directory
    augmentation_dir = dataset_root / "augmentation"
    augmentation_dir.mkdir(exist_ok=True)

    results = []
    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])
        num_frames = int(row["length"])
        frame_indices_4fps = compute_4fps_indices(num_frames)
        num_frames_4fps = len(frame_indices_4fps)

        augmentation = {
            "episode_index": ep_idx,
            "task": task_name,
            "task_description": task_description,
            "num_frames_30fps": num_frames,
            "num_frames_4fps": num_frames_4fps,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
            "chain_of_thought_template": COT_TEMPLATE,
            "frame_indices_4fps": frame_indices_4fps,
        }

        # Validate frame indices
        assert all(
            0 <= idx < num_frames for idx in frame_indices_4fps
        ), f"Episode {ep_idx}: frame indices out of bounds"

        # Write per-episode JSON
        out_path = augmentation_dir / f"episode_{ep_idx:04d}.json"
        with open(out_path, "w") as f:
            json.dump(augmentation, f, indent=2)

        results.append(augmentation)

    return results


def validate_augmentation(dataset_root: str | Path) -> None:
    """Validate generated augmentation JSON files."""
    dataset_root = Path(dataset_root)
    augmentation_dir = dataset_root / "augmentation"

    json_files = sorted(augmentation_dir.glob("episode_*.json"))
    assert len(json_files) > 0, "No augmentation files found"

    episodes_df = pd.read_parquet(
        dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    for json_file in json_files:
        with open(json_file) as f:
            aug = json.load(f)

        ep_idx = aug["episode_index"]
        ep_row = episodes_df[episodes_df["episode_index"] == ep_idx].iloc[0]

        # Check frame count consistency
        assert aug["num_frames_30fps"] == int(ep_row["length"]), (
            f"Episode {ep_idx}: frame count mismatch"
        )

        # Check 4fps frame count
        expected_4fps = math.ceil(aug["num_frames_30fps"] * TARGET_FPS / DATASET_FPS)
        assert aug["num_frames_4fps"] == expected_4fps, (
            f"Episode {ep_idx}: 4fps frame count mismatch"
        )

        # Check frame indices are within bounds
        for idx in aug["frame_indices_4fps"]:
            assert 0 <= idx < aug["num_frames_30fps"], (
                f"Episode {ep_idx}: frame index {idx} out of bounds"
            )

        # Check unified prompt keys (no arm-specific keys)
        assert "system_prompt" in aug
        assert "user_prompt_template" in aug
        assert "chain_of_thought_template" in aug
        assert "{action_history}" in aug["user_prompt_template"]
        assert "{task_description}" in aug["user_prompt_template"]
        assert "left_arm" not in aug, "Found old arm-specific key 'left_arm'"
        assert "right_arm" not in aug, "Found old arm-specific key 'right_arm'"

    print(f"Validated {len(json_files)} augmentation files - all OK")


if __name__ == "__main__":
    import sys

    dataset_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/grabber_picker_black_marker_20260226_211245"
    )

    print(f"Generating augmentation for: {dataset_root}")
    results = generate_augmentation(dataset_root)
    print(f"Generated {len(results)} episode augmentation files")

    for r in results[:3]:
        print(
            f"  Episode {r['episode_index']}: {r['num_frames_30fps']} frames @ 30fps "
            f"-> {r['num_frames_4fps']} frames @ 4fps"
        )

    print()
    print("=== Validation ===")
    validate_augmentation(dataset_root)
