"""Episode augmentation script for Siamese VLA fine-tuning.

Reads episode metadata and generates per-episode JSON files containing
prompt templates and 4 FPS frame indices for training.

Usage:
    python scripts/augment_episodes.py                    # all datasets under data/
    python scripts/augment_episodes.py path/to/dataset    # single dataset
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


DATASET_FPS = 30
TARGET_FPS = 4

SYSTEM_PROMPT_LEFT = (
    ""
)

SYSTEM_PROMPT_RIGHT = (
    ""
)

USER_PROMPT_TEMPLATE_LEFT = (
    "You are the LEFT arm. The task is: {task_description}. "
    "Here is your current wrist camera image and your last 4 actions:\n"
    "{action_history}\n"
    "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."
)

USER_PROMPT_TEMPLATE_RIGHT = (
    "You are the RIGHT arm. The task is: {task_description}. "
    "Here is your current wrist camera image and your last 4 actions:\n"
    "{action_history}\n"
    "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."
)

COT_TEMPLATE = "<think>\n</think>"


def load_ignore_list(dataset_root: str | Path) -> set[int]:
    """Load episode ignore list from dataset_root/ignore.txt.

    Returns an empty set if the file is missing or empty.
    """
    ignore_path = Path(dataset_root) / "ignore.txt"
    if not ignore_path.exists():
        return set()
    text = ignore_path.read_text().strip()
    if not text:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def compute_4fps_indices(num_frames_30fps: int) -> list[int]:
    """Compute 30 FPS frame indices corresponding to 4 FPS sampling.

    For each 4 FPS index i, the 30 FPS frame is round(i * 30 / 4).
    """
    num_frames_4fps = math.ceil(num_frames_30fps * TARGET_FPS / DATASET_FPS)
    indices = []
    for i in range(num_frames_4fps):
        frame_30fps = round(i * DATASET_FPS / TARGET_FPS)
        # Clamp to valid range
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
    """Generate augmentation JSON files for non-ignored episodes.

    Args:
        dataset_root: Path to the LeRobot dataset root.

    Returns:
        List of augmentation dicts (one per included episode).
    """
    dataset_root = Path(dataset_root)

    # Load ignore list
    ignore_episodes = load_ignore_list(dataset_root)

    # Load task name (task name is the DataFrame index, not a column)
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_name = str(tasks_df.index[0])
    task_description = parse_task_description(task_name)

    # Load episode metadata (concatenate all chunk files)
    episode_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    episodes_df = pd.concat(
        [pd.read_parquet(f) for f in episode_files], ignore_index=True
    )

    # Create output directory
    augmentation_dir = dataset_root / "augmentation"
    augmentation_dir.mkdir(exist_ok=True)

    results = []
    skipped = []
    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])

        if ep_idx in ignore_episodes:
            skipped.append(ep_idx)
            continue

        num_frames = int(row["length"])
        frame_indices_4fps = compute_4fps_indices(num_frames)
        num_frames_4fps = len(frame_indices_4fps)

        augmentation = {
            "episode_index": ep_idx,
            "task": task_name,
            "task_description": task_description,
            "num_frames_30fps": num_frames,
            "num_frames_4fps": num_frames_4fps,
            "left_arm": {
                "system_prompt": SYSTEM_PROMPT_LEFT,
                "user_prompt_template": USER_PROMPT_TEMPLATE_LEFT,
                "chain_of_thought_template": COT_TEMPLATE,
            },
            "right_arm": {
                "system_prompt": SYSTEM_PROMPT_RIGHT,
                "user_prompt_template": USER_PROMPT_TEMPLATE_RIGHT,
                "chain_of_thought_template": COT_TEMPLATE,
            },
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

    if skipped:
        print(f"  Skipped {len(skipped)} ignored episodes: {skipped}")

    return results


def validate_augmentation(dataset_root: str | Path) -> None:
    """Validate generated augmentation JSON files."""
    dataset_root = Path(dataset_root)
    augmentation_dir = dataset_root / "augmentation"

    json_files = sorted(augmentation_dir.glob("episode_*.json"))
    assert len(json_files) > 0, "No augmentation files found"

    # Load episode metadata (concatenate all chunk files)
    episode_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    episodes_df = pd.concat(
        [pd.read_parquet(f) for f in episode_files], ignore_index=True
    )

    ignore_episodes = load_ignore_list(dataset_root)

    for json_file in json_files:
        with open(json_file) as f:
            aug = json.load(f)

        ep_idx = aug["episode_index"]

        # Ensure ignored episodes don't have augmentation files
        assert ep_idx not in ignore_episodes, (
            f"Episode {ep_idx} is in ignore list but has augmentation file"
        )

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

        # Check required keys
        for arm in ["left_arm", "right_arm"]:
            assert "system_prompt" in aug[arm]
            assert "user_prompt_template" in aug[arm]
            assert "chain_of_thought_template" in aug[arm]
            assert "{action_history}" in aug[arm]["user_prompt_template"]
            assert "{task_description}" in aug[arm]["user_prompt_template"]

    print(f"  Validated {len(json_files)} augmentation files - all OK")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single dataset mode
        roots = [Path(sys.argv[1])]
    else:
        # All datasets under data/
        data_dir = Path("data")
        roots = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and (p / "meta" / "info.json").exists()
        )

    for dataset_root in roots:
        print(f"\n=== {dataset_root} ===")
        results = generate_augmentation(dataset_root)
        print(f"  Generated {len(results)} episode augmentation files")

        for r in results[:3]:
            print(f"    Episode {r['episode_index']}: {r['num_frames_30fps']} frames @ 30fps "
                  f"-> {r['num_frames_4fps']} frames @ 4fps")

        print("  Validation:")
        validate_augmentation(dataset_root)
