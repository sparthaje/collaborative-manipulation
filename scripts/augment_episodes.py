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

import numpy as np
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
    "You are the LEFT arm. {task_description}. "
    "Here is your current wrist camera image and your last 4 actions:\n"
    "{action_history}\n"
    "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."
)

USER_PROMPT_TEMPLATE_RIGHT = (
    "You are the RIGHT arm. {task_description}. "
    "Here is your current wrist camera image and your last 4 actions:\n"
    "{action_history}\n"
    "Output the next 8 joint actions (2 seconds at 4 FPS) as 48 tokens."
)

PICKER_STAGES = ["picking", "reaching", "releasing", "disengaging", "watching"]
GRASPER_STAGES = ["watching", "reaching", "holding", "disengaging", "dropping"]
ALL_STAGE_NAMES = set(PICKER_STAGES) | set(GRASPER_STAGES)

VELOCITY_THRESHOLD = 5.0  # deg/s — from DATA.md
GRIPPER_THRESHOLD = 7.0  # degrees — from DATA.md
GRIPPER_STABILITY_FRAMES = 60  # 2 seconds at 30 FPS
VELOCITY_WINDOW = 15  # half second at 30 FPS
WATCHING_SETTLE_FRAMES = 30  # ~1 second at 30 FPS
MIN_INTERMEDIATE_FRAMES = 4  # minimum 4fps frames of holding/releasing


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


def compute_arm_velocity(positions_30fps: np.ndarray) -> np.ndarray:
    """Compute smoothed L2 velocity for joints 0–4 (excluding gripper) at each frame.

    Args:
        positions_30fps: (N, 6) array — 6 joints for one arm at 30 FPS.

    Returns:
        (N,) array of smoothed L2 velocity in deg/s.
    """
    # Frame-to-frame velocity on joints 0-4 only
    vel = np.diff(positions_30fps[:, :5], axis=0) * DATASET_FPS  # (N-1, 5) deg/s
    # Pad first frame by repeating
    vel = np.vstack([vel[:1], vel])  # (N, 5)
    # L2 norm across joints
    l2 = np.linalg.norm(vel, axis=1)  # (N,)
    # Centered rolling average
    kernel = np.ones(VELOCITY_WINDOW) / VELOCITY_WINDOW
    return np.convolve(l2, kernel, mode="same")


def gripper_is_stable(
    gripper_values: np.ndarray, frame_idx: int, target_state: str
) -> bool:
    """Check if gripper stays in target state for GRIPPER_STABILITY_FRAMES.

    Args:
        gripper_values: (N,) array of gripper angles in degrees.
        frame_idx: current 30 FPS frame index.
        target_state: 'open' or 'closed'.

    Returns:
        True if all frames in the look-ahead window satisfy the condition.
    """
    end = min(frame_idx + GRIPPER_STABILITY_FRAMES, len(gripper_values))
    segment = gripper_values[frame_idx:end]
    if len(segment) == 0:
        return False
    if target_state == "closed":
        return bool(np.all(segment <= GRIPPER_THRESHOLD))
    return bool(np.all(segment > GRIPPER_THRESHOLD))


def classify_stages_joint(
    picker_actions_30fps: np.ndarray,
    grabber_actions_30fps: np.ndarray,
) -> tuple[list[str], list[str]]:
    """Classify per-frame stage labels for both arms using a joint state machine.

    Both arms are processed simultaneously so that if one arm reaches
    disengaging, the other arm is forced to disengaging on the next frame.

    Args:
        picker_actions_30fps: (N, 6) array for the picker arm at 30 FPS.
        grabber_actions_30fps: (N, 6) array for the grabber arm at 30 FPS.

    Returns:
        (picker_stages, grabber_stages): Lists of N stage label strings each.
    """
    N = len(picker_actions_30fps)

    p_velocity = compute_arm_velocity(picker_actions_30fps)
    p_gripper = picker_actions_30fps[:, 5]
    g_velocity = compute_arm_velocity(grabber_actions_30fps)
    g_gripper = grabber_actions_30fps[:, 5]

    p_stage = 0  # index into PICKER_STAGES
    g_stage = 0  # index into GRASPER_STAGES
    picker_seen_open = False
    grabber_seen_open = False
    holding_was_still = False

    picker_result: list[str] = []
    grabber_result: list[str] = []

    for i in range(N):
        # --- Failsafe: if one arm is disengaging (>=3), force the other forward ---
        if p_stage >= 3 and g_stage < 3:
            g_stage = 3
        elif g_stage >= 3 and p_stage < 3:
            p_stage = 3

        # --- Picker state machine ---
        if p_stage == 0:  # picking → reaching
            if p_gripper[i] > GRIPPER_THRESHOLD:
                picker_seen_open = True
            if picker_seen_open and p_gripper[i] <= GRIPPER_THRESHOLD and gripper_is_stable(
                p_gripper, i, "closed"
            ):
                p_stage = 1
        elif p_stage == 1:  # reaching → releasing
            if p_gripper[i] > GRIPPER_THRESHOLD and gripper_is_stable(
                p_gripper, i, "open"
            ):
                p_stage = 2
        elif p_stage == 2:  # releasing → disengaging
            if p_velocity[i] > VELOCITY_THRESHOLD:
                p_stage = 3
        elif p_stage == 3:  # disengaging → watching
            if p_velocity[i] < VELOCITY_THRESHOLD:
                remaining = p_velocity[i:]
                check = min(WATCHING_SETTLE_FRAMES, len(remaining))
                if np.all(remaining[:check] < VELOCITY_THRESHOLD):
                    p_stage = 4

        # --- Grabber state machine ---
        if g_stage == 0:  # watching → reaching
            picker_advanced = p_stage >= 1
            if picker_advanced and g_velocity[i] > VELOCITY_THRESHOLD:
                g_stage = 1
        elif g_stage == 1:  # reaching → holding
            if g_gripper[i] > GRIPPER_THRESHOLD:
                grabber_seen_open = True
            if grabber_seen_open and g_gripper[i] <= GRIPPER_THRESHOLD and gripper_is_stable(
                g_gripper, i, "closed"
            ):
                g_stage = 2
                holding_was_still = False
        elif g_stage == 2:  # holding → disengaging
            if g_velocity[i] < VELOCITY_THRESHOLD:
                holding_was_still = True
            elif holding_was_still and g_velocity[i] > VELOCITY_THRESHOLD:
                g_stage = 3
        elif g_stage == 3:  # disengaging → dropping
            if g_gripper[i] > GRIPPER_THRESHOLD and gripper_is_stable(
                g_gripper, i, "open"
            ):
                g_stage = 4

        picker_result.append(PICKER_STAGES[p_stage])
        grabber_result.append(GRASPER_STAGES[g_stage])

    return picker_result, grabber_result


def ensure_min_intermediate_frames(
    stages_4fps: list[str],
    intermediate_label: str,
    min_frames: int = MIN_INTERMEDIATE_FRAMES,
) -> list[str]:
    """Ensure at least *min_frames* of the intermediate stage between reaching and disengaging.

    At 4 FPS the holding/releasing stage can be very short or absent entirely.
    This backfills preceding "reaching" frames with *intermediate_label* so
    there are at least *min_frames* of that stage in the 4 FPS sequence.
    """
    result = list(stages_4fps)

    count = result.count(intermediate_label)
    if count >= min_frames:
        return result

    # Find the first frame that is either the intermediate or disengaging
    # — this marks the transition boundary after reaching.
    first_transition = None
    for i, s in enumerate(result):
        if s == intermediate_label or s == "disengaging":
            first_transition = i
            break

    if first_transition is None:
        return result  # arm never progressed past reaching

    needed = min_frames - count
    for i in range(first_transition - 1, -1, -1):
        if needed <= 0:
            break
        if result[i] == "reaching":
            result[i] = intermediate_label
            needed -= 1

    return result


def build_per_frame_cot(
    stages_30fps: list[str], frame_indices_4fps: list[int]
) -> list[str]:
    """Sample 30 FPS stage labels at 4 FPS indices and wrap as CoT strings."""
    return [f"<think>{stages_30fps[idx]}</think>" for idx in frame_indices_4fps]


def parse_task_description(task_name: str) -> tuple[str, str]:
    """Parse semicolon-separated task name into per-arm descriptions.

    Role mapping: role1 = left arm, role2 = right arm.
    'grabber' = accepts the handoff, 'picker' = picks up and hands off.

    Returns (left_description, right_description).
    """
    parts = task_name.split(";")
    if len(parts) == 3:
        left_role, right_role, obj = parts
        obj_readable = obj.replace("_", " ")

        def _arm_description(role: str, other_side: str) -> str:
            if role == "grabber":
                return f"Accept the handoff of the {obj_readable} from the {other_side} arm"
            elif role == "picker":
                return f"Pick up the {obj_readable} and hand it to the {other_side} arm"
            return f"{role} the {obj_readable}"

        left_desc = _arm_description(left_role, "right")
        right_desc = _arm_description(right_role, "left")
        return left_desc, right_desc

    fallback = task_name.replace(";", " ").replace("_", " ")
    return fallback, fallback


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
    left_task_description, right_task_description = parse_task_description(task_name)

    # Determine arm roles from task name (format: "role_left;role_right;object")
    parts = task_name.split(";")
    left_role = parts[0] if len(parts) >= 2 else "picker"
    right_role = parts[1] if len(parts) >= 2 else "grabber"

    # Load episode metadata (concatenate all chunk files)
    episode_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    episodes_df = pd.concat(
        [pd.read_parquet(f) for f in episode_files], ignore_index=True
    )

    # Load data parquet (all chunks) for action data
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    data_df = pd.concat(
        [pd.read_parquet(f) for f in data_files], ignore_index=True
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

        # Extract per-episode action data
        ep_start = int(row["dataset_from_index"])
        ep_end = int(row["dataset_to_index"])
        actions = np.stack(data_df.iloc[ep_start:ep_end]["action"].values)  # (N, 12)
        left_actions = actions[:, :6]   # joints 0-5
        right_actions = actions[:, 6:]  # joints 6-11

        # Classify stages jointly (both arms in lockstep)
        if left_role == "picker":
            left_stages, right_stages = classify_stages_joint(
                left_actions, right_actions
            )
        else:
            right_stages, left_stages = classify_stages_joint(
                right_actions, left_actions
            )

        # Sample stages at 4 FPS and ensure minimum intermediate frames
        left_stages_4fps = [left_stages[idx] for idx in frame_indices_4fps]
        right_stages_4fps = [right_stages[idx] for idx in frame_indices_4fps]

        left_intermediate = "releasing" if left_role == "picker" else "holding"
        right_intermediate = "holding" if left_role == "picker" else "releasing"

        left_stages_4fps = ensure_min_intermediate_frames(left_stages_4fps, left_intermediate)
        right_stages_4fps = ensure_min_intermediate_frames(right_stages_4fps, right_intermediate)

        left_cot = [f"<think>{s}</think>" for s in left_stages_4fps]
        right_cot = [f"<think>{s}</think>" for s in right_stages_4fps]

        augmentation = {
            "episode_index": ep_idx,
            "task": task_name,
            "num_frames_30fps": num_frames,
            "num_frames_4fps": num_frames_4fps,
            "left_arm": {
                "role": left_role,
                "task_description": left_task_description,
                "system_prompt": SYSTEM_PROMPT_LEFT,
                "user_prompt_template": USER_PROMPT_TEMPLATE_LEFT,
                "chain_of_thought": left_cot,
            },
            "right_arm": {
                "role": right_role,
                "task_description": right_task_description,
                "system_prompt": SYSTEM_PROMPT_RIGHT,
                "user_prompt_template": USER_PROMPT_TEMPLATE_RIGHT,
                "chain_of_thought": right_cot,
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

        # Check required keys and per-frame CoT
        for arm in ["left_arm", "right_arm"]:
            assert "task_description" in aug[arm]
            assert "system_prompt" in aug[arm]
            assert "user_prompt_template" in aug[arm]
            assert "chain_of_thought" in aug[arm]
            assert "{action_history}" in aug[arm]["user_prompt_template"]
            assert "{task_description}" in aug[arm]["user_prompt_template"]

            # Validate per-frame CoT array
            cot_list = aug[arm]["chain_of_thought"]
            assert isinstance(cot_list, list), (
                f"Episode {ep_idx} {arm}: chain_of_thought must be a list"
            )
            assert len(cot_list) == aug["num_frames_4fps"], (
                f"Episode {ep_idx} {arm}: chain_of_thought length "
                f"{len(cot_list)} != num_frames_4fps {aug['num_frames_4fps']}"
            )
            for t, cot in enumerate(cot_list):
                assert cot.startswith("<think>") and cot.endswith("</think>"), (
                    f"Episode {ep_idx} {arm} frame {t}: bad CoT format: {cot}"
                )
                stage = cot[len("<think>"):-len("</think>")]
                assert stage in ALL_STAGE_NAMES, (
                    f"Episode {ep_idx} {arm} frame {t}: unknown stage '{stage}'"
                )

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
