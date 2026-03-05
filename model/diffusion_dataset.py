"""Bimanual Diffusion Dataset for cross-arm conditioned action generation.

Each item yields both arms simultaneously with actions at 8 Hz for diffusion
policy training. The VLM messages (or cached embeddings) provide conditioning
signals for the diffusion model.

Usage:
    from model.diffusion_dataset import BimanualDiffusionDataset, build_combined_diffusion_dataset
    dataset = build_combined_diffusion_dataset("data")
    dataset = BimanualDiffusionDataset("data/some_dataset")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from lerobot.datasets.video_utils import decode_video_frames

from scripts.augment_episodes import load_ignore_list
from model.tokenizer import ActionTokenizer
from model.dataset import (
    DATASET_FPS,
    ACTION_HISTORY_LEN,
    ACTION_CHUNK_LEN,
    VIDEO_KEYS,
)

# ---- Constants ----
DIFFUSION_FPS = 8
DIFFUSION_HORIZON = 40   # 5 seconds at 8 Hz
INPAINT_FRAMES = 16      # 2 seconds at 8 Hz
VLM_FPS = 4


def compute_8hz_indices(ep_length_30fps: int) -> list[int]:
    """Compute 30fps frame indices corresponding to 8 Hz sampling.

    Args:
        ep_length_30fps: Total number of frames at 30 fps.

    Returns:
        List of 30fps frame indices at 8 Hz spacing.
    """
    num_frames_8hz = ep_length_30fps * DIFFUSION_FPS // DATASET_FPS
    return [round(i * DATASET_FPS / DIFFUSION_FPS) for i in range(num_frames_8hz)]


def _format_action_history(
    action_tokens_per_step: list[list[int]], action_tokenizer: ActionTokenizer
) -> str:
    """Format action history as text with token names."""
    lines = []
    for i, tokens in enumerate(action_tokens_per_step):
        label = f"t-{ACTION_HISTORY_LEN - i}"
        token_names = action_tokenizer.token_ids_to_names(tokens)
        lines.append(f"{label}: {' '.join(token_names)}")
    return "\n".join(lines)


class BimanualDiffusionDataset(Dataset):
    """Bimanual diffusion dataset yielding both arms with 8 Hz actions.

    Each item provides VLM conditioning messages (or cached embeddings) for
    both arms plus continuous normalized action trajectories for the diffusion
    policy to predict.

    Index table entries are (episode_idx, t_8hz_start) where
    t_8hz_start + DIFFUSION_HORIZON <= num_frames_8hz.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        action_tokenizer: ActionTokenizer | None = None,
        ignore_episodes: set[int] | None = None,
        video_backend: str = "torchcodec",
        tolerance_s: float = 0.05,
        vlm_cache_dir: str | Path | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_name = self.dataset_root.name
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self.vlm_cache_dir = Path(vlm_cache_dir) if vlm_cache_dir is not None else None

        # Load action tokenizer
        if action_tokenizer is not None:
            self.action_tokenizer = action_tokenizer
        else:
            stats_path = self.dataset_root / "meta" / "stats.json"
            self.action_tokenizer = ActionTokenizer.from_stats_json(stats_path)

        # Load ignore list if not provided
        if ignore_episodes is None:
            ignore_episodes = load_ignore_list(self.dataset_root)

        # Load data parquet (all chunk files)
        data_files = sorted(
            (self.dataset_root / "data").glob("chunk-*/*.parquet")
        )
        self.data_df = pd.concat(
            [pd.read_parquet(f) for f in data_files], ignore_index=True
        )

        # Load episode metadata (all chunk files)
        episode_files = sorted(
            (self.dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
        )
        self.episodes_df = pd.concat(
            [pd.read_parquet(f) for f in episode_files], ignore_index=True
        )

        # Load augmentation files (only for non-ignored episodes)
        self.augmentations = {}
        aug_dir = self.dataset_root / "augmentation"
        for _, row in self.episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx in ignore_episodes:
                continue
            aug_path = aug_dir / f"episode_{ep_idx:04d}.json"
            with open(aug_path) as f:
                self.augmentations[ep_idx] = json.load(f)

        # Build index table and episode data
        self._index_table: list[tuple[int, int]] = []
        self._episode_data: dict[int, dict] = {}

        for _, row in self.episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx in ignore_episodes:
                continue

            ep_length = int(row["length"])
            aug = self.augmentations[ep_idx]
            frame_indices_4fps = aug["frame_indices_4fps"]

            # Get episode data slice
            ep_start = int(row["dataset_from_index"])
            ep_end = int(row["dataset_to_index"])
            ep_data = self.data_df.iloc[ep_start:ep_end]

            # Pre-extract actions as numpy array
            actions = np.stack(ep_data["action"].values)  # (ep_length, 12)

            # Compute 8 Hz indices for this episode
            indices_8hz = compute_8hz_indices(ep_length)

            # Store episode info
            self._episode_data[ep_idx] = {
                "actions": actions,
                "length": ep_length,
                "frame_indices_4fps": frame_indices_4fps,
                "num_frames_4fps": len(frame_indices_4fps),
                "indices_8hz": indices_8hz,
                "num_frames_8hz": len(indices_8hz),
                # Video metadata
                "left_video_path": self._get_video_path(row, VIDEO_KEYS["left"]),
                "left_from_ts": float(
                    row[f"videos/{VIDEO_KEYS['left']}/from_timestamp"]
                ),
                "right_video_path": self._get_video_path(row, VIDEO_KEYS["right"]),
                "right_from_ts": float(
                    row[f"videos/{VIDEO_KEYS['right']}/from_timestamp"]
                ),
            }

            # Valid 8 Hz start indices: need DIFFUSION_HORIZON frames ahead
            num_frames_8hz = len(indices_8hz)
            for t_start in range(num_frames_8hz - DIFFUSION_HORIZON + 1):
                self._index_table.append((ep_idx, t_start))

    def _get_video_path(self, episode_row: pd.Series, video_key: str) -> Path:
        """Construct video file path from episode metadata."""
        chunk_idx = int(episode_row[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode_row[f"videos/{video_key}/file_index"])
        return (
            self.dataset_root
            / "videos"
            / video_key
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )

    def __len__(self) -> int:
        return len(self._index_table)

    def _load_frame(
        self, ep_idx: int, frame_30fps: int, arm: str
    ) -> torch.Tensor:
        """Load a single video frame.

        Returns:
            Tensor of shape (C, H, W) in [0, 1].
        """
        ep_data = self._episode_data[ep_idx]
        video_key = "left" if arm == "left" else "right"
        video_path = ep_data[f"{video_key}_video_path"]
        from_ts = ep_data[f"{video_key}_from_ts"]

        # Compute absolute timestamp within the MP4
        frame_ts = frame_30fps / DATASET_FPS
        absolute_ts = from_ts + frame_ts

        frames = decode_video_frames(
            video_path,
            [absolute_ts],
            self.tolerance_s,
            self.video_backend,
        )
        frame = frames.squeeze(0)
        # Ensure CHW format
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)
        return frame

    def _get_arm_actions(
        self, ep_idx: int, frame_30fps_indices: list[int], arm: str
    ) -> np.ndarray:
        """Get action values for a specific arm at given 30fps frame indices.

        Returns:
            (T, 6) array of joint values.
        """
        ep_data = self._episode_data[ep_idx]
        actions = ep_data["actions"]  # (ep_length, 12)
        arm_slice = slice(0, 6) if arm == "left" else slice(6, 12)
        return actions[frame_30fps_indices, arm_slice]

    def _normalize_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """Normalize raw joint actions to [-1, 1].

        Args:
            raw_actions: (T, 6) array of raw joint values.

        Returns:
            (T, 6) array of normalized actions in [-1, 1].
        """
        mins = self.action_tokenizer.joint_mins  # (6,)
        maxs = self.action_tokenizer.joint_maxs  # (6,)
        normalized = 2.0 * (raw_actions - mins) / (maxs - mins) - 1.0
        return normalized.astype(np.float32)

    def _build_vlm_messages(
        self, ep_idx: int, t_8hz_start: int, arm: str
    ) -> list[dict]:
        """Build TRL-compatible chat messages for one arm's VLM forward pass.

        The VLM image is at the prediction boundary (end of the GT/inpaint window).
        The action history covers the second half of the GT window (seconds 1-2).

        Args:
            ep_idx: Episode index.
            t_8hz_start: Starting 8 Hz frame index for this sample.
            arm: "left" or "right".

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        ep_data = self._episode_data[ep_idx]
        aug = self.augmentations[ep_idx]
        indices_8hz = ep_data["indices_8hz"]
        frame_indices_4fps = ep_data["frame_indices_4fps"]

        # The VLM image frame is at the end of the GT/inpaint window
        # t_8hz_start + 15 is the last frame of the 2-sec GT region
        t_8hz_image = t_8hz_start + INPAINT_FRAMES - 1
        t_4hz_image = round(t_8hz_image * VLM_FPS / DIFFUSION_FPS)

        # Clamp t_4hz_image to valid range
        num_frames_4fps = ep_data["num_frames_4fps"]
        t_4hz_image = min(t_4hz_image, num_frames_4fps - 1)

        # Load the image at the 4 Hz frame corresponding to prediction boundary
        frame_30fps_for_image = frame_indices_4fps[t_4hz_image]
        current_image = self._load_frame(ep_idx, frame_30fps_for_image, arm)

        # Action history: 4 frames at 4 Hz ending at t_4hz_image
        # These cover the second half of the GT window (seconds 1-2)
        history_4fps_indices = [
            max(0, t_4hz_image - ACTION_HISTORY_LEN + 1 + i)
            for i in range(ACTION_HISTORY_LEN)
        ]
        history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
        history_actions = self._get_arm_actions(ep_idx, history_30fps, arm)
        history_tokens_per_step = [
            self.action_tokenizer.encode_action(history_actions[i])
            for i in range(ACTION_HISTORY_LEN)
        ]
        action_history_text = _format_action_history(
            history_tokens_per_step, self.action_tokenizer
        )

        # Prompts from augmentation
        arm_key = f"{arm}_arm"
        system_prompt = aug[arm_key]["system_prompt"]
        user_prompt = aug[arm_key]["user_prompt_template"].format(
            task_description=aug[arm_key]["task_description"],
            action_history=action_history_text,
        )

        # Chain-of-thought label at the 4 Hz index
        # Clamp to valid range for the CoT list
        t_4fps_for_cot = min(t_4hz_image, len(aug[arm_key]["chain_of_thought"]) - 1)
        cot = aug[arm_key]["chain_of_thought"][t_4fps_for_cot]

        # Action chunk for assistant response (ACTION_CHUNK_LEN steps at 4 Hz)
        chunk_end = min(t_4hz_image + ACTION_CHUNK_LEN, num_frames_4fps)
        chunk_4fps_indices = list(range(t_4hz_image, chunk_end))
        # Pad if we don't have enough frames by repeating the last
        while len(chunk_4fps_indices) < ACTION_CHUNK_LEN:
            chunk_4fps_indices.append(chunk_4fps_indices[-1])
        chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
        chunk_actions = self._get_arm_actions(ep_idx, chunk_30fps, arm)
        chunk_token_ids = self.action_tokenizer.encode_action_chunk(chunk_actions)
        chunk_token_names = self.action_tokenizer.token_ids_to_names(chunk_token_ids)

        # Format assistant response: CoT + action tokens
        action_text_parts = []
        for t in range(ACTION_CHUNK_LEN):
            start = t * self.action_tokenizer.NUM_JOINTS
            end = start + self.action_tokenizer.NUM_JOINTS
            action_text_parts.append(" ".join(chunk_token_names[start:end]))
        action_text = " ".join(action_text_parts)
        assistant_text = f"{cot}{action_text}"

        # Build chat messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": current_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        return messages

    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample with both arms.

        Returns:
            Dict with keys depending on mode:
              - Normal mode: left_vlm_messages, right_vlm_messages,
                left_actions_8hz, right_actions_8hz, inpaint_mask,
                episode_idx, t_8hz_start
              - Cached mode: left_vlm_emb, right_vlm_emb (instead of messages),
                plus the action/mask/index keys
        """
        ep_idx, t_8hz_start = self._index_table[idx]
        ep_data = self._episode_data[ep_idx]
        indices_8hz = ep_data["indices_8hz"]

        # Extract 8 Hz frame indices for the diffusion horizon
        horizon_8hz_indices = list(
            range(t_8hz_start, t_8hz_start + DIFFUSION_HORIZON)
        )
        # Map to 30fps indices
        horizon_30fps = [indices_8hz[i] for i in horizon_8hz_indices]

        # Get actions for both arms at 8 Hz
        left_actions_raw = self._get_arm_actions(ep_idx, horizon_30fps, "left")
        right_actions_raw = self._get_arm_actions(ep_idx, horizon_30fps, "right")

        # Normalize to [-1, 1]
        left_actions_8hz = self._normalize_actions(left_actions_raw)
        right_actions_8hz = self._normalize_actions(right_actions_raw)

        # Inpaint mask: True for first INPAINT_FRAMES frames (the GT region)
        inpaint_mask = np.zeros(DIFFUSION_HORIZON, dtype=bool)
        inpaint_mask[:INPAINT_FRAMES] = True

        result = {
            "left_actions_8hz": left_actions_8hz,     # (40, 6)
            "right_actions_8hz": right_actions_8hz,   # (40, 6)
            "inpaint_mask": inpaint_mask,             # (40,)
            "episode_idx": ep_idx,
            "t_8hz_start": t_8hz_start,
        }

        # VLM conditioning: cached embeddings or full messages
        if self.vlm_cache_dir is not None:
            cache_path = (
                self.vlm_cache_dir
                / f"{self.dataset_name}_ep{ep_idx:04d}_t{t_8hz_start:04d}.pt"
            )
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            result["left_vlm_emb"] = cached["left_embedding"]    # (2048,)
            result["right_vlm_emb"] = cached["right_embedding"]  # (2048,)
        else:
            result["left_vlm_messages"] = self._build_vlm_messages(
                ep_idx, t_8hz_start, "left"
            )
            result["right_vlm_messages"] = self._build_vlm_messages(
                ep_idx, t_8hz_start, "right"
            )

        return result


def build_combined_diffusion_dataset(
    data_dir: str | Path,
    video_backend: str = "torchcodec",
    tolerance_s: float = 0.05,
    vlm_cache_dir: str | Path | None = None,
) -> ConcatDataset:
    """Build a ConcatDataset of BimanualDiffusionDatasets from all datasets under data_dir.

    Discovers all subdirectories with meta/info.json, loads their ignore lists,
    builds a global ActionTokenizer from all stats.json files, and creates one
    BimanualDiffusionDataset per directory.

    Args:
        data_dir: Root directory containing dataset subdirectories.
        video_backend: Video decoding backend.
        tolerance_s: Timestamp tolerance for video frame decoding.
        vlm_cache_dir: If set, datasets load cached VLM embeddings from this dir.

    Returns:
        A ConcatDataset combining all individual datasets.
    """
    data_dir = Path(data_dir)
    dataset_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").exists()
    )

    if not dataset_dirs:
        raise FileNotFoundError(f"No datasets found under {data_dir}")

    # Build global action tokenizer from all stats files
    stats_paths = [d / "meta" / "stats.json" for d in dataset_dirs]
    global_tokenizer = ActionTokenizer.from_multiple_stats(stats_paths)

    # Build individual datasets
    datasets = []
    total_episodes = 0
    total_ignored = 0
    for d in dataset_dirs:
        ignore_eps = load_ignore_list(d)
        ds = BimanualDiffusionDataset(
            d,
            action_tokenizer=global_tokenizer,
            ignore_episodes=ignore_eps,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
            vlm_cache_dir=vlm_cache_dir,
        )
        num_eps = len(ds._episode_data)
        total_episodes += num_eps
        total_ignored += len(ignore_eps)
        print(f"  {d.name}: {num_eps} episodes, {len(ds)} samples")
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    print(
        f"\nCombined: {total_episodes} episodes ({total_ignored} ignored), "
        f"{len(combined)} total samples"
    )
    return combined


if __name__ == "__main__":
    import sys

    dataset_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/grabber_picker_black_marker_20260228_150311"
    )

    print(f"=== BimanualDiffusionDataset: {dataset_root.name} ===\n")

    # --- Verify 8 Hz index computation ---
    print("--- 8 Hz Index Verification ---")
    test_lengths = [300, 600, 900, 1500]
    for length in test_lengths:
        indices = compute_8hz_indices(length)
        print(
            f"  ep_length_30fps={length:>5d} -> "
            f"num_frames_8hz={len(indices):>4d}, "
            f"first_5={indices[:5]}, "
            f"last_5={indices[-5:]}"
        )
    print()

    # --- Load dataset ---
    print("--- Loading Dataset ---")
    ds = BimanualDiffusionDataset(dataset_root)
    print(f"  Episodes: {len(ds._episode_data)}")
    print(f"  Total samples: {len(ds)}")
    print()

    # --- Print episode info ---
    print("--- Episode Info ---")
    for ep_idx, ep_data in ds._episode_data.items():
        print(
            f"  Episode {ep_idx}: "
            f"length_30fps={ep_data['length']}, "
            f"num_frames_8hz={ep_data['num_frames_8hz']}, "
            f"num_frames_4fps={ep_data['num_frames_4fps']}"
        )
    print()

    # --- Check a few samples ---
    print("--- Sample Inspection ---")
    num_to_check = min(5, len(ds))
    test_indices = [0, len(ds) // 4, len(ds) // 2, 3 * len(ds) // 4, len(ds) - 1]
    test_indices = test_indices[:num_to_check]

    for i in test_indices:
        sample = ds[i]
        ep_idx = sample["episode_idx"]
        t_start = sample["t_8hz_start"]

        print(f"\n  Sample {i}: episode={ep_idx}, t_8hz_start={t_start}")
        print(f"    left_actions_8hz shape:  {sample['left_actions_8hz'].shape}")
        print(f"    right_actions_8hz shape: {sample['right_actions_8hz'].shape}")
        print(f"    left_actions_8hz range:  [{sample['left_actions_8hz'].min():.3f}, {sample['left_actions_8hz'].max():.3f}]")
        print(f"    right_actions_8hz range: [{sample['right_actions_8hz'].min():.3f}, {sample['right_actions_8hz'].max():.3f}]")
        print(f"    inpaint_mask shape:      {sample['inpaint_mask'].shape}")
        print(f"    inpaint_mask sum:        {sample['inpaint_mask'].sum()} (expect {INPAINT_FRAMES})")
        print(f"    inpaint_mask[:18]:       {sample['inpaint_mask'][:18]}")

        # Check VLM messages structure
        if "left_vlm_messages" in sample:
            left_msgs = sample["left_vlm_messages"]
            right_msgs = sample["right_vlm_messages"]
            print(f"    left_vlm_messages:  {len(left_msgs)} messages, roles={[m['role'] for m in left_msgs]}")
            print(f"    right_vlm_messages: {len(right_msgs)} messages, roles={[m['role'] for m in right_msgs]}")

            # Check image in user message
            left_user = left_msgs[1]
            image_entry = left_user["content"][0]
            assert image_entry["type"] == "image", f"Expected image, got {image_entry['type']}"
            image = image_entry["image"]
            print(f"    left image shape:   {image.shape}")

            # Check assistant has CoT
            left_assistant = left_msgs[2]["content"][0]["text"]
            has_think = "<think>" in left_assistant and "</think>" in left_assistant
            print(f"    left assistant has CoT: {has_think}")
        elif "left_vlm_emb" in sample:
            print(f"    left_vlm_emb shape:  {sample['left_vlm_emb'].shape}")
            print(f"    right_vlm_emb shape: {sample['right_vlm_emb'].shape}")

    print("\n--- Inpaint Mask Verification ---")
    sample = ds[0]
    mask = sample["inpaint_mask"]
    assert mask.shape == (DIFFUSION_HORIZON,), f"Expected ({DIFFUSION_HORIZON},), got {mask.shape}"
    assert mask[:INPAINT_FRAMES].all(), "First INPAINT_FRAMES should be True"
    assert not mask[INPAINT_FRAMES:].any(), "Remaining frames should be False"
    print(f"  Mask shape: {mask.shape}")
    print(f"  GT frames (True):   {mask.sum()} (frames 0-{INPAINT_FRAMES - 1})")
    print(f"  Pred frames (False): {(~mask).sum()} (frames {INPAINT_FRAMES}-{DIFFUSION_HORIZON - 1})")
    print("  Inpaint mask verification PASSED")

    print("\n=== All checks passed ===")
