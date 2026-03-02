"""Siamese VLA Dataset for bimanual robot fine-tuning.

Each item yields a TRL-compatible chat message list with:
  - System prompt (arm identity)
  - User message (wrist camera image + action history as tokens)
  - Assistant message (chain-of-thought + 48 action tokens)

Every valid timestep produces 2 samples (left + right arm POV).

Usage:
    from scripts.dataloader import SiameseVLADataset
    dataset = SiameseVLADataset("data/grabber_picker_black_marker_20260228_150311")
    sample = dataset[0]  # returns list of chat message dicts
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lerobot.datasets.video_utils import decode_video_frames

from scripts.tokenize_actions import ActionTokenizer


DATASET_FPS = 30
TARGET_FPS = 4
ACTION_HISTORY_LEN = 4  # 1 second at 4 FPS
ACTION_CHUNK_LEN = 8  # 2 seconds at 4 FPS

VIDEO_KEYS = {
    "left": "observation.images.left.wrist_left",
    "right": "observation.images.right.wrist_right",
}


class SiameseVLADataset(Dataset):
    """Siamese VLA dataset that yields chat-format training samples.

    Each item is one (arm_pov, prompt, action_tokens) tuple formatted as
    TRL-compatible chat messages.

    For E episodes, each with F_i valid frames at 4 FPS:
        Total samples = 2 * sum(F_i for i in episodes)
        where F_i = num_frames_4fps_i - ACTION_CHUNK_LEN
    """

    def __init__(
        self,
        dataset_root: str | Path,
        video_backend: str = "torchcodec",
        tolerance_s: float = 0.05,
    ):
        self.dataset_root = Path(dataset_root)
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s

        # Load action tokenizer
        stats_path = self.dataset_root / "meta" / "stats.json"
        self.action_tokenizer = ActionTokenizer.from_stats_json(stats_path)

        # Load data parquet (all frames)
        data_path = self.dataset_root / "data" / "chunk-000" / "file-000.parquet"
        self.data_df = pd.read_parquet(data_path)

        # Load episode metadata
        episodes_path = (
            self.dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        self.episodes_df = pd.read_parquet(episodes_path)

        # Load augmentation files
        self.augmentations = {}
        aug_dir = self.dataset_root / "augmentation"
        for ep_idx in range(len(self.episodes_df)):
            aug_path = aug_dir / f"episode_{ep_idx:04d}.json"
            with open(aug_path) as f:
                self.augmentations[ep_idx] = json.load(f)

        # Build index table: list of (episode_idx, 4fps_frame_idx, arm)
        self._index_table: list[tuple[int, int, str]] = []
        self._episode_data: dict[int, dict] = {}

        for _, row in self.episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            ep_length = int(row["length"])
            aug = self.augmentations[ep_idx]
            frame_indices_4fps = aug["frame_indices_4fps"]
            num_frames_4fps = len(frame_indices_4fps)

            # Get episode data slice
            ep_start = int(row["dataset_from_index"])
            ep_end = int(row["dataset_to_index"])
            ep_data = self.data_df.iloc[ep_start:ep_end]

            # Pre-extract actions as numpy array for fast access
            actions = np.stack(ep_data["action"].values)  # (ep_length, 12)

            # Store episode info
            self._episode_data[ep_idx] = {
                "actions": actions,
                "length": ep_length,
                "frame_indices_4fps": frame_indices_4fps,
                "num_frames_4fps": num_frames_4fps,
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

            # Valid frame range: need ACTION_CHUNK_LEN future steps
            num_valid = num_frames_4fps - ACTION_CHUNK_LEN
            for t in range(num_valid):
                self._index_table.append((ep_idx, t, "left"))
                self._index_table.append((ep_idx, t, "right"))

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
        # decode_video_frames returns (N, C, H, W) or (N, H, W, C) depending on backend
        frame = frames.squeeze(0)  # Remove batch dim
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

    def _format_action_history(self, action_tokens_per_step: list[list[int]]) -> str:
        """Format action history as text with token names."""
        lines = []
        for i, tokens in enumerate(action_tokens_per_step):
            label = f"t-{ACTION_HISTORY_LEN - i}"
            token_names = self.action_tokenizer.token_ids_to_names(tokens)
            lines.append(f"{label}: {' '.join(token_names)}")
        return "\n".join(lines)

    def __getitem__(self, idx: int) -> list[dict]:
        """Get a single training sample as TRL-compatible chat messages.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        ep_idx, t_4fps, arm = self._index_table[idx]
        ep_data = self._episode_data[ep_idx]
        aug = self.augmentations[ep_idx]
        frame_indices_4fps = ep_data["frame_indices_4fps"]

        # 1. Current image
        current_30fps = frame_indices_4fps[t_4fps]
        current_image = self._load_frame(ep_idx, current_30fps, arm)

        # 2. Action history (last 4 steps at 4 FPS)
        history_4fps_indices = [
            max(0, t_4fps - ACTION_HISTORY_LEN + i) for i in range(ACTION_HISTORY_LEN)
        ]
        history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
        history_actions = self._get_arm_actions(ep_idx, history_30fps, arm)
        history_tokens_per_step = [
            self.action_tokenizer.encode_action(history_actions[i])
            for i in range(ACTION_HISTORY_LEN)
        ]
        action_history_text = self._format_action_history(history_tokens_per_step)

        # 3. Prompts from augmentation
        arm_key = f"{arm}_arm"
        system_prompt = aug[arm_key]["system_prompt"]
        user_prompt = aug[arm_key]["user_prompt_template"].format(
            task_description=aug["task_description"],
            action_history=action_history_text,
        )
        cot = aug[arm_key]["chain_of_thought_template"]

        # 4. Action chunk (8 future timesteps at 4 FPS)
        chunk_4fps_indices = list(range(t_4fps, t_4fps + ACTION_CHUNK_LEN))
        chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
        chunk_actions = self._get_arm_actions(ep_idx, chunk_30fps, arm)
        chunk_token_ids = self.action_tokenizer.encode_action_chunk(chunk_actions)
        chunk_token_names = self.action_tokenizer.token_ids_to_names(chunk_token_ids)

        # 5. Format assistant response: CoT + action tokens
        # Group tokens by timestep for readability (8 groups of 6)
        action_text_parts = []
        for t in range(ACTION_CHUNK_LEN):
            start = t * self.action_tokenizer.NUM_JOINTS
            end = start + self.action_tokenizer.NUM_JOINTS
            action_text_parts.append(" ".join(chunk_token_names[start:end]))
        action_text = " ".join(action_text_parts)
        assistant_text = f"{cot}{action_text}"

        # 6. Build chat messages
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


def validate_dataset(dataset_root: str | Path) -> None:
    """Validate the SiameseVLADataset."""
    dataset_root = Path(dataset_root)
    print("Loading dataset...")
    ds = SiameseVLADataset(dataset_root)

    # Check total size = 2 * sum(valid_frames per episode)
    total_valid = 0
    for ep_idx, ep_data in ds._episode_data.items():
        num_valid = ep_data["num_frames_4fps"] - ACTION_CHUNK_LEN
        total_valid += num_valid
    expected_len = 2 * total_valid
    actual_len = len(ds)
    print(f"Dataset length: {actual_len} (expected: {expected_len})")
    assert actual_len == expected_len, f"Length mismatch: {actual_len} != {expected_len}"
    print("Length check PASSED")

    # Iterate a few samples
    print("\nChecking 5 samples...")
    test_indices = [0, 1, len(ds) // 4, len(ds) // 2, len(ds) - 1]
    for i in test_indices:
        sample = ds[i]
        ep_idx, t_4fps, arm = ds._index_table[i]

        # Check message structure
        assert len(sample) == 3, f"Expected 3 messages, got {len(sample)}"
        assert sample[0]["role"] == "system"
        assert sample[1]["role"] == "user"
        assert sample[2]["role"] == "assistant"

        # Check image in user message
        user_content = sample[1]["content"]
        assert user_content[0]["type"] == "image"
        image = user_content[0]["image"]
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3, f"Expected 3D tensor, got {image.ndim}D"
        assert image.shape[0] == 3, f"Expected CHW format, got shape {image.shape}"
        assert image.min() >= 0 and image.max() <= 1, "Image values out of [0,1] range"

        # Check assistant response has action tokens
        assistant_text = sample[2]["content"][0]["text"]
        assert "<think>" in assistant_text
        assert "</think>" in assistant_text
        # Count action tokens
        import re
        action_tokens = re.findall(r"<action_j\d+_b\d+>", assistant_text)
        assert len(action_tokens) == 48, (
            f"Expected 48 action tokens, got {len(action_tokens)}"
        )

        print(
            f"  Sample {i}: ep={ep_idx}, t={t_4fps}, arm={arm}, "
            f"image={image.shape}, tokens={len(action_tokens)}"
        )

    print("\nAll validation checks PASSED")


if __name__ == "__main__":
    import sys

    dataset_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/grabber_picker_black_marker_20260228_150311"
    )
    validate_dataset(dataset_root)
