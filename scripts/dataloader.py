"""Unified Bimanual VLA Dataset for fine-tuning.

Each item yields a TRL-compatible chat message list with:
  - System prompt (unified, controls both arms)
  - User message (3 camera images + 12-joint action history as tokens)
  - Assistant message (chain-of-thought + 96 action tokens)

Each valid timestep produces 1 sample (both arms together).

Usage:
    from scripts.dataloader import BimanualVLADataset
    dataset = BimanualVLADataset("data/grabber_picker_black_marker_20260226_211245")
    sample = dataset[0]  # returns list of chat message dicts
"""

from __future__ import annotations

import json
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
NUM_JOINTS_PER_ARM = 6
NUM_JOINTS_TOTAL = 12

VIDEO_KEYS = {
    "left_wrist": "observation.images.left.wrist_left",
    "right_wrist": "observation.images.right.wrist_right",
    "overhead": "observation.images.left.top",
}


class BimanualVLADataset(Dataset):
    """Unified bimanual VLA dataset that yields chat-format training samples.

    Each item contains 3 camera images, 12-joint action history, and 96
    action tokens (8 timesteps x 12 joints) as the training target.

    For E episodes, each with F_i valid frames at 4 FPS:
        Total samples = sum(F_i for i in episodes)
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

        # Build index table: list of (episode_idx, 4fps_frame_idx)
        self._index_table: list[tuple[int, int]] = []
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

            # Store video metadata for all 3 cameras
            video_info = {}
            for cam_name, video_key in VIDEO_KEYS.items():
                chunk_idx = int(row[f"videos/{video_key}/chunk_index"])
                file_idx = int(row[f"videos/{video_key}/file_index"])
                video_info[cam_name] = {
                    "path": (
                        self.dataset_root
                        / "videos"
                        / video_key
                        / f"chunk-{chunk_idx:03d}"
                        / f"file-{file_idx:03d}.mp4"
                    ),
                    "from_ts": float(row[f"videos/{video_key}/from_timestamp"]),
                }

            self._episode_data[ep_idx] = {
                "actions": actions,
                "length": ep_length,
                "frame_indices_4fps": frame_indices_4fps,
                "num_frames_4fps": num_frames_4fps,
                "video_info": video_info,
            }

            # Valid frame range: need ACTION_CHUNK_LEN future steps
            num_valid = num_frames_4fps - ACTION_CHUNK_LEN
            for t in range(num_valid):
                self._index_table.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._index_table)

    def _load_frame(
        self, ep_idx: int, frame_30fps: int, cam_name: str
    ) -> torch.Tensor:
        """Load a single video frame from a specific camera.

        Returns:
            Tensor of shape (C, H, W) in [0, 1].
        """
        ep_data = self._episode_data[ep_idx]
        vi = ep_data["video_info"][cam_name]

        # Compute absolute timestamp within the MP4
        frame_ts = frame_30fps / DATASET_FPS
        absolute_ts = vi["from_ts"] + frame_ts

        frames = decode_video_frames(
            vi["path"],
            [absolute_ts],
            self.tolerance_s,
            self.video_backend,
        )
        frame = frames.squeeze(0)  # Remove batch dim
        # Ensure CHW format
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)
        return frame

    def _encode_12joint_action(self, action_12d: np.ndarray) -> list[int]:
        """Encode a 12-dim action (both arms) to 12 token IDs.

        Encodes left arm (indices 0:6) then right arm (indices 6:12)
        using the same ActionTokenizer for both.
        """
        left_tokens = self.action_tokenizer.encode_action(action_12d[:NUM_JOINTS_PER_ARM])
        right_tokens = self.action_tokenizer.encode_action(action_12d[NUM_JOINTS_PER_ARM:])
        return left_tokens + right_tokens

    def _format_action_history(
        self, actions_12d: np.ndarray
    ) -> str:
        """Format action history as text with token names.

        Args:
            actions_12d: (ACTION_HISTORY_LEN, 12) array of joint values.

        Returns:
            Formatted string with left | right token separation per timestep.
        """
        lines = []
        for i in range(ACTION_HISTORY_LEN):
            label = f"t-{ACTION_HISTORY_LEN - i}"
            left_tokens = self.action_tokenizer.encode_action(
                actions_12d[i, :NUM_JOINTS_PER_ARM]
            )
            right_tokens = self.action_tokenizer.encode_action(
                actions_12d[i, NUM_JOINTS_PER_ARM:]
            )
            left_names = self.action_tokenizer.token_ids_to_names(left_tokens)
            right_names = self.action_tokenizer.token_ids_to_names(right_tokens)
            lines.append(
                f"{label}: {' '.join(left_names)} | {' '.join(right_names)}"
            )
        return "\n".join(lines)

    def __getitem__(self, idx: int) -> list[dict]:
        """Get a single training sample as TRL-compatible chat messages.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        ep_idx, t_4fps = self._index_table[idx]
        ep_data = self._episode_data[ep_idx]
        aug = self.augmentations[ep_idx]
        frame_indices_4fps = ep_data["frame_indices_4fps"]

        # 1. Current images (all 3 cameras)
        current_30fps = frame_indices_4fps[t_4fps]
        left_wrist_img = self._load_frame(ep_idx, current_30fps, "left_wrist")
        right_wrist_img = self._load_frame(ep_idx, current_30fps, "right_wrist")
        overhead_img = self._load_frame(ep_idx, current_30fps, "overhead")

        # 2. Action history (last 4 steps at 4 FPS, all 12 joints)
        history_4fps_indices = [
            max(0, t_4fps - ACTION_HISTORY_LEN + i) for i in range(ACTION_HISTORY_LEN)
        ]
        history_30fps = [frame_indices_4fps[i] for i in history_4fps_indices]
        actions = ep_data["actions"]  # (ep_length, 12)
        history_actions = actions[history_30fps]  # (4, 12)
        action_history_text = self._format_action_history(history_actions)

        # 3. Prompts from augmentation
        system_prompt = aug["system_prompt"]
        user_prompt = aug["user_prompt_template"].format(
            task_description=aug["task_description"],
            action_history=action_history_text,
        )
        cot = aug["chain_of_thought_template"]

        # 4. Action chunk (8 future timesteps at 4 FPS, all 12 joints)
        chunk_4fps_indices = list(range(t_4fps, t_4fps + ACTION_CHUNK_LEN))
        chunk_30fps = [frame_indices_4fps[i] for i in chunk_4fps_indices]
        chunk_actions = actions[chunk_30fps]  # (8, 12)

        # Encode all 8 timesteps x 12 joints = 96 tokens
        chunk_token_ids = []
        for t in range(ACTION_CHUNK_LEN):
            chunk_token_ids.extend(self._encode_12joint_action(chunk_actions[t]))
        chunk_token_names = self.action_tokenizer.token_ids_to_names(chunk_token_ids)

        # 5. Format assistant response: CoT + action tokens
        # Group tokens by timestep (8 groups of 12) for the output
        action_text_parts = []
        for t in range(ACTION_CHUNK_LEN):
            start = t * NUM_JOINTS_TOTAL
            end = start + NUM_JOINTS_TOTAL
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
                    {"type": "image", "image": left_wrist_img},
                    {"type": "image", "image": right_wrist_img},
                    {"type": "image", "image": overhead_img},
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
    """Validate the BimanualVLADataset."""
    import re

    dataset_root = Path(dataset_root)
    print("Loading dataset...")
    ds = BimanualVLADataset(dataset_root)

    # Check total size = sum(valid_frames per episode)
    total_valid = 0
    for ep_idx, ep_data in ds._episode_data.items():
        num_valid = ep_data["num_frames_4fps"] - ACTION_CHUNK_LEN
        total_valid += num_valid
    expected_len = total_valid
    actual_len = len(ds)
    print(f"Dataset length: {actual_len} (expected: {expected_len})")
    assert actual_len == expected_len, f"Length mismatch: {actual_len} != {expected_len}"
    print("Length check PASSED")

    # Iterate a few samples
    print("\nChecking 5 samples...")
    test_indices = [0, 1, len(ds) // 4, len(ds) // 2, len(ds) - 1]
    for i in test_indices:
        sample = ds[i]
        ep_idx, t_4fps = ds._index_table[i]

        # Check message structure
        assert len(sample) == 3, f"Expected 3 messages, got {len(sample)}"
        assert sample[0]["role"] == "system"
        assert sample[1]["role"] == "user"
        assert sample[2]["role"] == "assistant"

        # Check 3 images in user message
        user_content = sample[1]["content"]
        assert user_content[0]["type"] == "image"
        assert user_content[1]["type"] == "image"
        assert user_content[2]["type"] == "image"
        assert user_content[3]["type"] == "text"

        for img_idx in range(3):
            image = user_content[img_idx]["image"]
            assert isinstance(image, torch.Tensor)
            assert image.ndim == 3, f"Expected 3D tensor, got {image.ndim}D"
            assert image.shape[0] == 3, f"Expected CHW format, got shape {image.shape}"
            assert image.min() >= 0 and image.max() <= 1, (
                "Image values out of [0,1] range"
            )

        # Check assistant response has 96 action tokens
        assistant_text = sample[2]["content"][0]["text"]
        assert "<think>" in assistant_text
        assert "</think>" in assistant_text
        action_tokens = re.findall(r"<action_j\d+_b\d+>", assistant_text)
        assert len(action_tokens) == 96, (
            f"Expected 96 action tokens, got {len(action_tokens)}"
        )

        # Verify token structure: 8 timesteps of 12 tokens each
        # Each timestep should have j0-j5 (left) then j0-j5 (right)
        for t in range(ACTION_CHUNK_LEN):
            timestep_tokens = action_tokens[
                t * NUM_JOINTS_TOTAL : (t + 1) * NUM_JOINTS_TOTAL
            ]
            for j in range(NUM_JOINTS_PER_ARM):
                assert f"_j{j}_" in timestep_tokens[j], (
                    f"Timestep {t}, left joint {j}: unexpected token {timestep_tokens[j]}"
                )
                assert f"_j{j}_" in timestep_tokens[NUM_JOINTS_PER_ARM + j], (
                    f"Timestep {t}, right joint {j}: unexpected token "
                    f"{timestep_tokens[NUM_JOINTS_PER_ARM + j]}"
                )

        cam_names = ["left_wrist", "right_wrist", "overhead"]
        img_shapes = [
            user_content[k]["image"].shape for k in range(3)
        ]
        print(
            f"  Sample {i}: ep={ep_idx}, t={t_4fps}, "
            f"images={[f'{cam_names[k]}:{s}' for k, s in enumerate(img_shapes)]}, "
            f"tokens={len(action_tokens)}"
        )

    print("\nAll validation checks PASSED")


if __name__ == "__main__":
    import sys

    dataset_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/grabber_picker_black_marker_20260226_211245"
    )
    validate_dataset(dataset_root)
