"""Action tokenization for Siamese VLA fine-tuning.

Converts continuous 6-DoF joint actions to/from discrete token IDs.
Each of the 6 joints gets N=256 bins with shared ranges across both arms.
Token names: <action_j{joint}_b{bin}> for joint in [0,5], bin in [0,255].

Usage:
    from model.tokenizer import ActionTokenizer

    tokenizer = ActionTokenizer.from_stats_json("data/.../meta/stats.json")
    token_ids = tokenizer.encode_action(joint_values_6d)
    reconstructed = tokenizer.decode_action(token_ids)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActionTokenizer:
    """Discretizes 6-joint action vectors into token IDs and back."""

    NUM_BINS = 256
    NUM_JOINTS = 6
    MARGIN = 0.05  # 5% margin on each side of q01/q99 range

    def __init__(self, joint_mins: np.ndarray, joint_maxs: np.ndarray, num_bins: int = 256):
        """
        Args:
            joint_mins: (6,) array of per-joint lower bounds.
            joint_maxs: (6,) array of per-joint upper bounds.
            num_bins: Number of bins per joint (default 256).
        """
        assert joint_mins.shape == (self.NUM_JOINTS,)
        assert joint_maxs.shape == (self.NUM_JOINTS,)
        self.num_bins = num_bins
        self.joint_mins = joint_mins.astype(np.float64)
        self.joint_maxs = joint_maxs.astype(np.float64)
        self.joint_ranges = self.joint_maxs - self.joint_mins

        # Token name list: <action_j0_b0> ... <action_j0_b255> <action_j1_b0> ...
        self.token_names = [
            f"<action_j{j}_b{b}>"
            for j in range(self.NUM_JOINTS)
            for b in range(self.num_bins)
        ]
        # Map token name -> index within our token list (0-indexed)
        self._name_to_local_idx = {name: i for i, name in enumerate(self.token_names)}

        # These get set after register_with_tokenizer()
        self._token_id_offset: int | None = None

    @classmethod
    def from_stats_json(cls, stats_path: str | Path, num_bins: int = 256) -> ActionTokenizer:
        """Create tokenizer from a LeRobot stats.json file.

        Computes shared per-joint ranges as:
            min = min(q01_left[i], q01_right[i]) - margin
            max = max(q99_left[i], q99_right[i]) + margin
        """
        stats_path = Path(stats_path)
        with open(stats_path) as f:
            stats = json.load(f)

        q01 = np.array(stats["action"]["q01"])  # (12,)
        q99 = np.array(stats["action"]["q99"])  # (12,)

        # Left arm = indices 0:6, Right arm = indices 6:12
        q01_left, q01_right = q01[:6], q01[6:]
        q99_left, q99_right = q99[:6], q99[6:]

        # Shared ranges: component-wise min of q01, max of q99
        shared_min = np.minimum(q01_left, q01_right)
        shared_max = np.maximum(q99_left, q99_right)

        # Add margin (5% of range on each side)
        ranges = shared_max - shared_min
        margin = ranges * cls.MARGIN
        shared_min = shared_min - margin
        shared_max = shared_max + margin

        return cls(shared_min, shared_max, num_bins)

    @classmethod
    def from_multiple_stats(
        cls, stats_paths: list[str | Path], num_bins: int = 256
    ) -> ActionTokenizer:
        """Create tokenizer from multiple LeRobot stats.json files.

        Computes global per-joint ranges by taking the component-wise min of
        all q01 values and max of all q99 values across all datasets, then
        applying the standard margin.
        """
        all_q01_left = []
        all_q01_right = []
        all_q99_left = []
        all_q99_right = []

        for sp in stats_paths:
            with open(sp) as f:
                stats = json.load(f)
            q01 = np.array(stats["action"]["q01"])  # (12,)
            q99 = np.array(stats["action"]["q99"])  # (12,)
            all_q01_left.append(q01[:6])
            all_q01_right.append(q01[6:])
            all_q99_left.append(q99[:6])
            all_q99_right.append(q99[6:])

        # Global min across all datasets and both arms
        global_q01 = np.minimum(
            np.min(all_q01_left, axis=0), np.min(all_q01_right, axis=0)
        )
        global_q99 = np.maximum(
            np.max(all_q99_left, axis=0), np.max(all_q99_right, axis=0)
        )

        # Add margin
        ranges = global_q99 - global_q01
        margin = ranges * cls.MARGIN
        shared_min = global_q01 - margin
        shared_max = global_q99 + margin

        return cls(shared_min, shared_max, num_bins)

    def register_with_tokenizer(self, tokenizer) -> int:
        """Add action tokens to a HuggingFace tokenizer.

        Returns the number of tokens added.
        """
        num_added = tokenizer.add_tokens(self.token_names)
        # The first action token ID is the vocab size before we added them
        self._token_id_offset = len(tokenizer) - len(self.token_names)
        return num_added

    def get_token_id_offset(self, tokenizer) -> int:
        """Get the token ID of the first action token from a tokenizer that already has them."""
        first_token = self.token_names[0]
        token_id = tokenizer.convert_tokens_to_ids(first_token)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Token '{first_token}' not found in tokenizer. "
                "Call register_with_tokenizer() first or use a tokenizer that already has action tokens."
            )
        self._token_id_offset = token_id
        return token_id

    def encode_action(self, joint_values: np.ndarray) -> list[int]:
        """Encode a single timestep (6 joint values) to 6 token IDs.

        Args:
            joint_values: (6,) array of continuous joint values in degrees.

        Returns:
            List of 6 token IDs (global IDs if registered with tokenizer,
            otherwise local 0-based indices).
        """
        joint_values = np.asarray(joint_values, dtype=np.float64)
        assert joint_values.shape == (self.NUM_JOINTS,), f"Expected (6,), got {joint_values.shape}"

        # Normalize to [0, 1] then quantize to bin index
        normalized = (joint_values - self.joint_mins) / self.joint_ranges
        bin_indices = np.floor(normalized * self.num_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Convert to token IDs: joint j, bin b -> j * num_bins + b
        local_ids = [j * self.num_bins + int(bin_indices[j]) for j in range(self.NUM_JOINTS)]

        if self._token_id_offset is not None:
            return [lid + self._token_id_offset for lid in local_ids]
        return local_ids

    def decode_action(self, token_ids: list[int]) -> np.ndarray:
        """Decode 6 token IDs back to continuous joint values (bin centers).

        Args:
            token_ids: List of 6 token IDs.

        Returns:
            (6,) array of reconstructed joint values in degrees.
        """
        assert len(token_ids) == self.NUM_JOINTS, f"Expected 6 tokens, got {len(token_ids)}"

        offset = self._token_id_offset if self._token_id_offset is not None else 0
        values = np.zeros(self.NUM_JOINTS, dtype=np.float64)

        for i, tid in enumerate(token_ids):
            local_id = tid - offset
            joint_idx = local_id // self.num_bins
            bin_idx = local_id % self.num_bins
            assert joint_idx == i, (
                f"Token {i} maps to joint {joint_idx}, expected joint {i}. "
                f"Token ID={tid}, local_id={local_id}"
            )
            # Reconstruct from bin center
            values[i] = self.joint_mins[i] + (bin_idx + 0.5) * self.joint_ranges[i] / self.num_bins

        return values

    def encode_action_chunk(self, chunk: np.ndarray) -> list[int]:
        """Encode multi-timestep action chunk to flat token ID list.

        Args:
            chunk: (T, 6) array of joint values.

        Returns:
            List of T*6 token IDs.
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        assert chunk.ndim == 2 and chunk.shape[1] == self.NUM_JOINTS
        token_ids = []
        for t in range(chunk.shape[0]):
            token_ids.extend(self.encode_action(chunk[t]))
        return token_ids

    def decode_action_chunk(self, token_ids: list[int], num_joints: int = 6) -> np.ndarray:
        """Decode flat token ID list back to multi-timestep action chunk.

        Args:
            token_ids: Flat list of T*num_joints token IDs.
            num_joints: Number of joints per timestep (default 6).

        Returns:
            (T, num_joints) array of reconstructed joint values.
        """
        assert len(token_ids) % num_joints == 0, (
            f"Token count {len(token_ids)} not divisible by {num_joints}"
        )
        T = len(token_ids) // num_joints
        values = np.zeros((T, num_joints), dtype=np.float64)
        for t in range(T):
            start = t * num_joints
            values[t] = self.decode_action(token_ids[start : start + num_joints])
        return values

    def token_ids_to_names(self, token_ids: list[int]) -> list[str]:
        """Convert token IDs to token name strings."""
        offset = self._token_id_offset if self._token_id_offset is not None else 0
        return [self.token_names[tid - offset] for tid in token_ids]

    def token_names_to_ids(self, names: list[str]) -> list[int]:
        """Convert token name strings to token IDs."""
        offset = self._token_id_offset if self._token_id_offset is not None else 0
        return [self._name_to_local_idx[name] + offset for name in names]

    def bin_width(self, joint_idx: int) -> float:
        """Get the bin width for a given joint."""
        return self.joint_ranges[joint_idx] / self.num_bins

    def max_reconstruction_error(self) -> float:
        """Maximum possible reconstruction error (half bin width) across all joints."""
        return float(np.max(self.joint_ranges / self.num_bins / 2))


def validate_round_trip(
    stats_path: str | Path,
    data_path: str | Path | None = None,
    num_samples: int = 1000,
) -> None:
    """Validate that action tokenization is invertible within 1 bin width.

    If data_path is provided, validates on actual dataset actions.
    Otherwise, validates on random samples within the joint ranges.
    """
    tokenizer = ActionTokenizer.from_stats_json(stats_path)

    if data_path is not None:
        import pandas as pd

        df = pd.read_parquet(data_path)
        actions = df["action"].tolist()
        actions = np.array(actions)  # (N, 12)
        # Test both left and right arms
        indices = np.random.choice(len(actions), min(num_samples, len(actions)), replace=False)
        max_error = 0.0
        max_error_in_range = 0.0
        num_clamped = 0
        num_total = 0
        for idx in indices:
            for arm_offset in [0, 6]:
                joint_vals = actions[idx, arm_offset : arm_offset + 6]
                token_ids = tokenizer.encode_action(joint_vals)
                reconstructed = tokenizer.decode_action(token_ids)
                error = np.abs(joint_vals - reconstructed)
                max_error = max(max_error, float(np.max(error)))

                # Check if any values are outside the quantization range (clamped)
                in_range = np.all(
                    (joint_vals >= tokenizer.joint_mins)
                    & (joint_vals <= tokenizer.joint_maxs)
                )
                num_total += 1
                if not in_range:
                    num_clamped += 1
                else:
                    max_error_in_range = max(max_error_in_range, float(np.max(error)))
    else:
        max_error = 0.0
        for _ in range(num_samples):
            joint_vals = (
                tokenizer.joint_mins
                + np.random.rand(tokenizer.NUM_JOINTS) * tokenizer.joint_ranges
            )
            token_ids = tokenizer.encode_action(joint_vals)
            reconstructed = tokenizer.decode_action(token_ids)
            error = np.abs(joint_vals - reconstructed)
            max_error = max(max_error, float(np.max(error)))

    max_bin_width = float(np.max(tokenizer.joint_ranges / tokenizer.num_bins))
    print(f"Max reconstruction error (all): {max_error:.6f} degrees")
    if data_path is not None:
        print(f"Max reconstruction error (in-range): {max_error_in_range:.6f} degrees")
        print(f"Clamped samples: {num_clamped}/{num_total} "
              f"({100*num_clamped/num_total:.1f}% — outliers beyond q01/q99+margin)")
    print(f"Max bin width: {max_bin_width:.6f} degrees")
    # For in-range values, error must be < 1 bin width
    check_error = max_error_in_range if data_path is not None else max_error
    assert check_error < max_bin_width, (
        f"Round-trip error {check_error:.6f} exceeds bin width {max_bin_width:.6f}"
    )
    print("Round-trip validation PASSED")

    # Also validate encode->decode->encode consistency
    for _ in range(num_samples):
        joint_vals = (
            tokenizer.joint_mins
            + np.random.rand(tokenizer.NUM_JOINTS) * tokenizer.joint_ranges
        )
        token_ids = tokenizer.encode_action(joint_vals)
        reconstructed = tokenizer.decode_action(token_ids)
        re_encoded = tokenizer.encode_action(reconstructed)
        assert token_ids == re_encoded, (
            f"encode(decode(tokens)) != tokens: {token_ids} vs {re_encoded}"
        )
    print("Idempotency validation PASSED (encode(decode(tokens)) == tokens)")


if __name__ == "__main__":
    import sys

    dataset_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/grabber_picker_black_marker_20260228_150311"
    )

    stats_path = dataset_root / "meta" / "stats.json"
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"

    print("=== Action Tokenizer Configuration ===")
    tok = ActionTokenizer.from_stats_json(stats_path)
    for j in range(tok.NUM_JOINTS):
        print(
            f"  Joint {j}: [{tok.joint_mins[j]:.2f}, {tok.joint_maxs[j]:.2f}] "
            f"bin_width={tok.bin_width(j):.4f}°"
        )
    print(f"  Total tokens: {len(tok.token_names)}")
    print(f"  Max reconstruction error bound: {tok.max_reconstruction_error():.4f}°")
    print()

    print("=== Round-Trip Validation (random samples) ===")
    validate_round_trip(stats_path, num_samples=1000)
    print()

    print("=== Round-Trip Validation (dataset actions) ===")
    validate_round_trip(stats_path, data_path=data_path, num_samples=1000)
