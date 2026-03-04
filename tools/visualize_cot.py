#!/usr/bin/env python3
"""Visualize per-frame CoT stage labels on side-by-side wrist camera videos.

Generates an MP4 with left and right wrist camera views side-by-side,
each captioned with the current CoT stage label and a color-coded bar.

Usage:
    python tools/visualize_cot.py data/picker_grabber_black_marker_20260226_204853 --episode 0
    python tools/visualize_cot.py data/picker_grabber_black_marker_20260226_204853 --episode 1 -o output.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from lerobot.datasets.video_utils import decode_video_frames

# Stage -> color (BGR for OpenCV drawing)
STAGE_COLORS = {
    "picking": (0, 200, 255),     # orange
    "reaching": (255, 200, 0),    # cyan-ish
    "releasing": (0, 255, 0),     # green
    "holding": (255, 0, 200),     # magenta
    "disengaging": (0, 100, 255), # dark orange
    "watching": (200, 200, 200),  # gray
    "dropping": (0, 0, 255),      # red
}

DATASET_FPS = 30
TARGET_FPS = 4
BAR_HEIGHT = 8

VIDEO_KEYS = {
    "left": "observation.images.left.wrist_left",
    "right": "observation.images.right.wrist_right",
}


def get_video_path(dataset_root: Path, episode_row: pd.Series, video_key: str) -> Path:
    chunk_idx = int(episode_row[f"videos/{video_key}/chunk_index"])
    file_idx = int(episode_row[f"videos/{video_key}/file_index"])
    return (
        dataset_root / "videos" / video_key
        / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    )


def draw_label(frame: np.ndarray, label: str, color: tuple[int, ...]) -> np.ndarray:
    """Draw stage label text and color bar on a frame."""
    h, w = frame.shape[:2]

    # Color bar at top
    frame[:BAR_HEIGHT, :] = color

    # Text at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(label, font, scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = h - 12

    # Background rectangle for readability
    cv2.rectangle(
        frame,
        (tx - 4, ty - text_size[1] - 4),
        (tx + text_size[0] + 4, ty + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, label, (tx, ty), font, scale, color, thickness)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize CoT stage labels")
    parser.add_argument("dataset", type=Path, help="Path to dataset root")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--video-backend", default="torchcodec", help="Video backend")
    args = parser.parse_args()

    dataset_root = args.dataset

    # Load augmentation data (has CoT at 4FPS)
    aug_path = dataset_root / "augmentation" / f"episode_{args.episode:04d}.json"
    with open(aug_path) as f:
        aug = json.load(f)

    frame_indices_4fps = aug["frame_indices_4fps"]
    left_cot = aug["left_arm"]["chain_of_thought"]
    right_cot = aug["right_arm"]["chain_of_thought"]
    num_frames_4fps = len(frame_indices_4fps)

    # Load episode metadata
    ep_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    ep_row = ep_df[ep_df["episode_index"] == args.episode].iloc[0]

    left_video = get_video_path(dataset_root, ep_row, VIDEO_KEYS["left"])
    right_video = get_video_path(dataset_root, ep_row, VIDEO_KEYS["right"])
    left_from_ts = float(ep_row[f"videos/{VIDEO_KEYS['left']}/from_timestamp"])
    right_from_ts = float(ep_row[f"videos/{VIDEO_KEYS['right']}/from_timestamp"])

    # Compute timestamps for all 4FPS frames
    timestamps = [idx / DATASET_FPS for idx in frame_indices_4fps]
    left_timestamps = [left_from_ts + ts for ts in timestamps]
    right_timestamps = [right_from_ts + ts for ts in timestamps]

    # Decode all 4FPS frames in batch using lerobot's decoder (handles AV1)
    print(f"Decoding {num_frames_4fps} frames from {left_video.name} ...")
    left_frames = decode_video_frames(
        left_video, left_timestamps, 0.05, args.video_backend
    )
    print(f"Decoding {num_frames_4fps} frames from {right_video.name} ...")
    right_frames = decode_video_frames(
        right_video, right_timestamps, 0.05, args.video_backend
    )

    # decode_video_frames returns (N, C, H, W) float32 [0,1] — convert to NHWC BGR uint8
    left_frames = left_frames.permute(0, 2, 3, 1)   # NCHW -> NHWC
    right_frames = right_frames.permute(0, 2, 3, 1)
    left_frames = (left_frames.numpy() * 255).astype(np.uint8)[:, :, :, ::-1].copy()
    right_frames = (right_frames.numpy() * 255).astype(np.uint8)[:, :, :, ::-1].copy()

    h, w = left_frames.shape[1:3]
    combined_w = w * 2

    out_path = args.output or (
        dataset_root / "augmentation" / f"cot_vis_ep{args.episode:04d}.mp4"
    )

    # Use pyav with H.264 for broad player compatibility
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=TARGET_FPS)
    stream.width = combined_w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for t in range(num_frames_4fps):
        frame_l = left_frames[t].copy()
        frame_r = right_frames[t].copy()

        left_stage = left_cot[t].replace("<think>", "").replace("</think>", "")
        right_stage = right_cot[t].replace("<think>", "").replace("</think>", "")

        left_color = STAGE_COLORS.get(left_stage, (255, 255, 255))
        right_color = STAGE_COLORS.get(right_stage, (255, 255, 255))

        frame_l = draw_label(frame_l, f"L: {left_stage}", left_color)
        frame_r = draw_label(frame_r, f"R: {right_stage}", right_color)

        combined = np.hstack([frame_l, frame_r])
        # BGR -> RGB for pyav
        combined_rgb = combined[:, :, ::-1].copy()
        video_frame = av.VideoFrame.from_ndarray(combined_rgb, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Written {num_frames_4fps} frames at {TARGET_FPS} FPS to {out_path}")


if __name__ == "__main__":
    main()
