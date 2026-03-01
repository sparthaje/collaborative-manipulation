#!/usr/bin/env python3
"""
Load camera definitions from configs/robot.yaml and verify each camera:
1) camera can open
2) frame width/height matches expected values
3) user confirms index<->key mapping (wrist_left/wrist_right/top)

Usage:
  python scripts/find_camera.py
  python scripts/find_camera.py --config configs/robot.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify cameras from robot.yaml")
    parser.add_argument(
        "--config",
        default="configs/robot.yaml",
        help="Path to robot config YAML (default: configs/robot.yaml)",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=10.0,
        help="Seconds to show each camera before auto-continue (default: 10)",
    )
    return parser.parse_args()


def _normalize_camera_entry(entry: dict) -> dict:
    """
    Convert entries shaped like:
      { "wrist_left": [ {"width": 1920}, {"height": 1080}, {"index": 0} ] }
    into:
      { "name": "wrist_left", "width": 1920, "height": 1080, "index": 0 }
    """
    if len(entry) != 1:
        raise ValueError(f"Invalid camera entry: {entry!r}")

    name, attrs = next(iter(entry.items()))
    merged = {"name": name}

    if isinstance(attrs, list):
        for item in attrs:
            if isinstance(item, dict):
                merged.update(item)
    elif isinstance(attrs, dict):
        merged.update(attrs)
    else:
        raise ValueError(f"Invalid attributes for camera {name!r}: {attrs!r}")

    required = ("index", "width", "height")
    missing = [k for k in required if k not in merged]
    if missing:
        raise ValueError(f"Camera {name!r} missing fields: {missing}")

    merged["index"] = int(merged["index"])
    merged["width"] = int(merged["width"])
    merged["height"] = int(merged["height"])
    return merged


def load_cameras(config_path: Path) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_cameras = cfg.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ValueError("No cameras list found in config")

    return [_normalize_camera_entry(entry) for entry in raw_cameras]


def prompt_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def verify_camera(camera: dict, preview_seconds: float) -> bool:
    name = camera["name"]
    cam_index = camera["index"]
    expected_w = camera["width"]
    expected_h = camera["height"]

    print(f"\n=== Checking {name} (index={cam_index}) ===")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[FAIL] Could not open camera index {cam_index}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_h)

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[FAIL] Failed to read frame from camera index {cam_index}")
        cap.release()
        return False

    actual_h, actual_w = frame.shape[:2]
    size_ok = actual_w == expected_w and actual_h == expected_h
    status = "PASS" if size_ok else "FAIL"
    print(f"[{status}] Resolution expected={expected_w}x{expected_h}, actual={actual_w}x{actual_h}")

    window_name = f"{name} @ index {cam_index}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(
        "Preview controls: focus the image window and press space/enter/q/esc to continue "
        f"(auto-continue in {preview_seconds:.1f}s)."
    )
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[FAIL] Stream dropped while previewing")
                return False

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, 13, 32, ord("q")):
                break
            if time.monotonic() - start >= preview_seconds:
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    mapping_ok = prompt_yes_no(f"Does camera index {cam_index} match key '{name}'?")
    print(f"[{'PASS' if mapping_ok else 'FAIL'}] Mapping confirmation for {name}")
    return size_ok and mapping_ok


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        cameras = load_cameras(config_path)
    except Exception as exc:
        print(f"Error: failed to parse config: {exc}", file=sys.stderr)
        return 2

    print(f"Loaded {len(cameras)} camera definitions from {config_path}")
    results = []
    for camera in cameras:
        results.append((camera["name"], verify_camera(camera, args.preview_seconds)))

    print("\n=== Summary ===")
    all_ok = True
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
