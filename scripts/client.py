import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
import yaml

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


def parse_args():
    parser = argparse.ArgumentParser(description="Robot-side closed-loop inference client.")
    parser.add_argument(
        "--task",
        required=True,
        help='Task string, e.g. "picker;grabber;black_marker".',
    )
    parser.add_argument(
        "--server",
        required=True,
        help='Inference server URL, e.g. "http://192.168.1.100:5000".',
    )
    parser.add_argument(
        "--config",
        default="configs/robot.yaml",
        help="Path to robot config YAML file.",
    )
    return parser.parse_args()


def parse_camera_index(config: dict, side: str) -> int:
    """Extract camera index for the given side from the config YAML."""
    for cam_item in config.get("cameras", []):
        if not isinstance(cam_item, dict) or len(cam_item) != 1:
            continue
        cam_name = next(iter(cam_item.keys()))
        cam_entries = cam_item[cam_name]
        if not isinstance(cam_entries, list):
            continue

        cam_cfg: dict[str, Any] = {}
        for entry in cam_entries:
            if isinstance(entry, dict):
                cam_cfg.update(entry)

        if cam_cfg.get("arm") == side and "index" in cam_cfg:
            return cam_cfg["index"]

    raise ValueError(f"No camera config found for side '{side}'")


def init_follower(config: dict, side: str) -> SO101Follower:
    """Create and connect a follower arm (no cameras — captured separately via OpenCV)."""
    robot_config = config["robot"]

    follower = SO101Follower(
        SO101FollowerConfig(
            port=robot_config[f"port_{side}"],
            id=robot_config.get(f"id_{side}"),
            cameras={},
            use_degrees=config.get("degrees", True),
        )
    )
    follower.connect(calibrate=False)
    return follower


def get_joint_positions(follower: SO101Follower) -> np.ndarray:
    """Read current 6-DOF joint positions from a follower."""
    obs = follower.get_observation()
    state = obs["observation.state"]
    if isinstance(state, torch.Tensor):
        return state.numpy().astype(np.float64)
    return np.array(state, dtype=np.float64)


def interpolate_actions(waypoints_4x6: np.ndarray, source_fps: int = 4, target_fps: int = 30) -> np.ndarray:
    """Interpolate 4 waypoints at source_fps to a target_fps trajectory.

    Returns ~(target_fps / source_fps * num_waypoints) frames.
    """
    num_waypoints = waypoints_4x6.shape[0]
    duration = num_waypoints / source_fps  # 1 second for 4 waypoints at 4 FPS
    num_target = int(duration * target_fps)

    source_times = np.linspace(0, duration, num_waypoints)
    target_times = np.linspace(0, duration, num_target)

    num_joints = waypoints_4x6.shape[1]
    result = np.zeros((num_target, num_joints))
    for j in range(num_joints):
        result[:, j] = np.interp(target_times, source_times, waypoints_4x6[:, j])

    return result


def capture_jpeg(cap: cv2.VideoCapture) -> bytes:
    """Capture a frame and encode it as JPEG bytes."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")
    _, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Connect followers (no leaders needed)
    print("Connecting left follower...")
    left_follower = init_follower(config, "left")
    print("Connecting right follower...")
    right_follower = init_follower(config, "right")

    # Open cameras via OpenCV for fast capture
    left_cam_index = parse_camera_index(config, "left")
    right_cam_index = parse_camera_index(config, "right")

    cap_left = cv2.VideoCapture(left_cam_index)
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap_right = cv2.VideoCapture(right_cam_index)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap_left.isOpened():
        raise RuntimeError(f"Cannot open left camera (index {left_cam_index})")
    if not cap_right.isOpened():
        raise RuntimeError(f"Cannot open right camera (index {right_cam_index})")

    # Read initial joint positions
    left_pos = get_joint_positions(left_follower)
    right_pos = get_joint_positions(right_follower)

    # Initialize action history buffers (4 slots each, filled with initial position)
    left_history = np.tile(left_pos, (4, 1))   # (4, 6)
    right_history = np.tile(right_pos, (4, 1))  # (4, 6)

    predict_url = f"{args.server.rstrip('/')}/predict"

    print(f"Client ready. Task: {args.task}")
    print(f"Server: {predict_url}")
    print("Starting control loop... Press Ctrl+C to stop.")

    try:
        while True:
            # 1. Capture wrist images
            left_jpeg = capture_jpeg(cap_left)
            right_jpeg = capture_jpeg(cap_right)

            # 2. POST /predict
            files = {
                "left_image": ("left.jpg", left_jpeg, "image/jpeg"),
                "right_image": ("right.jpg", right_jpeg, "image/jpeg"),
            }
            data = {
                "task": args.task,
                "left_history": json.dumps(left_history.tolist()),
                "right_history": json.dumps(right_history.tolist()),
            }

            try:
                resp = requests.post(predict_url, files=files, data=data, timeout=10)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"Server request failed: {e}")
                time.sleep(0.5)
                continue

            result = resp.json()

            # 3. Parse action chunk: (2, 8, 6)
            action_chunk = np.array(result["actions"])  # (2, 8, 6)
            cot = result.get("cot", "")

            # 4. Log CoT
            if cot:
                print(f"[CoT] {cot}")

            # 5. Keep only first 4 steps (discard steps 5-8)
            left_waypoints = action_chunk[0, :4, :]   # (4, 6)
            right_waypoints = action_chunk[1, :4, :]   # (4, 6)

            # 6. Interpolate 4 waypoints -> 30 FPS trajectory
            left_traj = interpolate_actions(left_waypoints)
            right_traj = interpolate_actions(right_waypoints)

            # 7. Execute trajectory at 30 FPS
            num_steps = min(left_traj.shape[0], right_traj.shape[0])
            for step in range(num_steps):
                loop_start = time.perf_counter()

                left_action = {"action": torch.tensor(left_traj[step], dtype=torch.float32)}
                right_action = {"action": torch.tensor(right_traj[step], dtype=torch.float32)}

                left_follower.send_action(left_action)
                right_follower.send_action(right_action)

                # Maintain 30 FPS
                dt = time.perf_counter() - loop_start
                sleep_time = max(1.0 / 30 - dt, 0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 8. Update history with the 4 commanded waypoints (not interpolated)
            left_history = left_waypoints.copy()
            right_history = right_waypoints.copy()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Release cameras
        cap_left.release()
        cap_right.release()

        # Disconnect arms
        if left_follower.is_connected:
            left_follower.disconnect()
        if right_follower.is_connected:
            right_follower.disconnect()

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
