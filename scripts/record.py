import argparse
import os
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import init_logging

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader

RobotPair = Tuple[SO101Leader, SO101Follower]
RobotPairList = Dict[str, RobotPair]


def _non_empty_task(value: str) -> str:
    task = value.strip()
    if not task:
        raise argparse.ArgumentTypeError("--task must be a non-empty string.")
    return task


def parse_args():
    parser = argparse.ArgumentParser(description="Unified teleoperate + record script.")
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to operate.",
    )
    parser.add_argument(
        "--record-data",
        dest="record_data",
        action="store_true",
        default=True,
        help="Enable data recording (default: enabled).",
    )

    parser.add_argument(
        "--teleop",
        dest="record_data",
        action="store_false",
        help="Disable data recording.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/collaborative_manipulation",
        help="LeRobot dataset repo_id format: <namespace>/<dataset_name>.",
    )
    parser.add_argument(
        "--task",
        type=_non_empty_task,
        required=True,
        help="Task string stored in the dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/robot.yaml",
        help="Path to robot config YAML file.",
    )
    return parser.parse_args()


def _build_output_dir(task: str) -> Path:
    task_name = "_".join(task.strip().split())
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data") / f"{task_name}_{now}"


def _camera_configs_for_side(
    config: dict, side: str, selected_arm: str, record_data: bool
) -> dict[str, OpenCVCameraConfig]:
    if not record_data:
        return {}

    parsed: dict[str, OpenCVCameraConfig] = {}
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

        cam_arm = cam_cfg.get("arm")
        if cam_arm == side:
            pass
        elif cam_arm == "both":
            # Shared cameras should only be attached when recording both arms,
            # and they are routed to the left follower to avoid duplication.
            if not (selected_arm == "both" and side == "left"):
                continue
        else:
            continue

        if "index" not in cam_cfg:
            continue

        parsed[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_cfg["index"],
            width=cam_cfg.get("width"),
            height=cam_cfg.get("height"),
            fps=cam_cfg.get("fps", config.get("fps", 30)),
        )

    return parsed


def init(arm: str, config: dict, record_data: bool) -> RobotPairList:
    """
    Reads config file and args to figure out which arms and cameras to use
    """
    robot_config = config["robot"]
    teleop_config = config["teleop"]

    # leader, follower pairs
    arms: RobotPairList = {}
    if arm == "left" or arm == "both":
        left_cameras = _camera_configs_for_side(
            config, side="left", selected_arm=arm, record_data=record_data
        )
        arms["left"] = (
            SO101Leader(
                SO101LeaderConfig(
                    port=teleop_config["port_left"],
                    id=teleop_config.get("id_left"),
                    use_degrees=config["degrees"],
                )
            ),
            SO101Follower(
                SO101FollowerConfig(
                    port=robot_config["port_left"],
                    id=robot_config.get("id_left"),
                    cameras=left_cameras,
                    use_degrees=config["degrees"],
                )
            ),
        )

    if arm == "right" or arm == "both":
        right_cameras = _camera_configs_for_side(
            config, side="right", selected_arm=arm, record_data=record_data
        )
        arms["right"] = (
            SO101Leader(
                SO101LeaderConfig(
                    port=teleop_config["port_right"],
                    id=teleop_config.get("id_right"),
                    use_degrees=config["degrees"],
                )
            ),
            SO101Follower(
                SO101FollowerConfig(
                    port=robot_config["port_right"],
                    id=robot_config.get("id_right"),
                    cameras=right_cameras,
                    use_degrees=config["degrees"],
                )
            ),
        )

    return arms


def _make_dataset(args, sides: list[str], arms: RobotPairList, fps: int) -> LeRobotDataset:
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features: dict[str, Any] = {}
    observation_features: dict[str, Any] = {}
    for side in sides:
        _, follower = arms[side]
        action_features.update({f"{side}.{k}": v for k, v in follower.action_features.items()})
        observation_features.update({f"{side}.{k}": v for k, v in follower.observation_features.items()})

    dataset_features = {}
    dataset_features.update(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=True,
        )
    )
    dataset_features.update(
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=True,
        )
    )

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=args.output_dir,
        robot_type="so101_follower",
        features=dataset_features,
        use_videos=True,
        image_writer_threads=max(1, len(sides) * 4),
        batch_encoding_size=1,
    )


class _EnterKeyMonitor:
    """Monitor Enter keypresses from the controlling terminal."""

    def __init__(self) -> None:
        self.fd: int | None = None
        self.file = None
        self.old_attrs = None
        self._stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._fallback_thread: threading.Thread | None = None

    def __enter__(self) -> "_EnterKeyMonitor":
        try:
            self.file = open("/dev/tty", "rb", buffering=0)
            self.fd = self.file.fileno()
            self.old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self._reader_thread = threading.Thread(target=self._read_tty_keys, daemon=True)
            self._reader_thread.start()
        except Exception:
            self.fd = None
            self.file = None
            self.old_attrs = None
            self._fallback_thread = threading.Thread(target=self._wait_stdin_enter, daemon=True)
            self._fallback_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None and self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        if self.file is not None:
            self.file.close()

    def _wait_stdin_enter(self) -> None:
        try:
            _ = sys.stdin.readline()
            self._stop.set()
        except Exception:
            return

    def _read_tty_keys(self) -> None:
        if self.fd is None:
            return
        while not self._stop.is_set():
            data = b""
            try:
                data = os.read(self.fd, 1)
            except OSError:
                return
            if data in (b"\n", b"\r"):
                self._stop.set()
                return

    def enter_pressed(self) -> bool:
        return self._stop.is_set()


def _run_episode(
    arms: RobotPairList,
    sides: list[str],
    dataset: LeRobotDataset | None,
    fps: int,
    task: str,
) -> None:
    print("Recording... press Enter to end this episode.")
    with _EnterKeyMonitor() as monitor:
        while True:
            loop_start = time.perf_counter()
            action_values: dict[str, float] = {}
            obs_values: dict[str, Any] = {}

            for side in sides:
                leader, follower = arms[side]
                raw_action = leader.get_action()
                sent_action = follower.send_action(raw_action)
                observation = follower.get_observation()

                action_values.update({f"{side}.{k}": v for k, v in sent_action.items()})
                obs_values.update({f"{side}.{k}": v for k, v in observation.items()})

            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_values, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {
                    **observation_frame,
                    **action_frame,
                    "task": task,
                }
                dataset.add_frame(frame)

            if monitor.enter_pressed():
                print("Hold on... Received keypress")
                break

            dt_s = time.perf_counter() - loop_start
            sleep_s = max(1.0 / fps - dt_s, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    if dataset is not None:
        dataset.save_episode()


def _run_reset_teleop(
    arms: RobotPairList,
    sides: list[str],
    fps: int,
) -> None:
    print("Reset mode... teleoperating without recording. Press Enter to start next episode.")
    with _EnterKeyMonitor() as monitor:
        while True:
            loop_start = time.perf_counter()

            for side in sides:
                leader, follower = arms[side]
                raw_action = leader.get_action()
                _ = follower.send_action(raw_action)
                _ = follower.get_observation()

            if monitor.enter_pressed():
                print("Hold on... Starting next episode")
                break

            dt_s = time.perf_counter() - loop_start
            sleep_s = max(1.0 / fps - dt_s, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)


def main() -> None:
    init_logging()
    args = parse_args()

    config_path = Path(args.config)
    config = dict()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    fps = config.get("fps", 30)

    arms = init(
        args.arm,
        config,
        record_data=args.record_data,
    )
    sides = list(sorted(arms.keys()))

    dataset: LeRobotDataset | None = None
    if args.record_data:
        args.output_dir = _build_output_dir(args.task)
        dataset = _make_dataset(args, sides, arms, fps=fps)

    for leader, follower in arms.values():
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)

    episode_idx = 0
    try:
        while True:
            if episode_idx == 0:
                input(f"Press Enter to start episode {episode_idx}: ")
            else:
                _run_reset_teleop(
                    arms=arms,
                    sides=sides,
                    fps=fps,
                )
            _run_episode(
                arms=arms,
                sides=sides,
                dataset=dataset,
                fps=fps,
                task=args.task,
            )

            print("Reset Environment")
            episode_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        if dataset is not None:
            dataset.finalize()
        for leader, follower in arms.values():
            if leader.is_connected:
                leader.disconnect()
            if follower.is_connected:
                follower.disconnect()


if __name__ == "__main__":
    main()
