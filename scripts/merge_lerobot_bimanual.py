#!/usr/bin/env python3
"""Merge two single-arm LeRobot datasets into one bimanual-style dataset.

Expected input layout:
  <input_dir>/<dataset_a>/...
  <input_dir>/<dataset_b>/...

Output layout mirrors LeRobot v3 basic structure:
  <output_dir>/data/chunk-000/file-000.parquet
  <output_dir>/videos/observation.images.wrist_left/chunk-000/file-000.mp4
  <output_dir>/videos/observation.images.wrist_right/chunk-000/file-000.mp4
  <output_dir>/meta/{info.json,stats.json,tasks.parquet,episodes/chunk-000/file-000.parquet}
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    root: Path
    name: str
    info: dict[str, Any]
    stats: dict[str, Any]
    data: pd.DataFrame
    episodes: pd.DataFrame
    tasks: pd.DataFrame


def load_bundle(root: Path) -> DatasetBundle:
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = root / "meta" / "tasks.parquet"
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"

    missing = [
        str(p)
        for p in [data_path, episodes_path, tasks_path, info_path, stats_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required files in {root}: {missing}")

    with info_path.open() as f:
        info = json.load(f)
    with stats_path.open() as f:
        stats = json.load(f)

    data = pd.read_parquet(data_path)
    episodes = pd.read_parquet(episodes_path)
    tasks = pd.read_parquet(tasks_path)

    required_data_cols = {
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if not required_data_cols.issubset(data.columns):
        raise ValueError(f"Dataset {root} missing required data columns")

    return DatasetBundle(root=root, name=root.name, info=info, stats=stats, data=data, episodes=episodes, tasks=tasks)


def normalize_task_map(tasks_df: pd.DataFrame) -> dict[int, str]:
    # LeRobot tasks.parquet has task strings as index and task_index as column.
    return {int(row["task_index"]): str(idx) for idx, row in tasks_df.iterrows()}


def align_indices(ts_a: np.ndarray, ts_b: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    ia, ib = 0, 0
    match_a: list[int] = []
    match_b: list[int] = []

    while ia < len(ts_a) and ib < len(ts_b):
        diff = float(ts_a[ia] - ts_b[ib])
        if abs(diff) <= tolerance:
            match_a.append(ia)
            match_b.append(ib)
            ia += 1
            ib += 1
        elif diff < 0:
            ia += 1
        else:
            ib += 1

    return np.asarray(match_a, dtype=np.int64), np.asarray(match_b, dtype=np.int64)


def summarize_numeric(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]

    q01 = np.quantile(arr, 0.01, axis=0)
    q10 = np.quantile(arr, 0.10, axis=0)
    q50 = np.quantile(arr, 0.50, axis=0)
    q90 = np.quantile(arr, 0.90, axis=0)
    q99 = np.quantile(arr, 0.99, axis=0)

    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": q01.tolist(),
        "q10": q10.tolist(),
        "q50": q50.tolist(),
        "q90": q90.tolist(),
        "q99": q99.tolist(),
    }


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)


def copy_video_tree(src_root: Path, dst_root: Path, dst_feature_name: str) -> None:
    src = src_root / "videos" / "observation.images.wrist"
    dst = dst_root / "videos" / dst_feature_name
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def get_wallclock_episode_ranges(bundle: DatasetBundle) -> dict[int, tuple[float, float]]:
    from_col = "videos/observation.images.wrist/from_timestamp"
    to_col = "videos/observation.images.wrist/to_timestamp"
    if from_col not in bundle.episodes.columns or to_col not in bundle.episodes.columns:
        return {}

    ranges: dict[int, tuple[float, float]] = {}
    rows = (
        bundle.episodes[["episode_index", from_col, to_col]]
        .sort_values("episode_index")
        .itertuples(index=False)
    )
    for row in rows:
        ep_idx = int(row[0])
        start = float(row[1])
        end = float(row[2])
        if np.isfinite(start) and np.isfinite(end) and end >= start:
            ranges[ep_idx] = (start, end)
    return ranges


def compute_episode_selection(
    left: DatasetBundle,
    right: DatasetBundle,
) -> tuple[list[int], dict[int, tuple[float, float]], dict[int, tuple[float, float]], str]:
    left_data_ids = sorted(int(x) for x in left.data["episode_index"].unique())
    right_data_ids = sorted(int(x) for x in right.data["episode_index"].unique())
    shared_data_ids = sorted(set(left_data_ids).intersection(right_data_ids))

    left_wallclock = get_wallclock_episode_ranges(left)
    right_wallclock = get_wallclock_episode_ranges(right)

    usable: list[int] = []
    dropped: list[tuple[int, str]] = []
    for ep_idx in shared_data_ids:
        left_has = ep_idx in left_wallclock
        right_has = ep_idx in right_wallclock
        if left_has and right_has:
            usable.append(ep_idx)
            continue
        if not left_has and not right_has:
            dropped.append((ep_idx, "missing wallclock metadata on both arms"))
        elif not left_has:
            dropped.append((ep_idx, "missing wallclock metadata on left arm"))
        else:
            dropped.append((ep_idx, "missing wallclock metadata on right arm"))

    lines: list[str] = []
    lines.append("Episode merge eligibility report")
    lines.append("")
    lines.append("1) Data available in both arms")
    lines.append(f"- left dataset: {left.root}")
    lines.append(f"- right dataset: {right.root}")
    lines.append(f"- left episode IDs in frame data: {left_data_ids}")
    lines.append(f"- right episode IDs in frame data: {right_data_ids}")
    lines.append(f"- shared episode IDs in frame data: {shared_data_ids}")
    lines.append(f"- left episode IDs with wallclock metadata: {sorted(left_wallclock)}")
    lines.append(f"- right episode IDs with wallclock metadata: {sorted(right_wallclock)}")
    lines.append(f"- episodes kept for merge (shared + wallclock on both): {usable}")
    lines.append("")
    lines.append("2) Data dropped")
    if dropped:
        for ep_idx, reason in dropped:
            lines.append(f"- episode {ep_idx}: {reason}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Policy: episodes missing wallclock metadata on either arm are dropped.")

    return usable, left_wallclock, right_wallclock, "\n".join(lines) + "\n"


def show_episode_overlap_plot(
    left_ranges: dict[int, tuple[float, float]],
    right_ranges: dict[int, tuple[float, float]],
    episode_ids: list[int],
) -> None:
    left_ranges_list = [(ep, left_ranges[ep][0], left_ranges[ep][1]) for ep in episode_ids if ep in left_ranges]
    right_ranges_list = [(ep, right_ranges[ep][0], right_ranges[ep][1]) for ep in episode_ids if ep in right_ranges]
    left_ranges = left_ranges_list
    right_ranges = right_ranges_list
    all_ranges = left_ranges + right_ranges
    if not all_ranges:
        raise ValueError("No episode timestamps found to plot")

    episode_ids = sorted({ep for ep, _, _ in all_ranges})
    ep_to_color_idx = {ep: i for i, ep in enumerate(episode_ids)}
    cmap = plt.get_cmap("tab20", max(len(episode_ids), 1))

    fig_height = max(6, 0.32 * len(all_ranges) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y = 0
    yticks: list[float] = []
    ylabels: list[str] = []
    left_by_ep = {ep: (start, end) for ep, start, end in left_ranges}
    right_by_ep = {ep: (start, end) for ep, start, end in right_ranges}
    all_episode_ids = sorted(set(left_by_ep).union(right_by_ep))

    for ep_idx in all_episode_ids:
        for arm_name, ep_map in [("right", right_by_ep), ("left", left_by_ep)]:
            if ep_idx not in ep_map:
                continue
            start, end = ep_map[ep_idx]
            width = max(end - start, 1e-12)
            color = cmap(ep_to_color_idx[ep_idx])
            ax.broken_barh([(start, width)], (y - 0.35, 0.7), facecolors=color, edgecolors="black")
            yticks.append(y)
            ylabels.append(f"{arm_name} ep {ep_idx}")
            y += 1

    ax.set_xlabel("Recorded timeline timestamp (s)")
    ax.set_ylabel("Arm / Episode")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Episode Start/End Span Overlap Across Datasets")

    norm = plt.Normalize(vmin=min(episode_ids), vmax=max(episode_ids))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Episode index")

    fig.tight_layout()
    plt.show()


def merge(
    left: DatasetBundle,
    right: DatasetBundle,
    output_dir: Path,
    tolerance: float,
    episode_ids: list[int],
) -> dict[str, Any]:
    task_map_left = normalize_task_map(left.tasks)
    task_map_right = normalize_task_map(right.tasks)

    if not episode_ids:
        raise ValueError("No episodes eligible for merge after wallclock metadata filtering")

    merged_rows: list[dict[str, Any]] = []
    merged_episodes: list[dict[str, Any]] = []
    merged_task_to_idx: dict[str, int] = {}

    global_index = 0
    for ep_idx in episode_ids:
        left_ep = left.data[left.data["episode_index"] == ep_idx].reset_index(drop=True)
        right_ep = right.data[right.data["episode_index"] == ep_idx].reset_index(drop=True)

        left_ts = left_ep["timestamp"].to_numpy(dtype=np.float64)
        right_ts = right_ep["timestamp"].to_numpy(dtype=np.float64)

        ia, ib = align_indices(left_ts, right_ts, tolerance=tolerance)
        if len(ia) == 0:
            raise ValueError(f"Episode {ep_idx}: no aligned frames within tolerance={tolerance}")

        # Ensure alignment quality: require near-complete overlap.
        overlap = len(ia) / min(len(left_ep), len(right_ep))
        if overlap < 0.95:
            raise ValueError(
                f"Episode {ep_idx}: poor timestamp overlap ({overlap:.3f}). "
                f"Left={len(left_ep)} Right={len(right_ep)} Matched={len(ia)}"
            )

        max_abs_delta = float(np.max(np.abs(left_ts[ia] - right_ts[ib])))

        left_task = task_map_left.get(int(left_ep.iloc[0]["task_index"]),
                                      f"left_task_{int(left_ep.iloc[0]['task_index'])}")
        right_task = task_map_right.get(int(right_ep.iloc[0]["task_index"]),
                                        f"right_task_{int(right_ep.iloc[0]['task_index'])}")
        merged_task = f"left:{left_task} | right:{right_task}"

        if merged_task not in merged_task_to_idx:
            merged_task_to_idx[merged_task] = len(merged_task_to_idx)
        merged_task_idx = merged_task_to_idx[merged_task]

        ep_from = global_index
        for k in range(len(ia)):
            lrow = left_ep.iloc[int(ia[k])]
            rrow = right_ep.iloc[int(ib[k])]
            action = np.concatenate(
                [np.asarray(lrow["action"], dtype=np.float32), np.asarray(rrow["action"], dtype=np.float32)]
            ).tolist()
            obs_state = np.concatenate(
                [
                    np.asarray(lrow["observation.state"], dtype=np.float32),
                    np.asarray(rrow["observation.state"], dtype=np.float32),
                ]
            ).tolist()
            ts = float((float(lrow["timestamp"]) + float(rrow["timestamp"])) / 2.0)

            merged_rows.append(
                {
                    "action": action,
                    "observation.state": obs_state,
                    "timestamp": np.float32(ts),
                    "frame_index": int(k),
                    "episode_index": int(ep_idx),
                    "index": int(global_index),
                    "task_index": int(merged_task_idx),
                }
            )
            global_index += 1

        ep_to = global_index
        merged_episodes.append(
            {
                "episode_index": int(ep_idx),
                "tasks": [merged_task],
                "length": int(len(ia)),
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": int(ep_from),
                "dataset_to_index": int(ep_to),
                "videos/observation.images.wrist_left/chunk_index": 0,
                "videos/observation.images.wrist_left/file_index": 0,
                "videos/observation.images.wrist_left/from_timestamp": float(left_ts[ia[0]]),
                "videos/observation.images.wrist_left/to_timestamp": float(left_ts[ia[-1]]),
                "videos/observation.images.wrist_right/chunk_index": 0,
                "videos/observation.images.wrist_right/file_index": 0,
                "videos/observation.images.wrist_right/from_timestamp": float(right_ts[ib[0]]),
                "videos/observation.images.wrist_right/to_timestamp": float(right_ts[ib[-1]]),
                "alignment/max_abs_timestamp_delta": max_abs_delta,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )

    merged_df = pd.DataFrame(merged_rows)
    episodes_df = pd.DataFrame(merged_episodes)

    merged_df.to_parquet(output_dir / "data" / "chunk-000" / "file-000.parquet", index=False)
    episodes_df.to_parquet(output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    tasks_df = pd.DataFrame({"task_index": list(merged_task_to_idx.values())}, index=list(merged_task_to_idx.keys()))
    tasks_df.to_parquet(output_dir / "meta" / "tasks.parquet")

    action_arr = np.vstack(merged_df["action"].to_numpy())
    obs_arr = np.vstack(merged_df["observation.state"].to_numpy())
    ts_arr = merged_df["timestamp"].to_numpy(dtype=np.float64)
    frame_arr = merged_df["frame_index"].to_numpy(dtype=np.float64)
    ep_arr = merged_df["episode_index"].to_numpy(dtype=np.float64)
    idx_arr = merged_df["index"].to_numpy(dtype=np.float64)
    task_arr = merged_df["task_index"].to_numpy(dtype=np.float64)

    left_action_names = left.info["features"]["action"]["names"]
    right_action_names = right.info["features"]["action"]["names"]
    left_obs_names = left.info["features"]["observation.state"]["names"]
    right_obs_names = right.info["features"]["observation.state"]["names"]

    stats = {
        "action": summarize_numeric(action_arr),
        "observation.state": summarize_numeric(obs_arr),
        "timestamp": summarize_numeric(ts_arr),
        "frame_index": summarize_numeric(frame_arr),
        "episode_index": summarize_numeric(ep_arr),
        "index": summarize_numeric(idx_arr),
        "task_index": summarize_numeric(task_arr),
        # Keep image stats from sources as placeholders for compatibility.
        "observation.images.wrist_left": left.stats.get("observation.images.wrist", {}),
        "observation.images.wrist_right": right.stats.get("observation.images.wrist", {}),
    }

    with (output_dir / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    left_image_feature = left.info["features"].get("observation.images.wrist", {})
    right_image_feature = right.info["features"].get("observation.images.wrist", {})

    features = {
        "action": {
            "dtype": "float32",
            "shape": [len(left_action_names) + len(right_action_names)],
            "names": [f"left.{n}" for n in left_action_names] + [f"right.{n}" for n in right_action_names],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(left_obs_names) + len(right_obs_names)],
            "names": [f"left.{n}" for n in left_obs_names] + [f"right.{n}" for n in right_obs_names],
        },
        "observation.images.wrist_left": left_image_feature,
        "observation.images.wrist_right": right_image_feature,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    fps = int(left.info.get("fps", right.info.get("fps", 30)))
    info = {
        "codebase_version": left.info.get("codebase_version", "v3.0"),
        "robot_type": "so100_bimanual",
        "total_episodes": int(len(episodes_df)),
        "total_frames": int(len(merged_df)),
        "total_tasks": int(len(tasks_df)),
        "chunks_size": 1000,
        "data_files_size_in_mb": float((output_dir / "data" / "chunk-000" / "file-000.parquet").stat().st_size / (1024 * 1024)),
        "video_files_size_in_mb": 0.0,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_df)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_chunk:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_chunk:03d}.mp4",
        "features": features,
    }

    with (output_dir / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    copy_video_tree(left.root, output_dir, "observation.images.wrist_left")
    copy_video_tree(right.root, output_dir, "observation.images.wrist_right")

    left_video = output_dir / "videos" / "observation.images.wrist_left" / "chunk-000" / "file-000.mp4"
    right_video = output_dir / "videos" / "observation.images.wrist_right" / "chunk-000" / "file-000.mp4"
    video_size_mb = 0.0
    if left_video.exists():
        video_size_mb += left_video.stat().st_size / (1024 * 1024)
    if right_video.exists():
        video_size_mb += right_video.stat().st_size / (1024 * 1024)

    info["video_files_size_in_mb"] = float(video_size_mb)
    with (output_dir / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    return {
        "episodes": len(episodes_df),
        "frames": len(merged_df),
        "tasks": len(tasks_df),
        "max_episode_timestamp_delta": float(episodes_df["alignment/max_abs_timestamp_delta"].max()),
    }


def resolve_input_datasets(input_dir: Path, left_name: str | None, right_name: str | None) -> tuple[Path, Path]:
    if left_name and right_name:
        left_path = input_dir / left_name
        right_path = input_dir / right_name
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError("Configured dataset subfolders do not exist")
        return left_path, right_path

    candidates = [p for p in sorted(input_dir.iterdir()) if p.is_dir() and (p / "meta" / "info.json").exists()]
    if len(candidates) != 2:
        raise ValueError(
            f"Expected exactly 2 dataset subfolders in {input_dir}, found {len(candidates)}. "
            "Use --left-name and --right-name to choose explicitly."
        )
    return candidates[0], candidates[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two LeRobot datasets into one bimanual dataset")
    parser.add_argument("input_dir", type=Path, help="Directory containing the two source LeRobot dataset folders")
    parser.add_argument("output_dir", type=Path, help="Output directory for merged dataset")
    parser.add_argument("--left-name", type=str, default=None, help="Left-arm dataset subfolder name")
    parser.add_argument("--right-name", type=str, default=None, help="Right-arm dataset subfolder name")
    parser.add_argument(
        "--timestamp-tolerance",
        type=float,
        default=1e-3,
        help="Maximum timestamp mismatch (seconds) allowed when aligning frames",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument(
        "--show-overlap-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a matplotlib window with per-episode timeline overlap before merging",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    left_dir, right_dir = resolve_input_datasets(input_dir, args.left_name, args.right_name)

    left = load_bundle(left_dir)
    right = load_bundle(right_dir)

    if left.info.get("fps") != right.info.get("fps"):
        raise ValueError(f"FPS mismatch: left={left.info.get('fps')} right={right.info.get('fps')}")

    usable_episode_ids, left_wallclock, right_wallclock, report_text = compute_episode_selection(left, right)
    if args.show_overlap_plot and usable_episode_ids:
        show_episode_overlap_plot(left_wallclock, right_wallclock, usable_episode_ids)

    ensure_clean_output(output_dir, force=args.force)
    (output_dir / "merge_episode_report.txt").write_text(report_text)
    summary = merge(
        left,
        right,
        output_dir=output_dir,
        tolerance=args.timestamp_tolerance,
        episode_ids=usable_episode_ids,
    )

    print("Merged dataset created")
    print(f"left:  {left.root}")
    print(f"right: {right.root}")
    print(f"out:   {output_dir}")
    print(f"report: {output_dir / 'merge_episode_report.txt'}")
    print(f"episodes={summary['episodes']} frames={summary['frames']} tasks={summary['tasks']}")
    print(f"max aligned timestamp delta={summary['max_episode_timestamp_delta']:.6f}s")


if __name__ == "__main__":
    main()
