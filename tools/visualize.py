#!/usr/bin/env python3
"""LeRobot Dataset v3 viewer.

Usage:
  python scripts/visualize_lerobot_v3.py --dataset data/RowData2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


@dataclass
class VideoStream:
    key: str
    fps: float
    from_timestamp: float
    cap: cv2.VideoCapture


class LerobotDatasetV3:
    def __init__(self, root: Path, max_videos: int | None = 3):
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text())

        episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        data_path = root / "data" / "chunk-000" / "file-000.parquet"

        self.episodes = pd.read_parquet(episodes_path)
        self.data = pd.read_parquet(data_path)

        self.video_keys = [
            key for key, value in self.info["features"].items() if value.get("dtype") == "video"
        ]
        if max_videos is not None and max_videos > 0:
            self.video_keys = self.video_keys[:max_videos]

        action_feature = self.info["features"].get("action", {})
        self.action_names = action_feature.get("names") or [f"action_{i}" for i in range(action_feature.get("shape", [0])[0])]
        self.dataset_fps = float(self.info.get("fps", 30))
        self.video_path_template = self.info.get(
            "video_path", "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_chunk:03d}.mp4"
        )

    def episode_indices(self) -> list[int]:
        return self.episodes["episode_index"].astype(int).tolist()

    def episode_row(self, episode_index: int) -> pd.Series:
        row = self.episodes[self.episodes["episode_index"] == episode_index]
        if row.empty:
            raise ValueError(f"Episode {episode_index} not found")
        return row.iloc[0]

    def episode_data(self, episode_row: pd.Series) -> pd.DataFrame:
        start = int(episode_row["dataset_from_index"])
        end = int(episode_row["dataset_to_index"])
        return self.data.iloc[start:end].reset_index(drop=True)

    def video_path(self, video_key: str, episode_row: pd.Series) -> Path:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        chunk_index = int(episode_row[chunk_col]) if chunk_col in episode_row else 0
        file_index = int(episode_row[file_col]) if file_col in episode_row else chunk_index

        try:
            rel = self.video_path_template.format(video_key=video_key, episode_chunk=chunk_index, file_index=file_index)
            path = self.root / rel
        except Exception:
            path = self.root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"

        if not path.exists():
            fallback = self.root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            path = fallback
        return path


class Viewer(QMainWindow):
    ACTION_COLORS = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
    ]

    def __init__(self, dataset: LerobotDatasetV3):
        super().__init__()
        self.dataset = dataset

        self.episode_df: pd.DataFrame | None = None
        self.episode_row: pd.Series | None = None
        self.current_frame = 0
        self.playing = False
        self.video_streams: list[VideoStream] = []
        self.action_cursor = None
        self.action_lines: list[Any] = []
        self.videos_visible = True
        self.playback_speed = 1.0

        self.setWindowTitle(f"LeRobot v3 Viewer - {dataset.root}")
        self.resize(1500, 950)

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Episode:"))

        self.episode_combo = QComboBox()
        for idx in self.dataset.episode_indices():
            self.episode_combo.addItem(str(idx), idx)
        self.episode_combo.currentIndexChanged.connect(self.on_episode_changed)
        controls.addWidget(self.episode_combo)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause)
        controls.addWidget(self.pause_btn)

        self.step_btn = QPushButton("Step")
        self.step_btn.clicked.connect(self.step_frame)
        controls.addWidget(self.step_btn)

        controls.addWidget(QLabel("Speed:"))
        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self.speed_buttons: dict[float, QPushButton] = {}
        for speed in (1.0, 2.0, 4.0):
            btn = QPushButton(f"{int(speed)}x")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, value=speed: self.set_playback_speed(value))
            self.speed_group.addButton(btn)
            self.speed_buttons[speed] = btn
            controls.addWidget(btn)
        self.speed_buttons[self.playback_speed].setChecked(True)

        self.video_toggle_btn = QPushButton("Hide Videos")
        self.video_toggle_btn.clicked.connect(self.toggle_videos_visibility)
        controls.addWidget(self.video_toggle_btn)

        controls.addStretch(1)
        main_layout.addLayout(controls)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        main_layout.addWidget(self.slider)

        self.frame_label = QLabel("Frame: 0")
        main_layout.addWidget(self.frame_label)

        self.video_grid = QGridLayout()
        self.video_labels: list[QLabel] = []
        self.video_source_labels: list[QLabel] = []
        self._rebuild_video_panels()
        main_layout.addLayout(self.video_grid)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(title="Actions")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        main_layout.addWidget(self.plot, stretch=1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self._update_timer_interval()

        self.load_episode(self.dataset.episode_indices()[0])

    def _rebuild_video_panels(self):
        while self.video_grid.count():
            item = self.video_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.video_labels.clear()
        self.video_source_labels.clear()

        for i, key in enumerate(self.dataset.video_keys):
            source = QLabel(f"Video {i+1}: {key}")
            source.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.video_source_labels.append(source)
            self.video_grid.addWidget(source, i * 2, 0, 1, 2)

            panel = QLabel("No stream")
            panel.setMinimumSize(320, 180)
            panel.setAlignment(Qt.AlignCenter)
            panel.setStyleSheet("background-color: black; color: white;")
            self.video_labels.append(panel)
            self.video_grid.addWidget(panel, i * 2 + 1, 0, 1, 2)

        self._apply_video_visibility()

    def _apply_video_visibility(self):
        for label in self.video_source_labels:
            label.setVisible(self.videos_visible)
        for panel in self.video_labels:
            panel.setVisible(self.videos_visible)
        self.video_toggle_btn.setText("Hide Videos" if self.videos_visible else "Show Videos")

    def _clear_video_streams(self):
        for stream in self.video_streams:
            stream.cap.release()
        self.video_streams.clear()

    def load_episode(self, episode_index: int):
        self.pause()
        self._clear_video_streams()
        self._rebuild_video_panels()

        self.episode_row = self.dataset.episode_row(episode_index)
        self.episode_df = self.dataset.episode_data(self.episode_row)
        self.current_frame = 0

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, len(self.episode_df) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._load_video_streams()
        self._draw_actions()
        self.render_frame(0)

    def _load_video_streams(self):
        assert self.episode_row is not None
        for key in self.dataset.video_keys:
            path = self.dataset.video_path(key, self.episode_row)
            if not path.exists():
                continue

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = float(self.dataset.info["features"][key].get("info", {}).get("video.fps", self.dataset.dataset_fps))

            from_ts_col = f"videos/{key}/from_timestamp"
            from_ts = float(self.episode_row[from_ts_col]) if from_ts_col in self.episode_row else 0.0
            self.video_streams.append(VideoStream(key=key, fps=fps, from_timestamp=from_ts, cap=cap))

    def _draw_actions(self):
        self.plot.clear()
        self.action_lines = []

        if self.episode_df is None or len(self.episode_df) == 0:
            return

        actions = np.stack(self.episode_df["action"].to_numpy())
        xs = np.arange(len(actions))

        for i in range(actions.shape[1]):
            name = self.dataset.action_names[i] if i < len(self.dataset.action_names) else f"action_{i}"
            color = self.ACTION_COLORS[i % len(self.ACTION_COLORS)]
            pen = pg.mkPen(color=color, width=1.2)
            line = self.plot.plot(xs, actions[:, i], pen=pen, name=name)
            self.action_lines.append(line)

        self.action_cursor = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen("y", width=2))
        self.plot.addItem(self.action_cursor)

    def on_episode_changed(self):
        episode_index = self.episode_combo.currentData()
        if episode_index is None:
            return
        self.load_episode(int(episode_index))

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("Playing")
            self.timer.start()
        else:
            self.play_btn.setText("Play")
            self.timer.stop()

    def pause(self):
        self.playing = False
        self.play_btn.setText("Play")
        self.timer.stop()

    def toggle_videos_visibility(self):
        self.videos_visible = not self.videos_visible
        self._apply_video_visibility()

    def set_playback_speed(self, speed: float):
        self.playback_speed = speed
        self._update_timer_interval()
        button = self.speed_buttons.get(speed)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def _update_timer_interval(self):
        interval_ms = int(1000 / max(1.0, self.dataset.dataset_fps))
        self.timer.setInterval(max(1, interval_ms))

    def on_tick(self):
        if not self.step_frame(step=int(self.playback_speed)):
            self.pause()

    def step_frame(self, step: int = 1) -> bool:
        if self.episode_df is None:
            return False
        if self.current_frame >= len(self.episode_df) - 1:
            return False
        self.current_frame = min(self.current_frame + max(1, step), len(self.episode_df) - 1)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.render_frame(self.current_frame)
        return self.current_frame < len(self.episode_df) - 1

    def on_slider_changed(self, value: int):
        self.current_frame = value
        self.render_frame(self.current_frame)

    def render_frame(self, idx: int):
        if self.episode_df is None or idx < 0 or idx >= len(self.episode_df):
            return

        row = self.episode_df.iloc[idx]
        ts0 = float(self.episode_df.iloc[0]["timestamp"])
        rel_ts = float(row["timestamp"]) - ts0

        for i, label in enumerate(self.video_labels):
            if i >= len(self.video_streams):
                self.video_source_labels[i].setText(f"Video {i+1}: N/A")
                label.setText("No stream")
                label.setPixmap(QPixmap())
                continue

            stream = self.video_streams[i]
            self.video_source_labels[i].setText(f"Video {i+1}: {stream.key}")
            frame_idx = int(round((stream.from_timestamp + rel_ts) * stream.fps))
            stream.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ok, frame = stream.cap.read()
            if not ok:
                label.setText(f"{stream.key}: frame unavailable")
                label.setPixmap(QPixmap())
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pix)
            label.setToolTip(stream.key)

        if self.action_cursor is not None:
            self.action_cursor.setValue(idx)

        self.frame_label.setText(f"Frame: {idx} / {len(self.episode_df)-1} | t={rel_ts:.3f}s")

    def closeEvent(self, event):
        self.pause()
        self._clear_video_streams()
        return super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeRobot v3 dataset visualizer")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to LeRobot dataset root")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=3,
        help="Limit number of video streams loaded at startup (<=0 loads all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_videos = None if args.max_videos <= 0 else args.max_videos
    dataset = LerobotDatasetV3(args.dataset, max_videos=max_videos)

    app = QApplication([])
    win = Viewer(dataset)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
