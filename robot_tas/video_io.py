from __future__ import annotations

import logging
from pathlib import Path

import cv2

from robot_tas.schemas import VideoMetadata


LOGGER = logging.getLogger(__name__)


def _count_frames_by_decode(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    frame_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frame_count += 1
    cap.release()
    return frame_count


def read_video_metadata(video_path: Path, sample_fps: float) -> VideoMetadata:
    """Read exact video metadata and fall back to decoding when needed."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0.0:
        raise ValueError(f"Invalid FPS reported by OpenCV for video: {video_path}")

    sampling_note = "Used OpenCV metadata for frame count."
    if total_frames <= 0:
        total_frames = _count_frames_by_decode(video_path)
        sampling_note = "Frame count recovered by full decode because metadata was unavailable."
        LOGGER.warning("Recovered frame count by decoding video: %s", video_path)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size reported by OpenCV for video: {video_path}")

    duration_seconds = total_frames / fps
    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        sample_fps=sample_fps,
        sampling_note=sampling_note,
    )

