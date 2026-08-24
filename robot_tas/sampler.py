from __future__ import annotations

from collections import deque
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from robot_tas.cache import ensure_dir, sha256_file
from robot_tas.schemas import SampledFrame, VideoMetadata


def compute_sample_frame_ids(total_frames: int, native_fps: float, sample_fps: float) -> list[int]:
    """Compute monotonic sample frame IDs without losing original frame references."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if native_fps <= 0.0 or sample_fps <= 0.0:
        raise ValueError("FPS values must be positive")

    frame_ids: list[int] = []
    sample_index = 0
    while True:
        frame_id = int(round(sample_index * native_fps / sample_fps))
        if frame_id >= total_frames:
            break
        if not frame_ids or frame_id > frame_ids[-1]:
            frame_ids.append(frame_id)
        sample_index += 1

    if not frame_ids:
        return [0]
    return frame_ids


def build_ffmpeg_select_expression(frame_ids: list[int]) -> str:
    """Build an ffmpeg select expression that keeps only the requested frame IDs."""

    if not frame_ids:
        raise ValueError("frame_ids must not be empty")
    return "select=" + "+".join(f"eq(n\\,{frame_id})" for frame_id in frame_ids)


def _sample_from_saved_images(
    image_paths: list[Path],
    target_frame_ids: list[int],
    metadata: VideoMetadata,
    sampled_frames_dir: Path,
) -> list[SampledFrame]:
    sampled_frames: list[SampledFrame] = []
    previous_gray: np.ndarray | None = None

    for sample_index, (temp_image_path, frame_id) in enumerate(zip(image_paths, target_frame_ids)):
        frame = cv2.imread(str(temp_image_path))
        if frame is None:
            raise RuntimeError(f"Failed to decode sampled image: {temp_image_path}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if previous_gray is not None:
            diff = cv2.absdiff(previous_gray, gray)
            motion_score = float(np.mean(diff) / 255.0)
        previous_gray = gray

        filename = f"sample_{sample_index:06d}_frame_{frame_id:06d}.jpg"
        final_image_path = sampled_frames_dir / filename
        if temp_image_path.resolve() != final_image_path.resolve():
            if not cv2.imwrite(str(final_image_path), frame):
                raise RuntimeError(f"Failed to write sampled frame image: {final_image_path}")
        sampled_frames.append(
            SampledFrame(
                sample_index=sample_index,
                original_frame_id=frame_id,
                timestamp_seconds=frame_id / metadata.fps,
                image_path=str(Path("sampled_frames") / filename),
                image_sha256=sha256_file(final_image_path),
                motion_score=motion_score,
                mean_luma=float(np.mean(gray) / 255.0),
            )
        )
    return sampled_frames


def _sample_video_frames_opencv(
    video_path: Path,
    metadata: VideoMetadata,
    sampled_frames_dir: Path,
    target_frame_ids: list[int],
) -> list[SampledFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    pending_frame_ids = deque(target_frame_ids)
    extracted_paths: list[Path] = []
    frame_id = 0

    while pending_frame_ids:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id != pending_frame_ids[0]:
            frame_id += 1
            continue

        temp_image_path = sampled_frames_dir / f".opencv_sample_{len(extracted_paths):06d}.jpg"
        if not cv2.imwrite(str(temp_image_path), frame):
            raise RuntimeError(f"Failed to write temporary sampled frame image: {temp_image_path}")
        extracted_paths.append(temp_image_path)
        pending_frame_ids.popleft()
        frame_id += 1

    cap.release()

    if len(extracted_paths) != len(target_frame_ids):
        for path in extracted_paths:
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"OpenCV extracted {len(extracted_paths)} of {len(target_frame_ids)} requested frames from {video_path}"
        )

    sampled_frames = _sample_from_saved_images(
        image_paths=extracted_paths,
        target_frame_ids=target_frame_ids,
        metadata=metadata,
        sampled_frames_dir=sampled_frames_dir,
    )
    for path in extracted_paths:
        path.unlink(missing_ok=True)
    return sampled_frames


def _sample_video_frames_ffmpeg(
    video_path: Path,
    metadata: VideoMetadata,
    sampled_frames_dir: Path,
    target_frame_ids: list[int],
) -> list[SampledFrame]:
    with tempfile.TemporaryDirectory(prefix="robot_tas_ffmpeg_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        output_pattern = temp_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            build_ffmpeg_select_expression(target_frame_ids),
            "-vsync",
            "0",
            "-q:v",
            "2",
            str(output_pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg sampling failed for "
                f"{video_path}: {result.stderr.strip() or result.stdout.strip() or 'unknown ffmpeg error'}"
            )

        extracted_paths = sorted(temp_dir.glob("frame_*.jpg"))
        if len(extracted_paths) != len(target_frame_ids):
            raise RuntimeError(
                f"ffmpeg extracted {len(extracted_paths)} of {len(target_frame_ids)} requested frames from {video_path}"
            )

        return _sample_from_saved_images(
            image_paths=extracted_paths,
            target_frame_ids=target_frame_ids,
            metadata=metadata,
            sampled_frames_dir=sampled_frames_dir,
        )


def sample_video_frames(video_path: Path, metadata: VideoMetadata, output_dir: Path) -> list[SampledFrame]:
    """Decode and save sampled frames while preserving original frame IDs."""

    sampled_frames_dir = ensure_dir(output_dir / "sampled_frames")
    target_frame_ids = compute_sample_frame_ids(metadata.total_frames, metadata.fps, metadata.sample_fps)

    try:
        sampled_frames = _sample_video_frames_opencv(
            video_path=video_path,
            metadata=metadata,
            sampled_frames_dir=sampled_frames_dir,
            target_frame_ids=target_frame_ids,
        )
    except RuntimeError as error:
        sampled_frames = _sample_video_frames_ffmpeg(
            video_path=video_path,
            metadata=metadata,
            sampled_frames_dir=sampled_frames_dir,
            target_frame_ids=target_frame_ids,
        )
        if not sampled_frames:
            raise RuntimeError(f"No sampled frames were decoded from video: {video_path}") from error

    if not sampled_frames:
        raise RuntimeError(f"No sampled frames were decoded from video: {video_path}")
    return sampled_frames
