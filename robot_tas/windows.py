from __future__ import annotations

from robot_tas.schemas import SampledFrame, Window


def build_sliding_windows(
    sampled_frames: list[SampledFrame], window_size: int, stride: int
) -> list[Window]:
    """Build overlapping chronological windows over sampled frames."""

    if not sampled_frames:
        return []
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    frame_count = len(sampled_frames)
    if frame_count <= window_size:
        return [
            Window(
                window_id=0,
                start_sample_index=sampled_frames[0].sample_index,
                end_sample_index=sampled_frames[-1].sample_index,
                frames=sampled_frames,
            )
        ]

    starts = list(range(0, frame_count - window_size + 1, stride))
    tail_start = frame_count - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)

    windows: list[Window] = []
    for window_id, start in enumerate(sorted(set(starts))):
        window_frames = sampled_frames[start : start + window_size]
        windows.append(
            Window(
                window_id=window_id,
                start_sample_index=window_frames[0].sample_index,
                end_sample_index=window_frames[-1].sample_index,
                frames=window_frames,
            )
        )
    return windows


def boundary_neighborhood(
    sampled_frames: list[SampledFrame], boundary_sample_index: int, radius: int
) -> list[SampledFrame]:
    """Return a compact neighborhood around a candidate boundary."""

    index_to_position = {frame.sample_index: position for position, frame in enumerate(sampled_frames)}
    if boundary_sample_index not in index_to_position:
        raise KeyError(f"Unknown boundary sample index: {boundary_sample_index}")

    center = index_to_position[boundary_sample_index]
    start = max(0, center - radius)
    end = min(len(sampled_frames), center + radius + 1)
    return sampled_frames[start:end]
