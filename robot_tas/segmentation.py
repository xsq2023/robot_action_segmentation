from __future__ import annotations

from robot_tas.schemas import MergedBoundary, RawSegment, SampledFrame, VideoMetadata


def construct_segments(
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    boundaries: list[MergedBoundary],
) -> list[RawSegment]:
    """Construct chronological segments from merged boundaries."""

    if not sampled_frames:
        return []

    index_to_position = {frame.sample_index: position for position, frame in enumerate(sampled_frames)}
    segments: list[RawSegment] = []
    start_position = 0

    for boundary in sorted(boundaries, key=lambda item: item.boundary_sample_index):
        boundary_position = index_to_position[boundary.boundary_sample_index]
        if boundary_position <= start_position:
            continue
        start_frame = sampled_frames[start_position]
        segments.append(
            RawSegment(
                segment_id=len(segments),
                start_sample_index=start_frame.sample_index,
                end_sample_index=boundary.boundary_sample_index,
                start_frame_id=start_frame.original_frame_id,
                end_frame_id=boundary.boundary_frame_id,
                start_time=start_frame.timestamp_seconds,
                end_time=boundary.boundary_time,
            )
        )
        start_position = boundary_position

    last_start = sampled_frames[start_position]
    segments.append(
        RawSegment(
            segment_id=len(segments),
            start_sample_index=last_start.sample_index,
            end_sample_index=sampled_frames[-1].sample_index,
            start_frame_id=last_start.original_frame_id,
            end_frame_id=metadata.total_frames - 1,
            start_time=last_start.timestamp_seconds,
            end_time=metadata.duration_seconds,
        )
    )
    return segments


def segment_frames(
    sampled_frames: list[SampledFrame],
    segment: RawSegment,
    is_last_segment: bool,
) -> list[SampledFrame]:
    """Resolve sampled frames belonging to a segment."""

    index_to_position = {frame.sample_index: position for position, frame in enumerate(sampled_frames)}
    start_position = index_to_position[segment.start_sample_index]
    end_position = index_to_position[segment.end_sample_index]
    end_exclusive = end_position + 1 if is_last_segment else max(start_position + 1, end_position)
    frames = sampled_frames[start_position:end_exclusive]
    return frames or [sampled_frames[start_position]]

