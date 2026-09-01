from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from robot_tas.api.base import MultimodalClient
from robot_tas.cache import cache_matches, ensure_dir, read_json, write_cache_metadata, write_json
from robot_tas.schemas import LabeledSegment, RawSegment, SampledFrame
from robot_tas.segmentation import segment_frames


LOGGER = logging.getLogger(__name__)


def select_representative_frames(
    sampled_frames: list[SampledFrame],
    segment: RawSegment,
    is_last_segment: bool,
) -> list[SampledFrame]:
    """Select a compact representative subset for semantic labeling."""

    frames = segment_frames(sampled_frames=sampled_frames, segment=segment, is_last_segment=is_last_segment)
    if len(frames) <= 5:
        return frames

    indices = [0, round((len(frames) - 1) * 0.25), round((len(frames) - 1) * 0.5), round((len(frames) - 1) * 0.75), len(frames) - 1]
    selected: list[SampledFrame] = []
    seen: set[int] = set()
    for index in indices:
        frame = frames[index]
        if frame.sample_index not in seen:
            selected.append(frame)
            seen.add(frame.sample_index)
    return selected


def run_segment_labeling(
    segments: list[RawSegment],
    sampled_frames: list[SampledFrame],
    client: MultimodalClient,
    prompt_text: str,
    prompt_version: str,
    output_dir: Path,
    force: bool = False,
    cache_fingerprint: dict[str, Any] | None = None,
) -> list[LabeledSegment]:
    """Label each grounded segment using representative frames."""

    stage_path = output_dir / "segments_labeled.json"
    can_reuse_stage = (
        stage_path.exists()
        and not force
        and (cache_fingerprint is None or cache_matches(stage_path, cache_fingerprint))
    )
    if can_reuse_stage:
        return [LabeledSegment.model_validate(item) for item in read_json(stage_path)]

    can_reuse_items = not force and cache_fingerprint is None
    cache_dir = ensure_dir(output_dir / "cache" / "segments_labeled")
    raw_dir = ensure_dir(output_dir / "raw_api" / "segments_labeled")
    labeled_segments: list[LabeledSegment] = []

    for segment in segments:
        item_path = cache_dir / f"segment_{segment.segment_id:04d}.json"
        if item_path.exists() and can_reuse_items:
            labeled_segment = LabeledSegment.model_validate(read_json(item_path))
        else:
            representatives = select_representative_frames(
                sampled_frames=sampled_frames,
                segment=segment,
                is_last_segment=segment.segment_id == len(segments) - 1,
            )
            call = client.label_segment(
                segment=segment,
                representative_frames=representatives,
                total_segments=len(segments),
                prompt_text=prompt_text,
                prompt_version=prompt_version,
            )
            labeled_segment = LabeledSegment(
                **segment.model_dump(mode="json"),
                **call.parsed.model_dump(mode="json"),
            )
            write_json(item_path, labeled_segment.model_dump(mode="json"))
            write_json(
                raw_dir / f"segment_{segment.segment_id:04d}.json",
                {"request": call.raw_request, "response": call.raw_response, "cache_key": call.cache_key},
            )
            LOGGER.info("Labeled segment %s as %s", segment.segment_id, labeled_segment.action_label)
        labeled_segments.append(labeled_segment)

    write_json(stage_path, [segment.model_dump(mode="json") for segment in labeled_segments])
    if cache_fingerprint is not None:
        write_cache_metadata(stage_path, cache_fingerprint)
    return labeled_segments
