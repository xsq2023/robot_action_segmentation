from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_tas.api.base import MultimodalClient
from robot_tas.cache import cache_matches, ensure_dir, read_json, write_cache_metadata, write_json
from robot_tas.schemas import GlobalCheckResult, GlobalIssue, LabeledSegment, MergedBoundary, SampledFrame, VideoMetadata
from robot_tas.windows import boundary_neighborhood


def build_boundary_contexts(
    sampled_frames: list[SampledFrame],
    boundaries: list[MergedBoundary],
    radius: int = 2,
) -> dict[int, list[SampledFrame]]:
    """Select compact boundary neighborhoods for the global check."""

    return {
        boundary.boundary_sample_index: boundary_neighborhood(sampled_frames, boundary.boundary_sample_index, radius)
        for boundary in boundaries
    }


def apply_global_issues(
    segments: list[LabeledSegment],
    boundaries: list[MergedBoundary],
    issues: list[GlobalIssue],
    confidence_threshold: float,
) -> tuple[list[LabeledSegment], list[MergedBoundary], list[GlobalIssue]]:
    """Apply safe constrained edits to the timeline."""

    updated_segments = [segment.model_copy(deep=True) for segment in segments]
    updated_boundaries = [boundary.model_copy(deep=True) for boundary in boundaries]
    applied: list[GlobalIssue] = []

    relabel_issues = [
        issue
        for issue in issues
        if issue.type == "relabel_segment" and issue.confidence >= confidence_threshold and issue.segment_id is not None and issue.new_label
    ]
    for issue in relabel_issues:
        if issue.segment_id is None or issue.segment_id >= len(updated_segments):
            continue
        current = updated_segments[issue.segment_id]
        updated_segments[issue.segment_id] = current.model_copy(
            update={
                "action_label": issue.new_label,
                "description": f"Global check relabeled this segment as {issue.new_label}.",
            }
        )
        applied.append(issue)

    merge_issues = sorted(
        [
            issue
            for issue in issues
            if issue.type == "merge_adjacent_segments" and issue.confidence >= confidence_threshold and len(issue.segment_ids) == 2
        ],
        key=lambda issue: issue.segment_ids[0],
        reverse=True,
    )
    for issue in merge_issues:
        first_id, second_id = issue.segment_ids
        if second_id != first_id + 1:
            continue
        if second_id >= len(updated_segments):
            continue
        first = updated_segments[first_id]
        second = updated_segments[second_id]
        merged = first.model_copy(
            update={
                "end_sample_index": second.end_sample_index,
                "end_frame_id": second.end_frame_id,
                "end_time": second.end_time,
                "description": first.description,
                "confidence": max(first.confidence, second.confidence),
            }
        )
        updated_segments[first_id : second_id + 1] = [merged]
        if first_id < len(updated_boundaries):
            updated_boundaries.pop(first_id)
        applied.append(issue)

    renumbered_segments: list[LabeledSegment] = []
    for new_id, segment in enumerate(updated_segments):
        renumbered_segments.append(segment.model_copy(update={"segment_id": new_id}))
    return renumbered_segments, updated_boundaries, applied


def run_global_consistency_check(
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    boundaries: list[MergedBoundary],
    segments: list[LabeledSegment],
    client: MultimodalClient,
    prompt_text: str,
    prompt_version: str,
    output_dir: Path,
    confidence_threshold: float,
    force: bool = False,
    cache_fingerprint: dict[str, Any] | None = None,
) -> tuple[GlobalCheckResult, list[LabeledSegment], list[MergedBoundary]]:
    """Run a final constrained global consistency check."""

    stage_path = output_dir / "global_check.json"
    if (
        stage_path.exists()
        and not force
        and (cache_fingerprint is None or cache_matches(stage_path, cache_fingerprint))
    ):
        cached = GlobalCheckResult.model_validate(read_json(stage_path))
        final_segments, final_boundaries, _ = apply_global_issues(
            segments=segments,
            boundaries=boundaries,
            issues=cached.issues,
            confidence_threshold=confidence_threshold,
        )
        return cached, final_segments, final_boundaries

    boundary_contexts = build_boundary_contexts(sampled_frames=sampled_frames, boundaries=boundaries)
    call = client.check_global_consistency(
        metadata=metadata,
        segments=segments,
        boundaries=boundaries,
        boundary_frames=boundary_contexts,
        prompt_text=prompt_text,
        prompt_version=prompt_version,
    )
    final_segments, final_boundaries, applied = apply_global_issues(
        segments=segments,
        boundaries=boundaries,
        issues=call.parsed.issues,
        confidence_threshold=confidence_threshold,
    )
    result = call.parsed.model_copy(update={"applied_issues": applied})
    write_json(stage_path, result.model_dump(mode="json"))
    if cache_fingerprint is not None:
        write_cache_metadata(stage_path, cache_fingerprint)
    raw_dir = ensure_dir(output_dir / "raw_api" / "global_check")
    write_json(
        raw_dir / "global_check.json",
        {"request": call.raw_request, "response": call.raw_response, "cache_key": call.cache_key},
    )
    return result, final_segments, final_boundaries
