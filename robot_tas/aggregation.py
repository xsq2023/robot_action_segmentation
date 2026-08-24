from __future__ import annotations

from statistics import fmean

from robot_tas.normalization import semantics_compatible
from robot_tas.schemas import MergedBoundary, VerifiedBoundary


def weighted_median_boundary(boundaries: list[VerifiedBoundary]) -> VerifiedBoundary:
    """Select the actual sampled boundary corresponding to the weighted median."""

    ordered = sorted(
        (
            item
            for item in boundaries
            if item.verified_boundary_sample_index is not None and item.status != "reject"
        ),
        key=lambda item: item.verified_boundary_sample_index,
    )
    if not ordered:
        raise ValueError("weighted_median_boundary requires at least one usable boundary")

    total_weight = sum(item.confidence for item in ordered)
    cumulative = 0.0
    for item in ordered:
        cumulative += item.confidence
        if cumulative >= total_weight / 2.0:
            return item
    return ordered[-1]


def _cluster_boundaries(
    verified_boundaries: list[VerifiedBoundary], tolerance: int
) -> list[list[VerifiedBoundary]]:
    usable = sorted(
        (
            boundary
            for boundary in verified_boundaries
            if boundary.status != "reject" and boundary.verified_boundary_sample_index is not None
        ),
        key=lambda boundary: boundary.verified_boundary_sample_index,
    )
    if not usable:
        return []

    clusters: list[list[VerifiedBoundary]] = [[usable[0]]]
    for boundary in usable[1:]:
        current_cluster = clusters[-1]
        last = current_cluster[-1]
        close_in_time = (
            boundary.verified_boundary_sample_index - last.verified_boundary_sample_index <= tolerance
        )
        compatible = semantics_compatible(
            last.before_action,
            last.after_action,
            boundary.before_action,
            boundary.after_action,
            last.transition_type,
            boundary.transition_type,
        )
        if close_in_time and compatible:
            current_cluster.append(boundary)
        else:
            clusters.append([boundary])
    return clusters


def _aggregate_cluster(cluster: list[VerifiedBoundary]) -> MergedBoundary:
    selected = weighted_median_boundary(cluster)
    evidence: list[str] = []
    for item in cluster:
        for note in item.visual_evidence:
            if note not in evidence:
                evidence.append(note)

    confidence = min(0.99, fmean(item.confidence for item in cluster) + min(0.08, 0.03 * (len(cluster) - 1)))
    return MergedBoundary(
        boundary_sample_index=selected.verified_boundary_sample_index or selected.original_boundary_sample_index,
        boundary_frame_id=selected.verified_original_frame_id or 0,
        boundary_time=selected.verified_timestamp or 0.0,
        before_action=selected.before_action,
        after_action=selected.after_action,
        transition_type=selected.transition_type,
        visual_evidence=evidence,
        supporting_windows=sorted({item.window_id for item in cluster}),
        confidence=confidence,
        source_proposal_ids=[item.proposal_id for item in cluster],
    )


def _prune_short_segments(
    boundaries: list[MergedBoundary],
    total_sample_count: int,
    min_segment_samples: int,
) -> list[MergedBoundary]:
    if not boundaries:
        return []

    kept: list[MergedBoundary] = []
    for boundary in sorted(boundaries, key=lambda item: item.boundary_sample_index):
        if boundary.boundary_sample_index <= 0:
            if boundary.confidence >= 0.9:
                kept.append(boundary)
            continue
        if not kept:
            if boundary.boundary_sample_index < min_segment_samples and boundary.confidence < 0.85:
                continue
            kept.append(boundary)
            continue

        gap = boundary.boundary_sample_index - kept[-1].boundary_sample_index
        if gap < min_segment_samples:
            if boundary.confidence > kept[-1].confidence:
                kept[-1] = boundary
            continue
        kept.append(boundary)

    if kept:
        tail_gap = (total_sample_count - 1) - kept[-1].boundary_sample_index
        if tail_gap < min_segment_samples and kept[-1].confidence < 0.85:
            kept.pop()
    return kept


def merge_verified_boundaries(
    verified_boundaries: list[VerifiedBoundary],
    tolerance: int,
    min_boundary_confidence: float,
    min_segment_samples: int,
    total_sample_count: int,
) -> list[MergedBoundary]:
    """Cluster nearby compatible verified boundaries deterministically."""

    clusters = _cluster_boundaries(verified_boundaries, tolerance=tolerance)
    merged: list[MergedBoundary] = []
    for cluster in clusters:
        candidate = _aggregate_cluster(cluster)
        if candidate.confidence < min_boundary_confidence and len(cluster) == 1:
            continue
        merged.append(candidate)
    return _prune_short_segments(
        boundaries=merged,
        total_sample_count=total_sample_count,
        min_segment_samples=min_segment_samples,
    )
