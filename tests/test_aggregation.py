from robot_tas.aggregation import merge_verified_boundaries, weighted_median_boundary
from robot_tas.schemas import VerifiedBoundary


def _verified(sample_index: int, confidence: float, proposal_id: str) -> VerifiedBoundary:
    return VerifiedBoundary(
        proposal_id=proposal_id,
        window_id=0,
        status="accept",
        original_boundary_sample_index=sample_index,
        verified_boundary_sample_index=sample_index,
        verified_original_frame_id=sample_index * 15,
        verified_timestamp=sample_index * 0.5,
        before_action="reach_for_workspace_object",
        after_action="active_manipulation",
        transition_type="reach_for_workspace_object_to_active_manipulation",
        visual_evidence=["peak"],
        confidence=confidence,
    )


def test_weighted_median_boundary_selects_real_sample() -> None:
    selected = weighted_median_boundary(
        [_verified(10, 0.6, "a"), _verified(11, 0.9, "b"), _verified(12, 0.4, "c")]
    )
    assert selected.verified_boundary_sample_index == 11


def test_merge_verified_boundaries_clusters_compatible_neighbors() -> None:
    merged = merge_verified_boundaries(
        verified_boundaries=[_verified(10, 0.6, "a"), _verified(11, 0.9, "b"), _verified(30, 0.8, "c")],
        tolerance=2,
        min_boundary_confidence=0.5,
        min_segment_samples=2,
        total_sample_count=40,
    )
    assert len(merged) == 2
    assert merged[0].boundary_sample_index == 11
    assert merged[1].boundary_sample_index == 30
