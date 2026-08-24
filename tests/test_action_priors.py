from robot_tas.action_priors import align_boundaries_with_prior, build_equal_prior_boundaries, label_segments_with_prior
from robot_tas.schemas import MergedBoundary, RawSegment, SampledFrame, VideoMetadata


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        path="demo.mp4",
        fps=30.0,
        total_frames=601,
        duration_seconds=20.0333333333,
        width=640,
        height=480,
        sample_fps=2.0,
    )


def _frame(sample_index: int) -> SampledFrame:
    return SampledFrame(
        sample_index=sample_index,
        original_frame_id=sample_index * 15,
        timestamp_seconds=sample_index * 0.5,
        image_path=f"sampled_frames/sample_{sample_index:06d}.jpg",
        image_sha256="abc123",
    )


def test_build_equal_prior_boundaries_snaps_to_real_samples() -> None:
    frames = [_frame(index) for index in range(41)]
    boundaries = build_equal_prior_boundaries(
        metadata=_metadata(),
        sampled_frames=frames,
        preset_actions=["pick", "place", "pick", "place", "pick", "place"],
    )
    assert len(boundaries) == 5
    assert all(boundary.boundary_sample_index in {frame.sample_index for frame in frames} for boundary in boundaries)
    assert boundaries[0].before_action == "pick"
    assert boundaries[0].after_action == "place"


def test_build_ratio_prior_boundaries_uses_supplied_ratios() -> None:
    frames = [_frame(index) for index in range(41)]
    boundaries = build_equal_prior_boundaries(
        metadata=_metadata(),
        sampled_frames=frames,
        preset_actions=["pick", "place", "pick"],
        boundary_ratios=[0.25, 0.75],
    )
    assert [boundary.boundary_sample_index for boundary in boundaries] == [10, 30]


def test_label_segments_with_prior_uses_expected_labels() -> None:
    segments = [
        RawSegment(
            segment_id=0,
            start_sample_index=0,
            end_sample_index=5,
            start_frame_id=0,
            end_frame_id=75,
            start_time=0.0,
            end_time=2.5,
        ),
        RawSegment(
            segment_id=1,
            start_sample_index=5,
            end_sample_index=10,
            start_frame_id=75,
            end_frame_id=150,
            start_time=2.5,
            end_time=5.0,
        ),
    ]
    labeled = label_segments_with_prior(segments, ["pick", "place"])
    assert [segment.action_label for segment in labeled] == ["pick", "place"]


def test_align_boundaries_with_prior_keeps_expected_transition_order() -> None:
    boundaries = [
        MergedBoundary(
            boundary_sample_index=8,
            boundary_frame_id=120,
            boundary_time=4.0,
            before_action="pick",
            after_action="place",
            transition_type="pick_to_place",
            visual_evidence=["too early"],
            supporting_windows=[0],
            confidence=0.9,
            source_proposal_ids=["early"],
        ),
        MergedBoundary(
            boundary_sample_index=24,
            boundary_frame_id=360,
            boundary_time=12.0,
            before_action="pick",
            after_action="place",
            transition_type="pick_to_place",
            visual_evidence=["near target"],
            supporting_windows=[1],
            confidence=0.8,
            source_proposal_ids=["near"],
        ),
        MergedBoundary(
            boundary_sample_index=33,
            boundary_frame_id=495,
            boundary_time=16.5,
            before_action="place",
            after_action="pick",
            transition_type="place_to_pick",
            visual_evidence=["next"],
            supporting_windows=[2],
            confidence=0.8,
            source_proposal_ids=["next"],
        ),
    ]

    aligned = align_boundaries_with_prior(
        boundaries=boundaries,
        metadata=_metadata(),
        sampled_frames=[_frame(index) for index in range(41)],
        preset_actions=["pick", "place", "pick"],
        boundary_ratios=[0.58, 0.80],
    )

    assert [boundary.boundary_frame_id for boundary in aligned] == [360, 495]
