from robot_tas.schemas import MergedBoundary, SampledFrame, VideoMetadata
from robot_tas.segmentation import construct_segments


def _frame(sample_index: int) -> SampledFrame:
    return SampledFrame(
        sample_index=sample_index,
        original_frame_id=sample_index * 15,
        timestamp_seconds=sample_index * 0.5,
        image_path=f"sampled_frames/sample_{sample_index:06d}.jpg",
        image_sha256="abc123",
        motion_score=0.0,
        mean_luma=0.5,
    )


def test_construct_segments_respects_boundary_as_new_segment_start() -> None:
    metadata = VideoMetadata(
        path="demo.mp4",
        fps=30.0,
        total_frames=91,
        duration_seconds=3.0333333333,
        width=640,
        height=360,
        sample_fps=2.0,
    )
    sampled_frames = [_frame(index) for index in range(6)]
    boundaries = [
        MergedBoundary(
            boundary_sample_index=2,
            boundary_frame_id=30,
            boundary_time=1.0,
            before_action="reach_for_workspace_object",
            after_action="active_manipulation",
            transition_type="reach_for_workspace_object_to_active_manipulation",
            visual_evidence=["peak"],
            supporting_windows=[0],
            confidence=0.9,
            source_proposal_ids=["a"],
        )
    ]
    segments = construct_segments(metadata=metadata, sampled_frames=sampled_frames, boundaries=boundaries)
    assert len(segments) == 2
    assert segments[0].start_sample_index == 0
    assert segments[0].end_sample_index == 2
    assert segments[1].start_sample_index == 2

