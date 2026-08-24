from robot_tas.schemas import SampledFrame
from robot_tas.visual_reference import frame_reference_line, transition_reference_lines


def _frame(sample_index: int, motion_score: float, mean_luma: float) -> SampledFrame:
    return SampledFrame(
        sample_index=sample_index,
        original_frame_id=sample_index * 10,
        timestamp_seconds=sample_index / 3.0,
        image_path=f"sampled_frames/sample_{sample_index:06d}.jpg",
        image_sha256=f"sha-{sample_index}",
        motion_score=motion_score,
        mean_luma=mean_luma,
    )


def test_frame_reference_line_includes_visual_cues() -> None:
    line = frame_reference_line(_frame(3, 0.125, 0.5))

    assert "sample_index=3" in line
    assert "original_frame_id=30" in line
    assert "motion_score=0.1250" in line
    assert "mean_luma=0.5000" in line


def test_transition_reference_lines_rank_visual_changes() -> None:
    frames = [_frame(0, 0.0, 0.2), _frame(1, 0.05, 0.21), _frame(2, 0.3, 0.6)]

    lines = transition_reference_lines(frames, top_k=1)

    assert len(lines) == 1
    assert "sample_index=2" in lines[0]
    assert "visual_change_score=" in lines[0]
