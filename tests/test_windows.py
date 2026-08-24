from robot_tas.schemas import SampledFrame
from robot_tas.windows import build_sliding_windows


def _frame(sample_index: int) -> SampledFrame:
    return SampledFrame(
        sample_index=sample_index,
        original_frame_id=sample_index * 15,
        timestamp_seconds=sample_index * 0.5,
        image_path=f"sampled_frames/sample_{sample_index:06d}.jpg",
        image_sha256="deadbeef",
        motion_score=0.0,
        mean_luma=0.5,
    )


def test_build_sliding_windows_uses_overlap() -> None:
    frames = [_frame(index) for index in range(24)]
    windows = build_sliding_windows(sampled_frames=frames, window_size=16, stride=8)
    assert [window.start_sample_index for window in windows] == [0, 8]
    assert [window.end_sample_index for window in windows] == [15, 23]

