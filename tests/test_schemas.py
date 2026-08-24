import pytest

from robot_tas.schemas import VideoMetadata


def test_video_metadata_requires_positive_fps() -> None:
    with pytest.raises(Exception):
        VideoMetadata(
            path="bad.mp4",
            fps=0.0,
            total_frames=10,
            duration_seconds=1.0,
            width=640,
            height=360,
            sample_fps=2.0,
        )

