from pathlib import Path

from robot_tas.cache import cache_metadata_path
from robot_tas.cli import prepare_multiview_codex_pack as multiview
from robot_tas.cli.run_tas import prepare_windows
from robot_tas.schemas import SampledFrame, VideoMetadata


def _frame(sample_index: int) -> SampledFrame:
    frame_id = sample_index * 10
    return SampledFrame(
        sample_index=sample_index,
        original_frame_id=frame_id,
        timestamp_seconds=float(sample_index),
        image_path=f"sampled_frames/sample_{sample_index:06d}_frame_{frame_id:06d}.jpg",
        image_sha256=f"sha-{sample_index}",
    )


def _metadata(path: Path, sample_fps: float = 1.0) -> VideoMetadata:
    return VideoMetadata(
        path=str(path),
        fps=10.0,
        total_frames=30,
        duration_seconds=3.0,
        width=32,
        height=24,
        sample_fps=sample_fps,
    )


def test_prepare_windows_invalidates_cache_when_window_params_change(tmp_path: Path) -> None:
    frames = [_frame(index) for index in range(8)]
    first = prepare_windows(
        sampled_frames=frames,
        output_dir=tmp_path,
        window_size=4,
        window_stride=4,
        force=False,
    )
    second = prepare_windows(
        sampled_frames=frames,
        output_dir=tmp_path,
        window_size=3,
        window_stride=3,
        force=False,
    )

    assert [window.start_sample_index for window in first] == [0, 4]
    assert [window.start_sample_index for window in second] == [0, 3, 5]
    assert cache_metadata_path(tmp_path / "windows.json").is_file()


def test_multiview_sample_view_invalidates_cache_when_video_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "head_color.mp4"
    video_path.write_bytes(b"first")
    view_dir = tmp_path / "view"
    decode_calls = 0

    def fake_read_video_metadata(video_path: Path, sample_fps: float) -> VideoMetadata:
        return _metadata(video_path, sample_fps=sample_fps)

    def fake_sample_video_frames_ffmpeg(
        video_path: Path,
        metadata: VideoMetadata,
        sampled_frames_dir: Path,
        target_frame_ids: list[int],
    ) -> list[SampledFrame]:
        nonlocal decode_calls
        decode_calls += 1
        sampled_frames_dir.mkdir(parents=True, exist_ok=True)
        return [_frame(index) for index, _frame_id in enumerate(target_frame_ids)]

    monkeypatch.setattr(multiview, "read_video_metadata", fake_read_video_metadata)
    monkeypatch.setattr(multiview, "_sample_video_frames_ffmpeg", fake_sample_video_frames_ffmpeg)

    _metadata_one, _frames_one, reused_one = multiview._sample_view(
        video_path=video_path,
        view_dir=view_dir,
        sample_fps=1.0,
        frame_ids=[0, 10],
        force=False,
        decoder="ffmpeg",
    )
    _metadata_two, _frames_two, reused_two = multiview._sample_view(
        video_path=video_path,
        view_dir=view_dir,
        sample_fps=1.0,
        frame_ids=[0, 10],
        force=False,
        decoder="ffmpeg",
    )

    video_path.write_bytes(b"second")
    _metadata_three, _frames_three, reused_three = multiview._sample_view(
        video_path=video_path,
        view_dir=view_dir,
        sample_fps=1.0,
        frame_ids=[0, 10],
        force=False,
        decoder="ffmpeg",
    )

    assert reused_one is False
    assert reused_two is True
    assert reused_three is False
    assert decode_calls == 2
    assert cache_metadata_path(view_dir / "metadata.json").is_file()
