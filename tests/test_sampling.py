from robot_tas.sampler import compute_sample_frame_ids


def test_compute_sample_frame_ids_preserves_original_spacing() -> None:
    frame_ids = compute_sample_frame_ids(total_frames=301, native_fps=30.0, sample_fps=2.0)
    assert frame_ids[0] == 0
    assert frame_ids[-1] == 300
    assert frame_ids[:5] == [0, 15, 30, 45, 60]
    assert len(frame_ids) == 21

