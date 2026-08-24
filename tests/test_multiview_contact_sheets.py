from pathlib import Path

import pytest
from PIL import Image

from robot_tas.cli.prepare_multiview_codex_pack import (
    _sheet_view_columns,
    _write_multiview_contact_sheet,
)
from robot_tas.schemas import SampledFrame


def _write_frame(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)


def _sample(sample_index: int) -> SampledFrame:
    frame_id = sample_index * 10
    return SampledFrame(
        sample_index=sample_index,
        original_frame_id=frame_id,
        timestamp_seconds=frame_id / 30.0,
        image_path=f"sampled_frames/sample_{sample_index:06d}_frame_{frame_id:06d}.jpg",
        image_sha256=f"sha-{sample_index}",
    )


def test_multiview_contact_sheet_is_enlarged_tri_view_frame_rows(tmp_path: Path) -> None:
    view_names = ["head_color", "hand_left_color", "hand_right_color"]
    frames_by_view = {view: [_sample(0), _sample(1)] for view in view_names}
    view_dirs: dict[str, Path] = {}
    for view_index, view in enumerate(view_names):
        view_dir = tmp_path / "views" / view
        view_dirs[view] = view_dir
        for frame in frames_by_view[view]:
            _write_frame(view_dir / frame.image_path, (40 + view_index * 60, 80, 120))

    output_path = tmp_path / "contact_sheets" / "multiview_global_000_001.jpg"
    _write_multiview_contact_sheet(
        sample_positions=[0, 1],
        view_names=_sheet_view_columns(frames_by_view),
        frames_by_view=frames_by_view,
        view_dirs=view_dirs,
        output_path=output_path,
        view_thumb_size=(360, 270),
        columns=3,
    )

    with Image.open(output_path) as image:
        assert image.size == (8 * 2 + 360 * 3, 42 + 2 * (34 + 270) + 8)


def test_multiview_contact_sheet_requires_fixed_three_view_columns() -> None:
    with pytest.raises(ValueError, match="fixed tri-view columns"):
        _sheet_view_columns({"head_color": [], "hand_right_color": []})
