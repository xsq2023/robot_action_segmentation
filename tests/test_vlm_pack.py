from pathlib import Path

from PIL import Image

from robot_tas.cache import write_json
from robot_tas.vlm_pack import ContactSheetConfig, build_codex_vlm_pack


def _write_frame(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_build_codex_vlm_pack_writes_prompt_inputs(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline" / "123"
    sampled_dir = pipeline_dir / "sampled_frames"
    frames = []
    for index, color in enumerate([(20, 20, 20), (80, 80, 80), (160, 160, 160), (220, 220, 220)]):
        filename = f"sample_{index:06d}_frame_{index * 10:06d}.jpg"
        _write_frame(sampled_dir / filename, color)
        frames.append(
            {
                "sample_index": index,
                "original_frame_id": index * 10,
                "timestamp_seconds": index / 3.0,
                "image_path": f"sampled_frames/{filename}",
                "image_sha256": f"sha-{index}",
                "motion_score": index / 10.0,
                "mean_luma": index / 10.0,
            }
        )

    write_json(
        pipeline_dir / "metadata.json",
        {
            "video": {
                "path": "/tmp/observations/123/videos/head_color.mp4",
                "fps": 30.0,
                "total_frames": 40,
                "duration_seconds": 4 / 3,
                "width": 32,
                "height": 24,
                "sample_fps": 3.0,
            },
            "sampled_frames": frames,
        },
    )
    write_json(
        pipeline_dir / "final_segments.json",
        {
            "video": {},
            "segments": [],
            "boundaries": [
                {
                    "boundary_sample_index": 2,
                    "boundary_frame_id": 20,
                    "boundary_time": 2 / 3,
                    "before_action": "pick_object",
                    "after_action": "place_object",
                    "transition_type": "pick_to_place",
                    "visual_evidence": ["object leaves row"],
                    "supporting_windows": [0],
                    "confidence": 0.8,
                    "source_proposal_ids": ["a"],
                }
            ],
            "prompt_versions": {},
        },
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return JSON only.", encoding="utf-8")

    manifest = build_codex_vlm_pack(
        pipeline_output_dir=pipeline_dir,
        output_dir=tmp_path / "pack",
        prompt_path=prompt,
        config=ContactSheetConfig(global_sheet_size=2, local_radius=1, top_visual_changes=2),
        force=True,
    )

    assert manifest["global_sheet_count"] == 2
    assert manifest["local_candidate_sheet_count"] == 1
    assert manifest["action_set"]
    assert (tmp_path / "pack" / "vlm_input.json").exists()
    assert (tmp_path / "pack" / "vlm_prompt.md").exists()
    assert "Allowed action labels:" in (tmp_path / "pack" / "prompt_used.txt").read_text(encoding="utf-8")
    assert "Allowed action labels:" in (tmp_path / "pack" / "vlm_prompt.md").read_text(encoding="utf-8")
    assert (tmp_path / "pack" / "contact_sheets" / "global_000_001.jpg").exists()
    assert (tmp_path / "pack" / "contact_sheets" / "local_candidate_frame_0020_samples_001_003.jpg").exists()
