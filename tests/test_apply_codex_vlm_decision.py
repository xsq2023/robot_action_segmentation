from pathlib import Path

import pytest
from PIL import Image

from robot_tas.cache import write_json
from robot_tas.cli.apply_codex_vlm_decision import convert_codex_decision


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (120, 120, 120)).save(path)


def _write_pipeline_dir(path: Path) -> None:
    frames = []
    for sample_index in range(3):
        filename = f"sample_{sample_index:06d}_frame_{sample_index * 10:06d}.jpg"
        _write_frame(path / "sampled_frames" / filename)
        frames.append(
            {
                "sample_index": sample_index,
                "original_frame_id": sample_index * 10,
                "timestamp_seconds": float(sample_index),
                "image_path": f"sampled_frames/{filename}",
                "image_sha256": f"sha-{sample_index}",
            }
        )
    write_json(
        path / "metadata.json",
        {
            "video": {
                "path": "/tmp/observations/123/videos/head_color.mp4",
                "fps": 10.0,
                "total_frames": 30,
                "duration_seconds": 3.0,
                "width": 32,
                "height": 24,
                "sample_fps": 1.0,
            },
            "sampled_frames": frames,
        },
    )


def test_convert_codex_decision_validates_action_set(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    _write_pipeline_dir(pipeline_dir)
    decision_path = tmp_path / "decision.json"
    write_json(
        decision_path,
        {
            "prompt_version": "codex_vlm_decision_v2",
            "segments": [
                {
                    "start_sample_index": 0,
                    "end_sample_index": 2,
                    "start_frame_id": 0,
                    "end_frame_id": 20,
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "action_label": "pick_red_block",
                }
            ],
            "accepted_or_corrected_boundaries": [],
        },
    )

    with pytest.raises(ValueError, match="outside the action set"):
        convert_codex_decision(
            pipeline_output_dir=pipeline_dir,
            decision_path=decision_path,
            output_dir=tmp_path / "out",
            action_set=["pick", "place"],
        )


def test_convert_codex_decision_writes_standard_output(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    _write_pipeline_dir(pipeline_dir)
    decision_path = tmp_path / "decision.json"
    write_json(
        decision_path,
        {
            "segments": [
                {
                    "start_sample_index": 0,
                    "end_sample_index": 1,
                    "start_frame_id": 0,
                    "end_frame_id": 10,
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "action_label": "pick",
                    "description": "The gripper picks the object.",
                    "primary_object": "object",
                    "actor_motion": "grasp",
                    "contact_state": "grasped",
                    "object_motion": "moving_with_gripper",
                    "selected_views": ["head_color", "hand_right_color"],
                    "trajectory_evidence": ["gripper state is closing"],
                    "confidence": 0.8,
                },
                {
                    "start_sample_index": 1,
                    "end_sample_index": 2,
                    "start_frame_id": 10,
                    "end_frame_id": 20,
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "action_label": "place",
                    "description": "The gripper places the object.",
                    "primary_object": "object",
                    "actor_motion": "place",
                    "contact_state": "released",
                    "object_motion": "placed",
                    "confidence": 0.8,
                },
            ],
            "accepted_or_corrected_boundaries": [
                {
                    "boundary_sample_index": 1,
                    "boundary_frame_id": 10,
                    "boundary_time": 1.0,
                    "before_action": "pick",
                    "after_action": "place",
                    "transition_type": "pick_to_place",
                    "source": "new_visual_boundary",
                    "selected_views": ["head_color"],
                    "view_evidence": [{"view": "head_color", "evidence": "object moves toward target"}],
                    "visual_evidence": ["object is now moving toward the target"],
                    "trajectory_evidence": ["end-effector motion peak supports timing"],
                    "confidence": 0.8,
                }
            ],
        },
    )

    output = convert_codex_decision(
        pipeline_output_dir=pipeline_dir,
        decision_path=decision_path,
        output_dir=tmp_path / "out",
        action_set=["pick", "place"],
    )

    assert [segment.action_label for segment in output.segments] == ["pick", "place"]
    assert output.segments[0].selected_views == ["head_color", "hand_right_color"]
    assert output.segments[0].trajectory_evidence == ["gripper state is closing"]
    assert output.boundaries[0].transition_type == "pick_to_place"
    assert output.boundaries[0].selected_views == ["head_color"]
    assert output.boundaries[0].trajectory_evidence == ["end-effector motion peak supports timing"]
    assert output.prompt_versions["codex_vlm_decision"] == "codex_vlm_decision_v2"
