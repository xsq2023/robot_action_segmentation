import json
from pathlib import Path

import pytest

from robot_tas.cli.evaluate_multiview_codex_task import (
    TRI_VIEW_EVIDENCE_VIEWS,
    _requested_views,
    build_bootstrap_decision,
    existing_episode_views,
    validate_decision,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_metadata(path: Path) -> None:
    _write_json(
        path / "metadata.json",
        {
            "video": {
                "path": "/tmp/episode/videos/head_color.mp4",
                "fps": 10.0,
                "total_frames": 21,
                "duration_seconds": 2.1,
                "width": 32,
                "height": 24,
                "sample_fps": 1.0,
            },
            "sampled_frames": [
                {
                    "sample_index": index,
                    "original_frame_id": index * 10,
                    "timestamp_seconds": float(index),
                    "image_path": f"sampled_frames/{index}.jpg",
                    "image_sha256": f"sha-{index}",
                }
                for index in range(3)
            ],
        },
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _valid_inspection_audit() -> dict:
    return {
        "reviewed_visual_sources": [
            {
                "path": "multiview_pack/contact_sheets/multiview_global_000_011.jpg",
                "views": TRI_VIEW_EVIDENCE_VIEWS,
                "purpose": "chronological tri-view review",
            }
        ],
        "semantic_change_observations": [
            {
                "sample_index": 1,
                "frame_id": 10,
                "time": 1.0,
                "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                "visual_state_change": "the object state visibly changes at the boundary",
                "decision": "accepted_boundary",
            }
        ],
        "trajectory_change_observations": [],
        "undersegmentation_checks": [
            {
                "segment_index": 0,
                "frame_range": [0, 20],
                "checked_frames": [0, 10, 20],
                "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                "result": "single test segment checked for hidden semantic changes",
            }
        ],
    }


def test_requested_views_keeps_base_camera_first() -> None:
    assert _requested_views("hand_right_color,head_color", "head_color.mp4") == [
        "head_color",
        "hand_right_color",
    ]


def test_existing_episode_views_reports_missing_optional_views(tmp_path: Path) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "head_color.mp4").touch()
    (tmp_path / "videos" / "hand_right_color.mp4").touch()

    existing, missing = existing_episode_views(
        episode_dir=tmp_path,
        requested_views=["head_color", "hand_left_color", "hand_right_color"],
    )

    assert existing == ["head_color", "hand_right_color"]
    assert missing == ["hand_left_color"]


def test_validate_decision_requires_sampled_boundary_and_multiview_evidence(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    _write_metadata(pipeline_dir)
    decision_path = tmp_path / "decision.json"
    _write_json(
        decision_path,
        {
            "segments": [
                {
                    "start_sample_index": 0,
                    "end_sample_index": 2,
                    "start_frame_id": 0,
                    "end_frame_id": 20,
                    "action_label": "move",
                    "selected_views": ["head_color", "hand_right_color"],
                }
            ],
            "accepted_or_corrected_boundaries": [
                {
                    "boundary_sample_index": 1,
                    "boundary_frame_id": 10,
                    "boundary_time": 1.0,
                    "before_action": "pick",
                    "after_action": "place",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="selected_views or view_evidence"):
        validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload["accepted_or_corrected_boundaries"][0]["selected_views"] = ["head_color", "hand_right_color"]
    _write_json(decision_path, payload)
    validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)

    payload["accepted_or_corrected_boundaries"][0]["boundary_frame_id"] = 11
    _write_json(decision_path, payload)
    with pytest.raises(ValueError, match="not a sampled base frame"):
        validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)


def test_validate_vlm_decision_requires_inspection_audit(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    _write_metadata(pipeline_dir)
    _touch(tmp_path / "multiview_pack" / "contact_sheets" / "multiview_global_000_011.jpg")
    decision_path = tmp_path / "decision" / "codex_vlm_decision.json"
    payload = {
        "prompt_version": "codex_vlm_decision_v2",
        "segments": [
                {
                    "start_sample_index": 0,
                    "end_sample_index": 2,
                    "start_frame_id": 0,
                    "end_frame_id": 20,
                    "action_label": "move",
                    "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                    "left_hand_state": "free",
                    "right_hand_state": "carrying",
                    "held_objects_by_hand": {"left": [], "right": ["object"]},
                }
        ],
        "accepted_or_corrected_boundaries": [
            {
                "boundary_sample_index": 1,
                "boundary_frame_id": 10,
                    "boundary_time": 1.0,
                    "before_action": "pick",
                    "after_action": "place",
                    "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                }
            ],
        }
    _write_json(decision_path, payload)

    with pytest.raises(ValueError, match="missing inspection_audit"):
        validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)

    payload["inspection_audit"] = _valid_inspection_audit()
    _write_json(decision_path, payload)
    validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)


def test_validate_vlm_decision_requires_high_risk_internal_event_review(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    _write_metadata(pipeline_dir)
    _touch(tmp_path / "multiview_pack" / "contact_sheets" / "multiview_global_000_011.jpg")
    _touch(tmp_path / "multiview_pack" / "contact_sheets" / "local_open.jpg")
    _write_json(
        tmp_path / "multiview_pack" / "trajectory_reference.json",
        {
            "selected_candidates": [
                {
                    "candidate_sample_index": 1,
                    "candidate_frame_id": 10,
                    "event_type": "gripper_command_open",
                    "trajectory_score": 3.0,
                    "signals": {"gripper_command_before": 1.0, "gripper_command_after": 0.0},
                }
            ]
        },
    )
    _write_json(
        tmp_path / "multiview_pack" / "candidate_reference.json",
        {"combined_candidates": []},
    )
    _write_json(
        tmp_path / "multiview_pack" / "manifest.json",
        {
            "local_candidate_sheets": [
                {
                    "path": "contact_sheets/local_open.jpg",
                    "candidate_frame_id": 10,
                }
            ]
        },
    )
    decision_path = tmp_path / "decision" / "codex_vlm_decision.json"
    payload = {
        "prompt_version": "codex_vlm_decision_v2",
        "segments": [
            {
                "segment_id": 0,
                "start_sample_index": 0,
                "end_sample_index": 2,
                "start_frame_id": 0,
                "end_frame_id": 20,
                "action_label": "carry",
                "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                "left_hand_state": "free",
                "right_hand_state": "carrying",
                "held_objects_by_hand": {"left": [], "right": ["object"]},
            }
        ],
        "accepted_or_corrected_boundaries": [],
        "inspection_audit": {
            "reviewed_visual_sources": [
                    {
                        "path": "multiview_pack/contact_sheets/multiview_global_000_011.jpg",
                        "views": TRI_VIEW_EVIDENCE_VIEWS,
                        "purpose": "chronological tri-view review",
                    }
                ],
            "semantic_change_observations": [
                {
                        "sample_index": 1,
                        "frame_id": 10,
                        "time": 1.0,
                        "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                        "visual_state_change": "candidate checked inside carry segment",
                        "decision": "rejected_boundary",
                    }
            ],
            "trajectory_change_observations": [
                {
                    "candidate_frame_id": 10,
                    "event_type": "gripper_command_open",
                    "metrics": {"gripper_command_before": 1.0, "gripper_command_after": 0.0},
                    "interpretation": "internal event must be visually checked before keeping the segment whole",
                }
            ],
            "undersegmentation_checks": [
                    {
                        "segment_index": 0,
                        "frame_range": [0, 20],
                        "checked_frames": [0, 10, 20],
                        "selected_views": TRI_VIEW_EVIDENCE_VIEWS,
                        "result": "missing explicit high-risk internal event review",
                    }
                ],
        },
    }
    _write_json(decision_path, payload)

    with pytest.raises(ValueError, match="did not explicitly review internal event frames"):
        validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)

    payload["inspection_audit"]["undersegmentation_checks"][0].update(
        {
            "internal_event_frames_reviewed": [10],
            "reviewed_local_sheets": ["multiview_pack/contact_sheets/local_open.jpg"],
            "split_decision": "keep_single_segment",
        }
    )
    _write_json(decision_path, payload)
    validate_decision(decision_path=decision_path, pipeline_output_dir=pipeline_dir)


def test_build_bootstrap_decision_adds_view_audit_fields(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    pack_dir = tmp_path / "pack"
    _write_metadata(pipeline_dir)
    _write_json(
        pipeline_dir / "final_segments.json",
        {
            "segments": [
                {
                    "start_sample_index": 0,
                    "end_sample_index": 1,
                    "start_frame_id": 0,
                    "end_frame_id": 10,
                    "action_label": "pick",
                },
                {
                    "start_sample_index": 1,
                    "end_sample_index": 2,
                    "start_frame_id": 10,
                    "end_frame_id": 20,
                    "action_label": "place",
                },
            ],
            "boundaries": [
                {
                    "boundary_sample_index": 1,
                    "boundary_frame_id": 10,
                    "boundary_time": 1.0,
                    "before_action": "pick",
                    "after_action": "place",
                    "transition_type": "pick_to_place",
                    "visual_evidence": ["rough visual boundary"],
                    "confidence": 0.7,
                }
            ],
        },
    )
    _write_json(
        pack_dir / "fused_cv_reference.json",
        {
            "selected_candidates": [
                {
                    "sample_index": 1,
                    "original_frame_id": 10,
                    "time_seconds": 1.0,
                    "top_views": [{"view": "hand_right_color", "score": 0.9}],
                }
            ]
        },
    )
    _write_json(pack_dir / "manifest.json", {"views": TRI_VIEW_EVIDENCE_VIEWS})

    decision_path = tmp_path / "decision" / "codex_vlm_decision.json"
    mode = build_bootstrap_decision(
        base_pipeline_dir=pipeline_dir,
        multiview_pack_dir=pack_dir,
        decision_path=decision_path,
        existing_views=TRI_VIEW_EVIDENCE_VIEWS,
    )

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert mode == "bootstrap_from_head_pipeline_with_multiview_hints"
    assert payload["accepted_or_corrected_boundaries"][0]["selected_views"] == TRI_VIEW_EVIDENCE_VIEWS
    assert payload["accepted_or_corrected_boundaries"][0]["view_evidence"]
