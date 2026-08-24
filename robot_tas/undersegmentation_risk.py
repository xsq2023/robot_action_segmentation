from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RISK_ACTION_LABELS = {
    "approach",
    "carry",
    "lower",
    "move",
    "place",
    "reach",
    "reach_for",
    "reposition",
    "retract",
    "transport",
}

LONG_SEGMENT_FRAME_THRESHOLD = 240
VISUAL_CANDIDATE_THRESHOLD = 3


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame(candidate: dict[str, Any]) -> int | None:
    for key in ("candidate_frame_id", "original_frame_id"):
        value = candidate.get(key)
        if value is not None:
            return int(value)
    return None


def _trajectory_candidates(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "multiview_pack" / "trajectory_reference.json"
    if not path.is_file():
        return []
    payload = read_json(path)
    return [item for item in payload.get("selected_candidates", []) if _frame(item) is not None]


def _combined_candidates(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "multiview_pack" / "candidate_reference.json"
    if not path.is_file():
        return []
    payload = read_json(path)
    return [item for item in payload.get("combined_candidates", []) if _frame(item) is not None]


def _local_sheets_by_frame(episode_dir: Path) -> dict[int, list[dict[str, Any]]]:
    path = episode_dir / "multiview_pack" / "manifest.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    sheets: dict[int, list[dict[str, Any]]] = {}
    for item in payload.get("local_candidate_sheets", []):
        frame = item.get("candidate_frame_id")
        if frame is None:
            continue
        sheets.setdefault(int(frame), []).append(item)
    return sheets


def _event_type(candidate: dict[str, Any]) -> str:
    value = candidate.get("event_type")
    if value is not None:
        return str(value)
    trajectory = candidate.get("trajectory_reference")
    if isinstance(trajectory, dict) and trajectory.get("event_type") is not None:
        return str(trajectory["event_type"])
    return ""


def _trajectory_score(candidate: dict[str, Any]) -> float | None:
    value = candidate.get("trajectory_score")
    if value is None and isinstance(candidate.get("trajectory_reference"), dict):
        value = candidate["trajectory_reference"].get("trajectory_score")
    return float(value) if value is not None else None


def _signals(candidate: dict[str, Any]) -> dict[str, Any]:
    signals = candidate.get("signals")
    if isinstance(signals, dict):
        return signals
    trajectory = candidate.get("trajectory_reference")
    if isinstance(trajectory, dict) and isinstance(trajectory.get("signals"), dict):
        return trajectory["signals"]
    return {}


def _visual_score(candidate: dict[str, Any]) -> float | None:
    visual = candidate.get("visual_prior")
    if isinstance(visual, dict) and visual.get("fused_score") is not None:
        return float(visual["fused_score"])
    return None


def _candidate_sources(candidate: dict[str, Any]) -> list[str]:
    raw_sources = candidate.get("candidate_sources", [])
    if not isinstance(raw_sources, list):
        return []
    return [str(source) for source in raw_sources]


def _candidate_summary(candidate: dict[str, Any], sheets_by_frame: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    frame = _frame(candidate)
    if frame is None:
        raise ValueError("candidate has no frame")
    return {
        "frame_id": frame,
        "sample_index": int(candidate.get("candidate_sample_index", candidate.get("sample_index", -1))),
        "event_type": _event_type(candidate),
        "trajectory_score": _trajectory_score(candidate),
        "signals": _signals(candidate),
        "candidate_sources": _candidate_sources(candidate),
        "visual_fused_score": _visual_score(candidate),
        "local_sheets": [str(item["path"]) for item in sheets_by_frame.get(frame, []) if item.get("path")],
    }


def find_undersegmentation_risks(episode_dir: Path, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajectory_candidates = _trajectory_candidates(episode_dir)
    combined_candidates = _combined_candidates(episode_dir)
    sheets_by_frame = _local_sheets_by_frame(episode_dir)
    risks: list[dict[str, Any]] = []

    for segment in segments:
        start_frame = int(segment["start_frame_id"])
        end_frame = int(segment["end_frame_id"])
        duration_frames = end_frame - start_frame
        action = str(segment.get("action_label", "")).lower()
        internal_trajectory = [
            candidate
            for candidate in trajectory_candidates
            if start_frame < (_frame(candidate) or -1) < end_frame
        ]
        internal_gripper = [
            candidate for candidate in internal_trajectory if "gripper" in _event_type(candidate)
        ]
        internal_visual = [
            candidate
            for candidate in combined_candidates
            if start_frame < (_frame(candidate) or -1) < end_frame and "visual_prior" in _candidate_sources(candidate)
        ]

        is_risk_action = action in RISK_ACTION_LABELS
        reasons: list[str] = []
        if is_risk_action and duration_frames >= LONG_SEGMENT_FRAME_THRESHOLD:
            reasons.append(f"long_{action}_segment_{duration_frames}_frames")
        if is_risk_action and internal_gripper:
            reasons.append("internal_gripper_event_inside_non_grasp_release_segment")
        if is_risk_action and len(internal_trajectory) >= 2:
            reasons.append(f"multiple_internal_trajectory_candidates_{len(internal_trajectory)}")
        if is_risk_action and duration_frames >= 120 and len(internal_visual) >= VISUAL_CANDIDATE_THRESHOLD:
            reasons.append(f"multiple_internal_visual_candidates_{len(internal_visual)}")

        if not reasons:
            continue

        required_candidates = internal_gripper or internal_trajectory
        if duration_frames >= LONG_SEGMENT_FRAME_THRESHOLD:
            required_candidates = internal_trajectory

        risks.append(
            {
                "segment_index": int(segment.get("segment_id", len(risks))),
                "action_label": action,
                "frame_range": [start_frame, end_frame],
                "duration_frames": duration_frames,
                "risk_reasons": reasons,
                "required_internal_event_frames": sorted(
                    {
                        int(summary["frame_id"])
                        for summary in (
                            _candidate_summary(candidate, sheets_by_frame)
                            for candidate in required_candidates
                        )
                    }
                ),
                "internal_trajectory_events": [
                    _candidate_summary(candidate, sheets_by_frame) for candidate in internal_trajectory
                ],
                "internal_visual_candidates": [
                    _candidate_summary(candidate, sheets_by_frame) for candidate in internal_visual
                ],
            }
        )

    return risks
