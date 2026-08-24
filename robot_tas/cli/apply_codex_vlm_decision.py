from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from robot_tas.action_set import parse_action_set
from robot_tas.cache import ensure_dir, read_json, write_json
from robot_tas.normalization import normalize_label, normalize_transition
from robot_tas.schemas import FinalOutput, LabeledSegment, MergedBoundary, SampledFrame
from robot_tas.visualization import write_annotated_action_video, write_timeline_html
from robot_tas.vlm_pack import load_sampled_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Codex-as-VLM decision JSON into the standard Robot TAS final_segments.json format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline-output-dir",
        required=True,
        help="Existing run_tas.py output directory containing metadata.json and sampled_frames/.",
    )
    parser.add_argument("--decision", required=True, help="Codex-as-VLM decision JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory for final_segments.json and timeline.html.")
    parser.add_argument(
        "--action-set",
        default=None,
        help="Comma-separated allowed action labels. Defaults to the broad robot manipulation action set.",
    )
    return parser.parse_args()


def _require_allowed(label: str, allowed: set[str], field_name: str) -> str:
    normalized = normalize_label(label)
    if normalized not in allowed:
        raise ValueError(f"{field_name} label is outside the action set: {label!r} -> {normalized!r}")
    return normalized


def _source_ids(boundary: dict[str, Any]) -> list[str]:
    source = str(boundary.get("source", "codex_vlm"))
    candidate = boundary.get("source_candidate_frame_id")
    frame_id = boundary.get("boundary_frame_id", "unknown")
    if candidate is None:
        return [f"codex_vlm_{source}_frame_{frame_id}"]
    return [f"codex_vlm_{source}_from_visual_candidate_{candidate}"]


def _timeline_frames(
    sampled_frames: list[SampledFrame],
    pipeline_output_dir: Path,
    output_dir: Path,
) -> list[SampledFrame]:
    return [
        frame.model_copy(
            update={
                "image_path": os.path.relpath(
                    pipeline_output_dir / frame.image_path,
                    start=output_dir,
                )
            }
        )
        for frame in sampled_frames
    ]


def _dict_evidence(items: list[Any]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence.append({str(key): str(value) for key, value in item.items()})
    return evidence


def convert_codex_decision(
    pipeline_output_dir: Path,
    decision_path: Path,
    output_dir: Path,
    action_set: list[str],
) -> FinalOutput:
    metadata, sampled_frames = load_sampled_artifacts(pipeline_output_dir)
    decision = read_json(decision_path)
    allowed = set(action_set)

    segments: list[LabeledSegment] = []
    for segment_id, payload in enumerate(decision["segments"]):
        action_label = _require_allowed(str(payload["action_label"]), allowed, "segment.action_label")
        description = str(payload.get("description", action_label.replace("_", " ")))
        segments.append(
            LabeledSegment(
                segment_id=segment_id,
                start_sample_index=int(payload["start_sample_index"]),
                end_sample_index=int(payload["end_sample_index"]),
                start_frame_id=int(payload["start_frame_id"]),
                end_frame_id=int(payload["end_frame_id"]),
                start_time=float(payload.get("start_time", 0.0)),
                end_time=float(payload.get("end_time", 0.0)),
                action_label=action_label,
                description=description,
                primary_object=str(payload.get("primary_object", "unknown")),
                secondary_objects=[str(item) for item in payload.get("secondary_objects", [])],
                actor_motion=str(payload.get("actor_motion", "other")),
                contact_state=str(payload.get("contact_state", "unclear")),
                object_motion=str(payload.get("object_motion", "unclear")),
                goal=str(payload.get("goal", description)),
                selected_views=[str(item) for item in payload.get("selected_views", []) or []],
                left_hand_state=str(payload.get("left_hand_state", "unclear")),
                right_hand_state=str(payload.get("right_hand_state", "unclear")),
                held_objects_by_hand={
                    "left": [str(item) for item in payload.get("held_objects_by_hand", {}).get("left", [])],
                    "right": [str(item) for item in payload.get("held_objects_by_hand", {}).get("right", [])],
                },
                visual_evidence=[str(item) for item in payload.get("visual_evidence", []) or []],
                trajectory_evidence=[str(item) for item in payload.get("trajectory_evidence", []) or []],
                confidence=float(payload.get("confidence", 0.0)),
            )
        )

    boundaries: list[MergedBoundary] = []
    for payload in decision.get("accepted_or_corrected_boundaries", []):
        before_action = _require_allowed(str(payload["before_action"]), allowed, "boundary.before_action")
        after_action = _require_allowed(str(payload["after_action"]), allowed, "boundary.after_action")
        transition_type = normalize_transition(
            before_action=before_action,
            after_action=after_action,
            explicit=str(payload.get("transition_type", "")),
        )
        boundaries.append(
            MergedBoundary(
                boundary_sample_index=int(payload["boundary_sample_index"]),
                boundary_frame_id=int(payload["boundary_frame_id"]),
                boundary_time=float(payload["boundary_time"]),
                before_action=before_action,
                after_action=after_action,
                transition_type=transition_type,
                visual_evidence=[str(item) for item in payload.get("visual_evidence", [])],
                selected_views=[str(item) for item in payload.get("selected_views", []) or []],
                view_evidence=_dict_evidence(payload.get("view_evidence", []) or []),
                trajectory_evidence=[str(item) for item in payload.get("trajectory_evidence", []) or []],
                supporting_windows=[],
                confidence=float(payload.get("confidence", 0.0)),
                source_proposal_ids=_source_ids(payload),
            )
        )

    return FinalOutput(
        video=metadata,
        segments=segments,
        boundaries=boundaries,
        prompt_versions={
            "codex_vlm_decision": str(decision.get("prompt_version", "codex_vlm_decision_v2")),
            "action_set": ",".join(action_set),
        },
    )


def main() -> None:
    args = parse_args()
    pipeline_output_dir = Path(args.pipeline_output_dir).resolve()
    decision_path = Path(args.decision).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    action_set = parse_action_set(args.action_set)

    final_output = convert_codex_decision(
        pipeline_output_dir=pipeline_output_dir,
        decision_path=decision_path,
        output_dir=output_dir,
        action_set=action_set,
    )
    decision = read_json(decision_path)
    write_json(output_dir / "codex_vlm_decision.json", decision)
    write_json(output_dir / "final_segments.json", final_output.model_dump(mode="json"))
    _, sampled_frames = load_sampled_artifacts(pipeline_output_dir)
    write_timeline_html(
        metadata=final_output.video,
        sampled_frames=_timeline_frames(sampled_frames, pipeline_output_dir, output_dir),
        boundaries=final_output.boundaries,
        segments=final_output.segments,
        output_path=output_dir / "timeline.html",
    )
    write_annotated_action_video(
        metadata=final_output.video,
        segments=final_output.segments,
        output_path=output_dir / "annotated_actions.mp4",
    )
    print(f"[done] final_segments={output_dir / 'final_segments.json'}")


if __name__ == "__main__":
    main()
