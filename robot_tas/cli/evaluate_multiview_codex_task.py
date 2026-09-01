from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_tas.cli.evaluate_agibot_task import EpisodeEval, evaluate_episode, find_episode_dirs, load_task_info
from robot_tas.undersegmentation_risk import find_undersegmentation_risks


DEFAULT_VIEWS = [
    "head_color",
    "hand_left_color",
    "hand_right_color",
]

TRI_VIEW_EVIDENCE_VIEWS = ["head_color", "hand_left_color", "hand_right_color"]


@dataclass(slots=True)
class EpisodeRunStatus:
    episode_id: int
    status: str
    reason: str
    episode_dir: str
    existing_views: list[str]
    missing_views: list[str]
    base_head_pipeline: str | None = None
    multiview_pack: str | None = None
    decision: str | None = None
    final: str | None = None
    decision_mode: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run multi-view Codex TAS decisions for one AgiBotWorld task and evaluate locked predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations-dir", required=True, help="Observation task directory.")
    parser.add_argument("--task-info", required=True, help="External task_info JSON path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for this task, e.g. task_327/head_multiview.")
    parser.add_argument("--camera", default="head_color.mp4", help="Base camera video filename.")
    parser.add_argument("--model", default="codex-local", help="Base run_tas.py model.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Sampling FPS.")
    parser.add_argument("--window-size", type=int, default=16, help="Sliding window size in sampled frames.")
    parser.add_argument("--window-stride", type=int, default=8, help="Sliding window stride in sampled frames.")
    parser.add_argument("--boundary-tolerance", type=int, default=30, help="Evaluation boundary tolerance in frames.")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated multi-view camera names without .mp4.")
    parser.add_argument("--candidate-count", type=int, default=24, help="Fused CV candidate count.")
    parser.add_argument("--candidate-min-gap", type=int, default=5, help="Minimum sample gap between fused CV candidates.")
    parser.add_argument("--overhead-view", default="head_color", help="Primary overhead/time-axis view for 50%% visual-source weighting.")
    parser.add_argument(
        "--tri-view-views",
        default="head_color,hand_left_color,hand_right_color",
        help="Comma-separated overhead+hand views for the 25%% tri-view visual source.",
    )
    parser.add_argument(
        "--proprio-root",
        default=None,
        help="Optional proprio_stats_extracted root. When set, uses <root>/<task_id>/<episode_id>/timeseries.csv.",
    )
    parser.add_argument("--trajectory-candidate-count", type=int, default=12, help="Trajectory candidate count per episode.")
    parser.add_argument(
        "--trajectory-candidate-min-gap-frames",
        type=int,
        default=30,
        help="Minimum raw-frame gap between trajectory candidates.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute generated pipeline and multiview pack artifacts.")
    parser.add_argument(
        "--require-existing-decisions",
        action="store_true",
        help="Deprecated compatibility flag; reviewed decisions are required unless --allow-bootstrap-decisions is set.",
    )
    parser.add_argument(
        "--allow-bootstrap-decisions",
        action="store_true",
        help=(
            "Allow deterministic placeholder decisions when decision/codex_vlm_decision.json is missing. "
            "This is for baseline/debug runs only and is not a final semantic result."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_input_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / path).resolve()


def _view_name(raw: str) -> str:
    return raw.strip().removesuffix(".mp4")


def _base_view(camera: str) -> str:
    return _view_name(Path(camera).name)


def _requested_views(raw_views: str, camera: str) -> list[str]:
    base = _base_view(camera)
    views: list[str] = []
    seen: set[str] = set()
    for view in [base, *(_view_name(item) for item in raw_views.split(","))]:
        if not view or view in seen:
            continue
        views.append(view)
        seen.add(view)
    return views


def existing_episode_views(episode_dir: Path, requested_views: list[str]) -> tuple[list[str], list[str]]:
    existing: list[str] = []
    missing: list[str] = []
    for view in requested_views:
        if (episode_dir / "videos" / f"{view}.mp4").exists():
            existing.append(view)
        else:
            missing.append(view)
    return existing, missing


def _run_command(command: list[str], cwd: Path, label: str) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        details = "\n".join(part for part in [stdout, stderr] if part)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {details or 'no output'}")


def run_base_head_pipeline(
    api_dir: Path,
    video_path: Path,
    output_dir: Path,
    model: str,
    sample_fps: float,
    window_size: int,
    window_stride: int,
    force: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "robot_tas.cli.run_tas",
        "--video",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--sample-fps",
        str(sample_fps),
        "--window-size",
        str(window_size),
        "--window-stride",
        str(window_stride),
    ]
    if force:
        command.append("--force")
    _run_command(command=command, cwd=api_dir, label=f"base pipeline for {video_path}")


def run_multiview_pack(
    api_dir: Path,
    episode_dir: Path,
    output_dir: Path,
    sample_fps: float,
    views: list[str],
    candidate_count: int,
    candidate_min_gap: int,
    overhead_view: str,
    tri_view_views: str,
    trajectory_csv: Path | None,
    trajectory_candidate_count: int,
    trajectory_candidate_min_gap_frames: int,
    force: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "robot_tas.cli.prepare_multiview_codex_pack",
        "--episode-dir",
        str(episode_dir),
        "--output-dir",
        str(output_dir),
        "--sample-fps",
        str(sample_fps),
        "--views",
        ",".join(views),
        "--candidate-count",
        str(candidate_count),
        "--candidate-min-gap",
        str(candidate_min_gap),
        "--overhead-view",
        overhead_view,
        "--tri-view-views",
        tri_view_views,
        "--decoder",
        "ffmpeg",
    ]
    if trajectory_csv is not None:
        command.extend(
            [
                "--trajectory-csv",
                str(trajectory_csv),
                "--trajectory-candidate-count",
                str(trajectory_candidate_count),
                "--trajectory-candidate-min-gap-frames",
                str(trajectory_candidate_min_gap_frames),
            ]
        )
    if force:
        command.append("--force")
    _run_command(command=command, cwd=api_dir, label=f"multiview pack for {episode_dir}")


def _trajectory_csv_for_episode(proprio_root: Path | None, episode_dir: Path) -> Path | None:
    if proprio_root is None:
        return None
    task_id = episode_dir.parent.parent.name
    candidate = proprio_root / task_id / episode_dir.name / "timeseries.csv"
    return candidate if candidate.is_file() else None


def run_apply_decision(
    api_dir: Path,
    pipeline_output_dir: Path,
    decision_path: Path,
    output_dir: Path,
    *,
    allow_unreviewed_decision: bool = False,
) -> None:
    command = [
        sys.executable,
        "-m",
        "robot_tas.cli.apply_codex_vlm_decision",
        "--pipeline-output-dir",
        str(pipeline_output_dir),
        "--decision",
        str(decision_path),
        "--output-dir",
        str(output_dir),
    ]
    if allow_unreviewed_decision:
        command.append("--allow-unreviewed-decision")
    _run_command(command=command, cwd=api_dir, label=f"apply decision {decision_path}")


def _nearest_candidate(boundary_frame_id: int, fused_reference: dict[str, Any]) -> dict[str, Any] | None:
    candidates = fused_reference.get("selected_candidates", [])
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(int(item["original_frame_id"]) - boundary_frame_id))


def _selected_views(
    existing_views: list[str],
    action_label: str,
    candidate: dict[str, Any] | None = None,
    max_views: int = 3,
) -> list[str]:
    _ = (action_label, candidate)
    return [view for view in TRI_VIEW_EVIDENCE_VIEWS if view in existing_views][:max_views]


def _view_evidence(view: str, boundary_frame_id: int, candidate: dict[str, Any] | None) -> str:
    suffix = ""
    if candidate is not None:
        suffix = f"; fused CV candidate frame {int(candidate['original_frame_id'])} selected this neighborhood as a review hint"
    if view.startswith("hand_"):
        return f"selected to inspect close gripper-object contact around frame {boundary_frame_id}{suffix}"
    if view == "head_color":
        return f"selected as the frame/time reference and object-target workspace view around frame {boundary_frame_id}{suffix}"
    return f"selected as one of the fixed enlarged tri-view evidence columns around frame {boundary_frame_id}{suffix}"


def build_bootstrap_decision(
    base_pipeline_dir: Path,
    multiview_pack_dir: Path,
    decision_path: Path,
    existing_views: list[str],
) -> str:
    base_output = _read_json(base_pipeline_dir / "final_segments.json")
    fused_reference = _read_json(multiview_pack_dir / "fused_cv_reference.json")
    manifest = _read_json(multiview_pack_dir / "manifest.json")
    accepted_boundaries: list[dict[str, Any]] = []
    matched_candidate_frames: set[int] = set()

    for boundary in base_output.get("boundaries", []):
        frame_id = int(boundary["boundary_frame_id"])
        candidate = _nearest_candidate(frame_id, fused_reference)
        source_candidate_frame_id = int(candidate["original_frame_id"]) if candidate else frame_id
        if candidate:
            matched_candidate_frames.add(source_candidate_frame_id)
        after_action = str(boundary.get("after_action", "move"))
        selected = _selected_views(existing_views=existing_views, action_label=after_action, candidate=candidate)
        accepted_boundaries.append(
            {
                "boundary_sample_index": int(boundary["boundary_sample_index"]),
                "boundary_frame_id": frame_id,
                "boundary_time": float(boundary["boundary_time"]),
                "before_action": str(boundary.get("before_action", after_action)),
                "after_action": after_action,
                "transition_type": str(boundary.get("transition_type", "")),
                "source": "accepted_candidate",
                "source_candidate_frame_id": source_candidate_frame_id,
                "selected_views": selected,
                "view_evidence": [
                    {"view": view, "evidence": _view_evidence(view, frame_id, candidate)}
                    for view in selected
                ],
                "visual_evidence": [
                    str(item) for item in boundary.get("visual_evidence", [])
                ]
                or [
                    f"rough head-camera boundary at sampled frame {frame_id}",
                    "multi-view sheets and fused candidates are attached for audit",
                ],
                "confidence": float(boundary.get("confidence", 0.0)),
            }
        )

    segments: list[dict[str, Any]] = []
    for segment in base_output.get("segments", []):
        action_label = str(segment.get("action_label", "move"))
        selected = _selected_views(existing_views=existing_views, action_label=action_label)
        enriched = dict(segment)
        enriched.setdefault("start_time", segment.get("start_time", 0.0))
        enriched.setdefault("end_time", segment.get("end_time", 0.0))
        enriched.setdefault("description", action_label.replace("_", " "))
        enriched.setdefault("primary_object", "unknown")
        enriched.setdefault("secondary_objects", [])
        enriched.setdefault("actor_motion", "other")
        enriched.setdefault("contact_state", "unclear")
        enriched.setdefault("object_motion", "unclear")
        enriched.setdefault("goal", enriched["description"])
        enriched.setdefault("confidence", 0.0)
        enriched["selected_views"] = selected
        segments.append(enriched)

    rejected_candidates = []
    for candidate in fused_reference.get("selected_candidates", []):
        frame_id = int(candidate["original_frame_id"])
        if frame_id in matched_candidate_frames:
            continue
        rejected_candidates.append(
            {
                "candidate_frame_id": frame_id,
                "reason": "kept as a fused multi-view CV hint but not selected by the locked rough semantic boundary sequence",
            }
        )

    payload = {
        "prompt_version": "deterministic_bootstrap_no_vlm",
        "video_summary": "Bootstrap Codex decision using the head timeline with synchronized multi-view candidate sheets for view selection.",
        "decision_mode": "bootstrap_from_head_pipeline_with_multiview_hints",
        "multiview_manifest": str(multiview_pack_dir / "manifest.json"),
        "views_used": manifest.get("views", existing_views),
        "accepted_or_corrected_boundaries": accepted_boundaries,
        "rejected_visual_candidates": rejected_candidates,
        "segments": segments,
        "uncertainties": [
            "This decision was generated deterministically from the base head pipeline and fused multi-view hints; replace it with a Codex/VLM visual review decision JSON for final semantic claims."
        ],
    }
    _write_json(decision_path, payload)
    return str(payload["decision_mode"])


def ensure_decision(
    base_pipeline_dir: Path,
    multiview_pack_dir: Path,
    decision_path: Path,
    existing_views: list[str],
    require_existing: bool,
) -> str:
    if decision_path.exists():
        return "existing_codex_vlm_decision"
    if require_existing:
        raise FileNotFoundError(f"Missing required Codex/VLM decision JSON: {decision_path}")
    return build_bootstrap_decision(
        base_pipeline_dir=base_pipeline_dir,
        multiview_pack_dir=multiview_pack_dir,
        decision_path=decision_path,
        existing_views=existing_views,
    )


def _metadata_samples(pipeline_output_dir: Path) -> list[dict[str, Any]]:
    metadata = _read_json(pipeline_output_dir / "metadata.json")
    return list(metadata["sampled_frames"])


def _string_field(record: dict[str, Any], field_name: str, context: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} missing non-empty {field_name}.")
    return value


def _list_field(record: dict[str, Any], field_name: str, context: str) -> list[Any]:
    value = record.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} missing non-empty {field_name}.")
    return value


def _views_from_record(record: dict[str, Any]) -> list[str]:
    raw_views = record.get("selected_views", record.get("views", []))
    if not isinstance(raw_views, list):
        return []
    return [str(view) for view in raw_views if str(view).strip()]


def _resolve_episode_artifact_path(episode_dir: Path, raw_path: str) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None
    candidates = [episode_dir / path]
    if len(episode_dir.parents) >= 3:
        candidates.append(episode_dir.parents[2] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_manifest_artifact_path(manifest_path: Path, raw_path: str) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve() if path.exists() else None
    candidate = manifest_path.parent / path
    return candidate.resolve() if candidate.exists() else None


def _artifact_label(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _has_fixed_tri_view(views: list[str]) -> bool:
    return set(TRI_VIEW_EVIDENCE_VIEWS).issubset(set(views))


def _validate_reviewed_sheet_coverage(
    *,
    decision: dict[str, Any],
    episode_dir: Path,
    reviewed_source_paths: set[Path],
) -> None:
    manifest_path = episode_dir / "multiview_pack" / "manifest.json"
    if not manifest_path.is_file():
        return

    manifest = _read_json(manifest_path)
    global_sheet_paths: list[Path] = []
    for index, sheet in enumerate(manifest.get("global_sheets", []) or []):
        if not isinstance(sheet, dict):
            raise ValueError(f"multiview manifest global_sheets[{index}] must be an object.")
        raw_path = _string_field(sheet, "path", f"multiview manifest global_sheets[{index}]")
        resolved = _resolve_manifest_artifact_path(manifest_path, raw_path)
        if resolved is None:
            raise ValueError(f"multiview manifest global_sheets[{index}] path does not exist: {raw_path}")
        global_sheet_paths.append(resolved)

    missing_global = sorted(set(global_sheet_paths) - reviewed_source_paths, key=str)
    if missing_global:
        labels = [_artifact_label(path, episode_dir) for path in missing_global]
        raise ValueError(
            "inspection_audit.reviewed_visual_sources missing global contact sheets from manifest: "
            f"{labels}"
        )

    boundary_frame_ids = {
        int(boundary["boundary_frame_id"])
        for boundary in decision.get("accepted_or_corrected_boundaries", []) or []
    }
    boundary_candidate_frame_ids = {
        int(boundary["source_candidate_frame_id"])
        for boundary in decision.get("accepted_or_corrected_boundaries", []) or []
        if boundary.get("source_candidate_frame_id") is not None
    }
    boundary_sample_ids = {
        int(boundary["boundary_sample_index"])
        for boundary in decision.get("accepted_or_corrected_boundaries", []) or []
    }
    referenced_local_paths: set[Path] = set()
    for index, sheet in enumerate(manifest.get("local_candidate_sheets", []) or []):
        if not isinstance(sheet, dict):
            raise ValueError(f"multiview manifest local_candidate_sheets[{index}] must be an object.")
        raw_path = _string_field(sheet, "path", f"multiview manifest local_candidate_sheets[{index}]")
        resolved = _resolve_manifest_artifact_path(manifest_path, raw_path)
        if resolved is None:
            raise ValueError(f"multiview manifest local_candidate_sheets[{index}] path does not exist: {raw_path}")

        candidate_frame = sheet.get("candidate_frame_id")
        candidate_sample = sheet.get("candidate_sample_index")
        if candidate_frame is not None and int(candidate_frame) in boundary_frame_ids | boundary_candidate_frame_ids:
            referenced_local_paths.add(resolved)
        if candidate_sample is not None and int(candidate_sample) in boundary_sample_ids:
            referenced_local_paths.add(resolved)

    audit = decision.get("inspection_audit")
    if isinstance(audit, dict):
        for check in audit.get("undersegmentation_checks", []) or []:
            if not isinstance(check, dict):
                continue
            for raw_sheet in check.get("reviewed_local_sheets", []) or []:
                resolved = _resolve_episode_artifact_path(episode_dir, str(raw_sheet))
                if resolved is not None:
                    referenced_local_paths.add(resolved.resolve())

    missing_local = sorted(referenced_local_paths - reviewed_source_paths, key=str)
    if missing_local:
        labels = [_artifact_label(path, episode_dir) for path in missing_local]
        raise ValueError(
            "inspection_audit.reviewed_visual_sources missing referenced local candidate sheets: "
            f"{labels}"
        )


def _validate_inspection_audit(
    *,
    decision: dict[str, Any],
    decision_path: Path,
    sampled_frame_ids: set[int],
    sampled_sample_ids: set[int],
) -> None:
    episode_dir = decision_path.parent.parent
    audit = decision.get("inspection_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"VLM decision missing inspection_audit: {decision_path}")

    reviewed_sources = _list_field(audit, "reviewed_visual_sources", "inspection_audit")
    reviewed_views: set[str] = set()
    reviewed_source_paths: set[Path] = set()
    for index, source in enumerate(reviewed_sources):
        if not isinstance(source, dict):
            raise ValueError(f"inspection_audit.reviewed_visual_sources[{index}] must be an object.")
        source_path = _string_field(source, "path", f"inspection_audit.reviewed_visual_sources[{index}]")
        resolved_source_path = _resolve_episode_artifact_path(episode_dir, source_path)
        if resolved_source_path is None:
            raise ValueError(
                f"inspection_audit.reviewed_visual_sources[{index}] path does not exist: {source_path}"
            )
        reviewed_source_paths.add(resolved_source_path.resolve())
        views = _views_from_record(source)
        if not _has_fixed_tri_view(views):
            raise ValueError(
                "inspection_audit.reviewed_visual_sources must include the fixed tri-view columns: "
                "head_color, hand_left_color, hand_right_color."
            )
        reviewed_views.update(views)
    if not _has_fixed_tri_view(list(reviewed_views)):
        raise ValueError("inspection_audit does not document the fixed tri-view review.")

    _validate_reviewed_sheet_coverage(
        decision=decision,
        episode_dir=episode_dir,
        reviewed_source_paths=reviewed_source_paths,
    )

    semantic_observations = _list_field(audit, "semantic_change_observations", "inspection_audit")
    observed_boundary_frames: set[int] = set()
    for index, observation in enumerate(semantic_observations):
        if not isinstance(observation, dict):
            raise ValueError(f"inspection_audit.semantic_change_observations[{index}] must be an object.")
        sample_index = int(observation["sample_index"])
        frame_id = int(observation["frame_id"])
        if sample_index not in sampled_sample_ids:
            raise ValueError(f"inspection_audit semantic observation {index} sample is not sampled: {sample_index}")
        if frame_id not in sampled_frame_ids:
            raise ValueError(f"inspection_audit semantic observation {index} frame is not sampled: {frame_id}")
        views = _views_from_record(observation)
        if not _has_fixed_tri_view(views):
            raise ValueError(
                f"inspection_audit semantic observation {index} must cite the fixed tri-view columns."
            )
        _string_field(observation, "visual_state_change", f"inspection_audit.semantic_change_observations[{index}]")
        decision_label = _string_field(observation, "decision", f"inspection_audit.semantic_change_observations[{index}]")
        if decision_label in {"accepted_boundary", "corrected_boundary", "new_boundary"}:
            observed_boundary_frames.add(frame_id)

    accepted_boundary_frames = {
        int(boundary["boundary_frame_id"])
        for boundary in decision.get("accepted_or_corrected_boundaries", [])
    }
    missing_boundary_frames = accepted_boundary_frames - observed_boundary_frames
    if missing_boundary_frames:
        raise ValueError(
            "inspection_audit semantic_change_observations missing accepted boundary frames: "
            f"{sorted(missing_boundary_frames)}"
        )

    trajectory_reference = decision_path.parent.parent / "multiview_pack" / "trajectory_reference.json"
    if trajectory_reference.is_file():
        trajectory_payload = _read_json(trajectory_reference)
        if trajectory_payload.get("selected_candidates"):
            trajectory_observations = _list_field(audit, "trajectory_change_observations", "inspection_audit")
            for index, observation in enumerate(trajectory_observations):
                if not isinstance(observation, dict):
                    raise ValueError(f"inspection_audit.trajectory_change_observations[{index}] must be an object.")
                int(observation["candidate_frame_id"])
                _string_field(observation, "event_type", f"inspection_audit.trajectory_change_observations[{index}]")
                _string_field(observation, "interpretation", f"inspection_audit.trajectory_change_observations[{index}]")
                metrics = observation.get("metrics")
                if not isinstance(metrics, dict) or not metrics:
                    raise ValueError(
                        f"inspection_audit.trajectory_change_observations[{index}] missing trajectory metrics."
                    )

    undersegmentation_checks = _list_field(audit, "undersegmentation_checks", "inspection_audit")
    checked_segments: set[int] = set()
    checks_by_segment: dict[int, dict[str, Any]] = {}
    for index, check in enumerate(undersegmentation_checks):
        if not isinstance(check, dict):
            raise ValueError(f"inspection_audit.undersegmentation_checks[{index}] must be an object.")
        segment_index = int(check["segment_index"])
        checked_segments.add(segment_index)
        checks_by_segment[segment_index] = check
        checked_frames = _list_field(check, "checked_frames", f"inspection_audit.undersegmentation_checks[{index}]")
        for frame_id in checked_frames:
            if int(frame_id) not in sampled_frame_ids:
                raise ValueError(f"inspection_audit undersegmentation check {index} frame is not sampled: {frame_id}")
        views = _views_from_record(check)
        if not _has_fixed_tri_view(views):
            raise ValueError(
                f"inspection_audit undersegmentation check {index} must cite the fixed tri-view columns."
            )
        _string_field(check, "result", f"inspection_audit.undersegmentation_checks[{index}]")

    segment_indices = set(range(len(decision["segments"])))
    missing_segments = segment_indices - checked_segments
    if missing_segments:
        raise ValueError(
            "inspection_audit undersegmentation_checks missing segment indices: "
            f"{sorted(missing_segments)}"
        )

    risk_segments = find_undersegmentation_risks(episode_dir=episode_dir, segments=decision["segments"])
    for risk in risk_segments:
        segment_index = int(risk["segment_index"])
        check = checks_by_segment.get(segment_index)
        if check is None:
            raise ValueError(f"inspection_audit missing high-risk undersegmentation check for segment {segment_index}.")
        reviewed_frames = {int(frame) for frame in check.get("internal_event_frames_reviewed", [])}
        required_frames = {int(frame) for frame in risk.get("required_internal_event_frames", [])}
        missing_frames = required_frames - reviewed_frames
        if missing_frames:
            raise ValueError(
                "inspection_audit high-risk segment "
                f"{segment_index} did not explicitly review internal event frames: {sorted(missing_frames)}"
            )
        local_sheets = check.get("reviewed_local_sheets", [])
        if not isinstance(local_sheets, list) or not local_sheets:
            raise ValueError(
                f"inspection_audit high-risk segment {segment_index} missing reviewed_local_sheets."
            )
        for raw_sheet in local_sheets:
            if _resolve_episode_artifact_path(episode_dir, str(raw_sheet)) is None:
                raise ValueError(
                    f"inspection_audit high-risk segment {segment_index} local sheet does not exist: {raw_sheet}"
                )
        split_decision = str(check.get("split_decision", "")).strip()
        if split_decision not in {"keep_single_segment", "split_inserted", "already_split"}:
            raise ValueError(
                f"inspection_audit high-risk segment {segment_index} missing valid split_decision."
            )


def validate_decision(decision_path: Path, pipeline_output_dir: Path) -> None:
    decision = _read_json(decision_path)
    sampled_frames = _metadata_samples(pipeline_output_dir)
    sampled_frame_ids = {int(frame["original_frame_id"]) for frame in sampled_frames}
    sampled_sample_ids = {int(frame["sample_index"]) for frame in sampled_frames}
    is_vlm_decision = str(decision.get("prompt_version", "")) == "codex_vlm_decision_v2"
    if not decision.get("segments"):
        raise ValueError(f"Decision has no segments: {decision_path}")

    previous_end_frame: int | None = None
    previous_end_sample: int | None = None
    for index, segment in enumerate(decision["segments"]):
        start_frame = int(segment["start_frame_id"])
        end_frame = int(segment["end_frame_id"])
        start_sample = int(segment["start_sample_index"])
        end_sample = int(segment["end_sample_index"])
        if start_sample not in sampled_sample_ids:
            raise ValueError(f"Segment {index} start_sample_index is not sampled: {start_sample}")
        if start_frame not in sampled_frame_ids:
            raise ValueError(f"Segment {index} start_frame_id is not sampled: {start_frame}")
        if end_frame < start_frame or end_sample < start_sample:
            raise ValueError(f"Segment {index} is not time-increasing.")
        if previous_end_frame is not None and start_frame not in {previous_end_frame, previous_end_frame + 1}:
            raise ValueError(f"Segment {index} is not continuous with the previous segment.")
        if previous_end_sample is not None and start_sample not in {previous_end_sample, previous_end_sample + 1}:
            raise ValueError(f"Segment {index} sample range is not continuous with the previous segment.")
        if not segment.get("selected_views"):
            raise ValueError(f"Segment {index} has no selected_views.")
        if is_vlm_decision:
            if not str(segment.get("left_hand_state", "")).strip():
                raise ValueError(f"Segment {index} has no left_hand_state.")
            if not str(segment.get("right_hand_state", "")).strip():
                raise ValueError(f"Segment {index} has no right_hand_state.")
            if not isinstance(segment.get("held_objects_by_hand"), dict):
                raise ValueError(f"Segment {index} has no held_objects_by_hand.")
        previous_end_frame = end_frame
        previous_end_sample = end_sample

    previous_boundary_frame: int | None = None
    for index, boundary in enumerate(decision.get("accepted_or_corrected_boundaries", [])):
        frame_id = int(boundary["boundary_frame_id"])
        sample_index = int(boundary["boundary_sample_index"])
        if frame_id not in sampled_frame_ids:
            raise ValueError(f"Boundary {index} frame_id is not a sampled base frame: {frame_id}")
        if sample_index not in sampled_sample_ids:
            raise ValueError(f"Boundary {index} sample_index is not sampled: {sample_index}")
        if previous_boundary_frame is not None and frame_id <= previous_boundary_frame:
            raise ValueError(f"Boundary {index} frame_id is not strictly increasing.")
        if not boundary.get("selected_views") and not boundary.get("view_evidence"):
            raise ValueError(f"Boundary {index} has no selected_views or view_evidence.")
        previous_boundary_frame = frame_id

    if is_vlm_decision:
        _validate_inspection_audit(
            decision=decision,
            decision_path=decision_path,
            sampled_frame_ids=sampled_frame_ids,
            sampled_sample_ids=sampled_sample_ids,
        )


def validate_final_output(final_dir: Path, pipeline_output_dir: Path) -> None:
    final_output = _read_json(final_dir / "final_segments.json")
    if not final_output.get("segments"):
        raise ValueError(f"Final output has no segments: {final_dir / 'final_segments.json'}")
    sampled_frame_ids = {int(frame["original_frame_id"]) for frame in _metadata_samples(pipeline_output_dir)}
    previous_boundary_frame: int | None = None
    for index, boundary in enumerate(final_output.get("boundaries", [])):
        frame_id = int(boundary["boundary_frame_id"])
        if frame_id not in sampled_frame_ids:
            raise ValueError(f"Final boundary {index} is not a sampled base frame: {frame_id}")
        if previous_boundary_frame is not None and frame_id <= previous_boundary_frame:
            raise ValueError(f"Final boundary {index} frame_id is not strictly increasing.")
        previous_boundary_frame = frame_id


def summarize(
    observations_dir: Path,
    task_info_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    per_episode: list[EpisodeEval],
    statuses: list[EpisodeRunStatus],
) -> dict[str, Any]:
    total_tp = sum(item.boundary_tp for item in per_episode)
    total_fp = sum(item.boundary_fp for item in per_episode)
    total_fn = sum(item.boundary_fn for item in per_episode)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)
    maes = [item.boundary_mae_frames for item in per_episode if item.boundary_mae_frames is not None]

    return {
        "observations_dir": str(observations_dir),
        "task_info": str(task_info_path),
        "output_dir": str(output_dir),
        "camera": args.camera,
        "base_camera_role": "head_color is the base timeline/output reference; multi-view sheets provide decision evidence.",
        "provider": "codex",
        "model": args.model,
        "sample_fps": args.sample_fps,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "boundary_tolerance_frames": args.boundary_tolerance,
        "views_requested": _requested_views(args.views, args.camera),
        "vision_source_weights": {
            "overhead_video": 0.50,
            "tri_view_video": 0.25,
            "pixel_cv_prior": 0.25,
        },
        "overhead_view": args.overhead_view,
        "tri_view_views": args.tri_view_views,
        "proprio_root": args.proprio_root,
        "episodes_discovered": len(statuses),
        "episodes_evaluated": len(per_episode),
        "failed_or_skipped_episodes": [asdict(item) for item in statuses if item.status != "evaluated"],
        "episode_statuses": [asdict(item) for item in statuses],
        "micro_boundary_precision": precision,
        "micro_boundary_recall": recall,
        "micro_boundary_f1": f1,
        "mean_boundary_mae_frames": statistics.mean(maes) if maes else None,
        "mean_coarse_framewise_accuracy": statistics.mean(item.coarse_framewise_accuracy for item in per_episode) if per_episode else 0.0,
        "mean_coarse_segment_f1_10": statistics.mean(item.coarse_segment_f1_10 for item in per_episode) if per_episode else 0.0,
        "mean_coarse_segment_f1_25": statistics.mean(item.coarse_segment_f1_25 for item in per_episode) if per_episode else 0.0,
        "mean_coarse_segment_f1_50": statistics.mean(item.coarse_segment_f1_50 for item in per_episode) if per_episode else 0.0,
        "mean_predicted_segment_count": statistics.mean(item.predicted_segment_count for item in per_episode) if per_episode else 0.0,
        "mean_predicted_boundary_count": statistics.mean(item.predicted_boundary_count for item in per_episode) if per_episode else 0.0,
    }


def process_episode(
    api_dir: Path,
    episode_dir: Path,
    task_lookup: dict[int, dict],
    task_output_dir: Path,
    args: argparse.Namespace,
    requested_views: list[str],
) -> tuple[EpisodeEval | None, EpisodeRunStatus]:
    episode_id = int(episode_dir.name)
    existing_views, missing_views = existing_episode_views(episode_dir=episode_dir, requested_views=requested_views)
    episode_output_dir = task_output_dir / "episodes" / str(episode_id)
    base_dir = episode_output_dir / "base_head_pipeline"
    pack_dir = episode_output_dir / "multiview_pack"
    decision_path = episode_output_dir / "decision" / "codex_vlm_decision.json"
    final_dir = episode_output_dir / "final"
    status = EpisodeRunStatus(
        episode_id=episode_id,
        status="failed",
        reason="",
        episode_dir=str(episode_dir),
        existing_views=existing_views,
        missing_views=missing_views,
        base_head_pipeline=str(base_dir),
        multiview_pack=str(pack_dir),
        decision=str(decision_path),
        final=str(final_dir),
    )

    if episode_id not in task_lookup:
        status.status = "skipped"
        status.reason = "episode_id not present in task_info"
        return None, status
    if _base_view(args.camera) not in existing_views:
        status.reason = f"missing required base camera: {episode_dir / 'videos' / args.camera}"
        return None, status

    try:
        proprio_root = Path(args.proprio_root).resolve() if args.proprio_root else None
        trajectory_csv = _trajectory_csv_for_episode(proprio_root=proprio_root, episode_dir=episode_dir)
        run_base_head_pipeline(
            api_dir=api_dir,
            video_path=episode_dir / "videos" / args.camera,
            output_dir=base_dir,
            model=args.model,
            sample_fps=args.sample_fps,
            window_size=args.window_size,
            window_stride=args.window_stride,
            force=args.force,
        )
        run_multiview_pack(
            api_dir=api_dir,
            episode_dir=episode_dir,
            output_dir=pack_dir,
            sample_fps=args.sample_fps,
            views=existing_views,
            candidate_count=args.candidate_count,
            candidate_min_gap=args.candidate_min_gap,
            overhead_view=args.overhead_view,
            tri_view_views=args.tri_view_views,
            trajectory_csv=trajectory_csv,
            trajectory_candidate_count=args.trajectory_candidate_count,
            trajectory_candidate_min_gap_frames=args.trajectory_candidate_min_gap_frames,
            force=args.force,
        )
        status.decision_mode = ensure_decision(
            base_pipeline_dir=base_dir,
            multiview_pack_dir=pack_dir,
            decision_path=decision_path,
            existing_views=existing_views,
            require_existing=not args.allow_bootstrap_decisions,
        )
        run_apply_decision(
            api_dir=api_dir,
            pipeline_output_dir=base_dir,
            decision_path=decision_path,
            output_dir=final_dir,
            allow_unreviewed_decision=status.decision_mode != "existing_codex_vlm_decision",
        )
        validate_final_output(final_dir=final_dir, pipeline_output_dir=base_dir)
        result = evaluate_episode(
            episode_id=episode_id,
            video_path=episode_dir / "videos" / args.camera,
            predicted_output_dir=final_dir,
            gt_item=task_lookup[episode_id],
            tolerance=args.boundary_tolerance,
        )
    except Exception as exc:
        status.reason = str(exc)
        return None, status

    status.status = "evaluated"
    status.reason = "ok"
    return result, status


def main() -> None:
    args = parse_args()
    api_dir = Path(__file__).resolve().parents[2]
    repo_root = api_dir.parent
    observations_dir = _resolve_input_path(args.observations_dir, repo_root=repo_root)
    task_info_path = _resolve_input_path(args.task_info, repo_root=repo_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not observations_dir.is_dir():
        raise FileNotFoundError(f"Missing observations directory: {observations_dir}")
    if not task_info_path.is_file():
        raise FileNotFoundError(f"Missing external task_info file: {task_info_path}")

    task_lookup = load_task_info(task_info_path)
    episode_dirs = find_episode_dirs(observations_dir)
    requested_views = _requested_views(args.views, args.camera)
    per_episode: list[EpisodeEval] = []
    statuses: list[EpisodeRunStatus] = []

    for episode_dir in episode_dirs:
        print(f"[run] episode={episode_dir.name} camera={args.camera}", flush=True)
        result, status = process_episode(
            api_dir=api_dir,
            episode_dir=episode_dir,
            task_lookup=task_lookup,
            task_output_dir=output_dir,
            args=args,
            requested_views=requested_views,
        )
        statuses.append(status)
        if result is not None:
            per_episode.append(result)
            print(f"[ok] episode={episode_dir.name} decision_mode={status.decision_mode}", flush=True)
        else:
            print(f"[{status.status}] episode={episode_dir.name} reason={status.reason}", flush=True)

    _write_json(output_dir / "per_episode.json", [asdict(item) for item in per_episode])
    summary = summarize(
        observations_dir=observations_dir,
        task_info_path=task_info_path,
        output_dir=output_dir,
        args=args,
        per_episode=per_episode,
        statuses=statuses,
    )
    _write_json(output_dir / "summary.json", summary)
    print(f"[done] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
