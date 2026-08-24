from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any


def _vector_columns(prefix: str, width: int) -> list[str]:
    return [f"{prefix}[{index}]" for index in range(width)]


def _available_columns(columns: list[str], prefix: str) -> list[str]:
    return [column for column in columns if column.startswith(f"{prefix}[")]


def _column_index(column: str) -> int:
    start = column.rfind("[")
    end = column.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Column does not contain an index: {column}")
    return int(column[start + 1 : end])


def _load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Trajectory CSV has no header: {csv_path}")
        columns = list(reader.fieldnames)
        rows: list[dict[str, float]] = []
        for raw in reader:
            row: dict[str, float] = {}
            for column, value in raw.items():
                if value in (None, ""):
                    continue
                try:
                    row[column] = float(value)
                except ValueError:
                    continue
            rows.append(row)
    if not rows:
        raise ValueError(f"Trajectory CSV has no rows: {csv_path}")
    return columns, rows


def _column_range(rows: list[dict[str, float]], column: str) -> float:
    values = [row[column] for row in rows if column in row]
    if not values:
        return 0.0
    return max(values) - min(values)


def _total_motion(rows: list[dict[str, float]], columns: list[str]) -> float:
    total = 0.0
    for previous, current in zip(rows, rows[1:]):
        if any(column not in previous or column not in current for column in columns):
            continue
        total += math.sqrt(sum((current[column] - previous[column]) ** 2 for column in columns))
    return total


def _infer_active_effector(columns: list[str], rows: list[dict[str, float]]) -> int:
    effector_columns = sorted(_available_columns(columns, "state.effector.position"), key=_column_index)
    if effector_columns:
        ranges = [_column_range(rows, column) for column in effector_columns]
        return max(range(len(ranges)), key=lambda index: ranges[index])
    return 0


def _infer_vector_chunk(
    columns: list[str],
    rows: list[dict[str, float]],
    prefix: str,
    chunk_width: int,
    preferred_chunk: int,
) -> list[str]:
    available = sorted(_available_columns(columns, prefix), key=_column_index)
    if len(available) < chunk_width:
        return []
    chunks = [available[start : start + chunk_width] for start in range(0, len(available), chunk_width)]
    complete_chunks = [chunk for chunk in chunks if len(chunk) == chunk_width]
    if not complete_chunks:
        return []
    if preferred_chunk < len(complete_chunks) and _total_motion(rows, complete_chunks[preferred_chunk]) > 0.0:
        return complete_chunks[preferred_chunk]
    return max(complete_chunks, key=lambda chunk: _total_motion(rows, chunk))


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index] or 1.0


def _group_consecutive(indices: list[int], max_gap: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for index in indices:
        if not groups or index - groups[-1][-1] > max_gap:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _group_signed_changes(changes: list[tuple[int, int]], max_gap: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current_sign: int | None = None
    for index, sign in changes:
        if not groups or current_sign != sign or index - groups[-1][-1] > max_gap:
            groups.append([index])
            current_sign = sign
        else:
            groups[-1].append(index)
    return groups


def _nearest_sample(frame_id: int, frame_ids: list[int]) -> tuple[int, int]:
    if not frame_ids:
        return frame_id, frame_id
    sample_index, nearest_frame = min(
        enumerate(frame_ids),
        key=lambda item: (abs(item[1] - frame_id), item[0]),
    )
    return sample_index, nearest_frame


def _row_vector(row: dict[str, float], columns: list[str]) -> list[float]:
    return [row[column] for column in columns if column in row]


def _delta(previous: dict[str, float], current: dict[str, float], columns: list[str]) -> float:
    if not columns or any(column not in previous or column not in current for column in columns):
        return 0.0
    return math.sqrt(sum((current[column] - previous[column]) ** 2 for column in columns))


def _direction(before: float, after: float, close_label: str, open_label: str) -> str:
    return close_label if after > before else open_label


def _candidate(
    *,
    frame_id: int,
    frame_ids: list[int],
    fps: float,
    event_type: str,
    score: float,
    evidence: list[str],
    semantic_hint: str,
    signals: dict[str, float],
) -> dict[str, Any]:
    sample_index, nearest_frame_id = _nearest_sample(frame_id, frame_ids)
    return {
        "candidate_sample_index": sample_index,
        "candidate_frame_id": nearest_frame_id,
        "raw_trajectory_frame_id": frame_id,
        "candidate_time_seconds": nearest_frame_id / fps if fps > 0 else 0.0,
        "event_type": event_type,
        "trajectory_score": score,
        "semantic_hint": semantic_hint,
        "timing_refinement_hint": (
            "Use this as a local timing anchor only after visual evidence confirms the semantic transition."
        ),
        "evidence": evidence,
        "signals": signals,
    }


def _dedupe_candidates(candidates: list[dict[str, Any]], min_gap_frames: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-float(item["trajectory_score"]), int(item["raw_trajectory_frame_id"])),
    ):
        frame_id = int(candidate["raw_trajectory_frame_id"])
        if all(abs(frame_id - int(chosen["raw_trajectory_frame_id"])) >= min_gap_frames for chosen in selected):
            selected.append(candidate)
    return selected


def _select_balanced_candidates(
    candidates: list[dict[str, Any]],
    *,
    candidate_count: int,
    min_gap_frames: int,
) -> list[dict[str, Any]]:
    if candidate_count <= 0:
        return []

    gripper_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("event_type", "")).startswith("gripper_")
    ]
    motion_candidates = [
        candidate
        for candidate in candidates
        if not str(candidate.get("event_type", "")).startswith("gripper_")
    ]
    selected = _dedupe_candidates(gripper_candidates, min_gap_frames=min_gap_frames)[: max(1, candidate_count // 2)]
    selected_frames = {int(candidate["raw_trajectory_frame_id"]) for candidate in selected}

    remaining_motion = [
        candidate
        for candidate in motion_candidates
        if all(
            abs(int(candidate["raw_trajectory_frame_id"]) - int(chosen["raw_trajectory_frame_id"])) >= min_gap_frames
            for chosen in selected
        )
    ]
    selected.extend(
        candidate
        for candidate in _dedupe_candidates(remaining_motion, min_gap_frames=min_gap_frames)
        if int(candidate["raw_trajectory_frame_id"]) not in selected_frames
    )
    return sorted(selected[:candidate_count], key=lambda item: int(item["candidate_frame_id"]))


def build_trajectory_reference(
    csv_path: Path,
    *,
    frame_ids: list[int],
    fps: float,
    candidate_count: int = 12,
    min_gap_frames: int = 30,
    top_record_count: int = 24,
) -> dict[str, Any]:
    columns, rows = _load_rows(csv_path)
    active_effector = _infer_active_effector(columns, rows)
    end_position_columns = _infer_vector_chunk(columns, rows, "state.end.position", 3, active_effector)
    end_orientation_columns = _infer_vector_chunk(columns, rows, "state.end.orientation", 4, active_effector)
    joint_position_columns = _infer_vector_chunk(columns, rows, "state.joint.position", 7, active_effector)
    action_gripper_column = f"action.effector.position[{active_effector}]"
    state_gripper_column = f"state.effector.position[{active_effector}]"

    records: list[dict[str, Any]] = []
    end_deltas: list[float] = []
    orientation_deltas: list[float] = []
    joint_deltas: list[float] = []
    gripper_command_deltas: list[float] = []
    gripper_state_deltas: list[float] = []
    for frame_id, (previous, current) in enumerate(zip(rows, rows[1:]), start=1):
        end_delta = _delta(previous, current, end_position_columns)
        orientation_delta = _delta(previous, current, end_orientation_columns)
        joint_delta = _delta(previous, current, joint_position_columns)
        gripper_command_delta = abs(current.get(action_gripper_column, 0.0) - previous.get(action_gripper_column, 0.0))
        gripper_state_delta = abs(current.get(state_gripper_column, 0.0) - previous.get(state_gripper_column, 0.0))
        end_deltas.append(end_delta)
        orientation_deltas.append(orientation_delta)
        joint_deltas.append(joint_delta)
        gripper_command_deltas.append(gripper_command_delta)
        gripper_state_deltas.append(gripper_state_delta)
        records.append(
            {
                "raw_trajectory_frame_id": frame_id,
                "end_position_delta": end_delta,
                "end_orientation_delta": orientation_delta,
                "joint_delta": joint_delta,
                "gripper_command_delta": gripper_command_delta,
                "gripper_state_delta": gripper_state_delta,
            }
        )

    end_scale = _percentile(end_deltas, 0.95)
    orientation_scale = _percentile(orientation_deltas, 0.95)
    joint_scale = _percentile(joint_deltas, 0.95)
    command_scale = _percentile([value for value in gripper_command_deltas if value > 0], 0.95)
    state_scale = _percentile([value for value in gripper_state_deltas if value > 0], 0.95)
    for record in records:
        score = (
            0.30 * min(float(record["end_position_delta"]) / end_scale, 3.0)
            + 0.15 * min(float(record["end_orientation_delta"]) / orientation_scale, 3.0)
            + 0.30 * min(float(record["joint_delta"]) / joint_scale, 3.0)
            + 0.15 * min(float(record["gripper_state_delta"]) / state_scale, 3.0)
            + 0.10 * min(float(record["gripper_command_delta"]) / command_scale, 3.0)
        )
        record["trajectory_score"] = score
        sample_index, nearest_frame_id = _nearest_sample(int(record["raw_trajectory_frame_id"]), frame_ids)
        record["candidate_sample_index"] = sample_index
        record["candidate_frame_id"] = nearest_frame_id
        record["candidate_time_seconds"] = nearest_frame_id / fps if fps > 0 else 0.0

    candidates: list[dict[str, Any]] = []
    ranked_motion = sorted(records, key=lambda item: (-float(item["trajectory_score"]), int(item["raw_trajectory_frame_id"])))
    for record in ranked_motion[: max(candidate_count * 2, top_record_count)]:
        frame_id = int(record["raw_trajectory_frame_id"])
        candidates.append(
            _candidate(
                frame_id=frame_id,
                frame_ids=frame_ids,
                fps=fps,
                event_type="trajectory_motion_peak",
                score=float(record["trajectory_score"]),
                semantic_hint="possible phase change; verify object/contact semantics visually",
                evidence=[
                    (
                        "active end/joint motion has a local high score "
                        f"at trajectory frame {frame_id}"
                    )
                ],
                signals={
                    "end_position_delta": float(record["end_position_delta"]),
                    "end_orientation_delta": float(record["end_orientation_delta"]),
                    "joint_delta": float(record["joint_delta"]),
                    "gripper_command_delta": float(record["gripper_command_delta"]),
                    "gripper_state_delta": float(record["gripper_state_delta"]),
                },
            )
        )

    command_threshold = max(0.02, _percentile([value for value in gripper_command_deltas if value > 0], 0.50) * 0.5)
    command_changes = [
        (
            index,
            1 if rows[index].get(action_gripper_column, 0.0) > rows[index - 1].get(action_gripper_column, 0.0) else -1,
        )
        for index, delta in enumerate(gripper_command_deltas, start=1)
        if delta >= command_threshold and action_gripper_column in rows[index] and action_gripper_column in rows[index - 1]
    ]
    for group in _group_signed_changes(command_changes, max_gap=2):
        start, end = group[0], group[-1]
        before = rows[start - 1].get(action_gripper_column, 0.0)
        after = rows[end].get(action_gripper_column, 0.0)
        event_type = _direction(before, after, "gripper_command_close", "gripper_command_open")
        candidates.append(
            _candidate(
                frame_id=start,
                frame_ids=frame_ids,
                fps=fps,
                event_type=event_type,
                score=2.0 + abs(after - before),
                semantic_hint=(
                    "possible grasp subphase" if event_type.endswith("close") else "possible release/place subphase"
                ),
                evidence=[
                    f"{event_type} from {before:.3f} to {after:.3f} over trajectory frames {start}..{end}"
                ],
                signals={
                    "gripper_command_before": before,
                    "gripper_command_after": after,
                },
            )
        )

    state_values = [row.get(state_gripper_column, 0.0) for row in rows if state_gripper_column in row]
    state_range = (max(state_values) - min(state_values)) if state_values else 0.0
    state_threshold = max(5.0, state_range * 0.06)
    state_changes = [
        (
            index,
            1 if rows[index].get(state_gripper_column, 0.0) > rows[index - 1].get(state_gripper_column, 0.0) else -1,
        )
        for index, delta in enumerate(gripper_state_deltas, start=1)
        if delta >= state_threshold and state_gripper_column in rows[index] and state_gripper_column in rows[index - 1]
    ]
    for group in _group_signed_changes(state_changes, max_gap=16):
        start, end = group[0], group[-1]
        before = rows[start - 1].get(state_gripper_column, 0.0)
        after = rows[end].get(state_gripper_column, 0.0)
        event_type = _direction(before, after, "gripper_state_closing", "gripper_state_opening")
        anchor = end if event_type.endswith("opening") else start
        candidates.append(
            _candidate(
                frame_id=anchor,
                frame_ids=frame_ids,
                fps=fps,
                event_type=event_type,
                score=1.7 + min(abs(after - before) / (state_range or 1.0), 1.0),
                semantic_hint=(
                    "possible stable grasp timing" if event_type.endswith("closing") else "possible release completion timing"
                ),
                evidence=[
                    f"{event_type} from {before:.3f} to {after:.3f} over trajectory frames {start}..{end}"
                ],
                signals={
                    "gripper_state_before": before,
                    "gripper_state_after": after,
                },
            )
        )

    selected_candidates = _select_balanced_candidates(
        candidates,
        candidate_count=candidate_count,
        min_gap_frames=min_gap_frames,
    )
    top_records = sorted(records, key=lambda item: (-float(item["trajectory_score"]), int(item["raw_trajectory_frame_id"])))[:top_record_count]
    compact_records = [
        {
            key: record[key]
            for key in (
                "raw_trajectory_frame_id",
                "candidate_sample_index",
                "candidate_frame_id",
                "candidate_time_seconds",
                "trajectory_score",
                "end_position_delta",
                "end_orientation_delta",
                "joint_delta",
                "gripper_command_delta",
                "gripper_state_delta",
            )
        }
        for record in top_records
    ]

    return {
        "description": (
            "Trajectory-derived non-visual hints for candidate recall, local timing refinement, "
            "and joint evidence. Visual semantics remain primary."
        ),
        "timeseries_csv": str(csv_path),
        "row_count": len(rows),
        "active_signals": {
            "effector_index": active_effector,
            "action_gripper_column": action_gripper_column if action_gripper_column in columns else None,
            "state_gripper_column": state_gripper_column if state_gripper_column in columns else None,
            "end_position_columns": end_position_columns,
            "end_orientation_columns": end_orientation_columns,
            "joint_position_columns": joint_position_columns,
        },
        "signal_scales": {
            "end_position_delta_p95": end_scale,
            "end_orientation_delta_p95": orientation_scale,
            "joint_delta_p95": joint_scale,
            "gripper_command_delta_scale": command_scale,
            "gripper_state_delta_scale": state_scale,
            "gripper_state_median": median(state_values) if state_values else 0.0,
        },
        "selected_candidates": selected_candidates,
        "top_records": compact_records,
    }


def trajectory_reference_lines(reference: dict[str, Any], max_lines: int = 16) -> list[str]:
    lines: list[str] = []
    for candidate in reference.get("selected_candidates", [])[:max_lines]:
        evidence = "; ".join(candidate.get("evidence", []))
        lines.append(
            "trajectory_candidate "
            f"sample_index={candidate['candidate_sample_index']} "
            f"frame={candidate['candidate_frame_id']} "
            f"raw_frame={candidate['raw_trajectory_frame_id']} "
            f"time={candidate['candidate_time_seconds']:.3f}s "
            f"type={candidate['event_type']} "
            f"score={candidate['trajectory_score']:.3f} "
            f"hint={candidate['semantic_hint']} "
            f"evidence={evidence}"
        )
    return lines
