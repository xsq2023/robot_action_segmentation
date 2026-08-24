from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from robot_tas.action_set import format_action_set_contract, parse_action_set
from robot_tas.cache import ensure_dir, read_json, write_json
from robot_tas.schemas import MergedBoundary, SampledFrame, VideoMetadata
from robot_tas.visual_reference import frame_reference_line, transition_change_score, transition_reference_lines


@dataclass(frozen=True, slots=True)
class ContactSheetConfig:
    global_sheet_size: int = 24
    global_columns: int = 4
    global_thumb_width: int = 220
    global_thumb_height: int = 165
    local_radius: int = 5
    local_columns: int = 3
    local_thumb_width: int = 320
    local_thumb_height: int = 240
    top_visual_changes: int = 12


def load_sampled_artifacts(pipeline_output_dir: Path) -> tuple[VideoMetadata, list[SampledFrame]]:
    metadata_path = pipeline_output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing pipeline metadata: {metadata_path}")

    payload = read_json(metadata_path)
    metadata = VideoMetadata.model_validate(payload["video"])
    sampled_frames = [SampledFrame.model_validate(item) for item in payload["sampled_frames"]]
    if not sampled_frames:
        raise ValueError(f"No sampled frames found in {metadata_path}")
    return metadata, sampled_frames


def load_reference_boundaries(pipeline_output_dir: Path) -> list[MergedBoundary]:
    final_path = pipeline_output_dir / "final_segments.json"
    if final_path.exists():
        payload = read_json(final_path)
        return [MergedBoundary.model_validate(item) for item in payload.get("boundaries", [])]

    merged_path = pipeline_output_dir / "merged_boundaries.json"
    if merged_path.exists():
        return [MergedBoundary.model_validate(item) for item in read_json(merged_path)]
    return []


def _resolve_image_path(pipeline_output_dir: Path, frame: SampledFrame) -> Path:
    path = pipeline_output_dir / frame.image_path
    if not path.exists():
        raise FileNotFoundError(f"Missing sampled frame image: {path}")
    return path


def _sheet_label(frame: SampledFrame) -> str:
    return (
        f"s={frame.sample_index:03d} "
        f"f={frame.original_frame_id:04d} "
        f"t={frame.timestamp_seconds:.2f}s"
    )


def write_contact_sheet(
    frames: list[SampledFrame],
    pipeline_output_dir: Path,
    output_path: Path,
    thumb_size: tuple[int, int],
    columns: int,
) -> Path:
    if not frames:
        raise ValueError("frames must not be empty")
    if columns <= 0:
        raise ValueError("columns must be positive")

    ensure_dir(output_path.parent)
    thumb_width, thumb_height = thumb_size
    label_height = 26
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, frame in enumerate(frames):
        image = Image.open(_resolve_image_path(pipeline_output_dir, frame)).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)

        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        tile = Image.new("RGB", thumb_size, (245, 245, 245))
        tile.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y + thumb_height, x + thumb_width - 1, y + thumb_height + label_height - 1], fill=(20, 20, 20))
        draw.text((x + 4, y + thumb_height + 6), _sheet_label(frame), fill="white", font=font)

    sheet.save(output_path, quality=92)
    return output_path


def _safe_episode_name(metadata: VideoMetadata, pipeline_output_dir: Path) -> str:
    video_path = Path(metadata.path)
    if video_path.parent.name == "videos":
        return video_path.parent.parent.name
    if re.fullmatch(r"\d+", pipeline_output_dir.name):
        return pipeline_output_dir.name
    return pipeline_output_dir.name


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.relative_to(output_dir))


def _frame_positions(sampled_frames: list[SampledFrame]) -> dict[int, int]:
    return {frame.sample_index: position for position, frame in enumerate(sampled_frames)}


def _nearest_frame(sampled_frames: list[SampledFrame], frame_id: int) -> SampledFrame:
    return min(sampled_frames, key=lambda frame: abs(frame.original_frame_id - frame_id))


def _top_visual_change_records(sampled_frames: list[SampledFrame], top_k: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for previous, current in zip(sampled_frames, sampled_frames[1:]):
        score = transition_change_score(previous, current)
        records.append(
            {
                "sample_index": current.sample_index,
                "original_frame_id": current.original_frame_id,
                "time": current.timestamp_seconds,
                "previous_sample_index": previous.sample_index,
                "visual_change_score": score,
                "motion_score": current.motion_score,
                "luma_delta": abs(current.mean_luma - previous.mean_luma),
            }
        )
    return sorted(records, key=lambda item: (-item["visual_change_score"], item["sample_index"]))[:top_k]


def build_codex_vlm_pack(
    pipeline_output_dir: Path,
    output_dir: Path,
    prompt_path: Path,
    config: ContactSheetConfig = ContactSheetConfig(),
    action_set: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    pipeline_output_dir = pipeline_output_dir.resolve()
    output_dir = ensure_dir(output_dir.resolve())
    contact_dir = ensure_dir(output_dir / "contact_sheets")

    metadata, sampled_frames = load_sampled_artifacts(pipeline_output_dir)
    reference_boundaries = load_reference_boundaries(pipeline_output_dir)
    normalized_action_set = parse_action_set(",".join(action_set)) if action_set else parse_action_set(None)
    source_prompt_text = prompt_path.read_text(encoding="utf-8").rstrip()
    prompt_text = (
        f"{source_prompt_text}\n\n"
        f"{format_action_set_contract(normalized_action_set)}\n"
    )
    shutil.copyfile(prompt_path, output_dir / "prompt_source.txt")
    (output_dir / "prompt_used.txt").write_text(prompt_text, encoding="utf-8")

    global_sheets: list[dict[str, Any]] = []
    for start in range(0, len(sampled_frames), config.global_sheet_size):
        chunk = sampled_frames[start : start + config.global_sheet_size]
        output_path = contact_dir / f"global_{chunk[0].sample_index:03d}_{chunk[-1].sample_index:03d}.jpg"
        if force or not output_path.exists():
            write_contact_sheet(
                frames=chunk,
                pipeline_output_dir=pipeline_output_dir,
                output_path=output_path,
                thumb_size=(config.global_thumb_width, config.global_thumb_height),
                columns=config.global_columns,
            )
        global_sheets.append(
            {
                "path": _relative_to_output(output_path, output_dir),
                "start_sample_index": chunk[0].sample_index,
                "end_sample_index": chunk[-1].sample_index,
                "start_frame_id": chunk[0].original_frame_id,
                "end_frame_id": chunk[-1].original_frame_id,
            }
        )

    positions = _frame_positions(sampled_frames)
    local_candidate_sheets: list[dict[str, Any]] = []
    for boundary in reference_boundaries:
        nearest = _nearest_frame(sampled_frames, boundary.boundary_frame_id)
        center = positions[nearest.sample_index]
        start = max(0, center - config.local_radius)
        end = min(len(sampled_frames) - 1, center + config.local_radius)
        chunk = sampled_frames[start : end + 1]
        output_path = (
            contact_dir
            / f"local_candidate_frame_{boundary.boundary_frame_id:04d}_samples_{chunk[0].sample_index:03d}_{chunk[-1].sample_index:03d}.jpg"
        )
        if force or not output_path.exists():
            write_contact_sheet(
                frames=chunk,
                pipeline_output_dir=pipeline_output_dir,
                output_path=output_path,
                thumb_size=(config.local_thumb_width, config.local_thumb_height),
                columns=config.local_columns,
            )
        local_candidate_sheets.append(
            {
                "path": _relative_to_output(output_path, output_dir),
                "candidate_boundary_frame_id": boundary.boundary_frame_id,
                "candidate_boundary_sample_index": boundary.boundary_sample_index,
                "nearest_sample_index": nearest.sample_index,
                "nearest_frame_id": nearest.original_frame_id,
                "start_sample_index": chunk[0].sample_index,
                "end_sample_index": chunk[-1].sample_index,
                "reference_before_action": boundary.before_action,
                "reference_after_action": boundary.after_action,
                "reference_transition_type": boundary.transition_type,
                "reference_confidence": boundary.confidence,
            }
        )

    visual_reference_lines = transition_reference_lines(sampled_frames, top_k=config.top_visual_changes)
    vlm_input = {
        "episode_id": _safe_episode_name(metadata, pipeline_output_dir),
        "pipeline_output_dir": str(pipeline_output_dir),
        "video": metadata.model_dump(mode="json"),
        "frame_reference": [frame_reference_line(frame) for frame in sampled_frames],
        "visual_reference": {
            "description": "Non-binding low-level motion/luma hints. Use for recall only.",
            "top_adjacent_frame_changes": visual_reference_lines,
            "top_adjacent_frame_change_records": _top_visual_change_records(
                sampled_frames, top_k=config.top_visual_changes
            ),
        },
        "action_set": normalized_action_set,
        "heuristic_boundary_candidates": [
            boundary.model_dump(mode="json") for boundary in reference_boundaries
        ],
        "contact_sheets": {
            "global": global_sheets,
            "local_candidates": local_candidate_sheets,
        },
    }

    prompt_markdown = build_prompt_markdown(prompt_text=prompt_text, vlm_input=vlm_input)
    write_json(output_dir / "vlm_input.json", vlm_input)
    (output_dir / "vlm_prompt.md").write_text(prompt_markdown, encoding="utf-8")

    manifest = {
        "episode_id": vlm_input["episode_id"],
        "prompt": "prompt_used.txt",
        "prompt_source": "prompt_source.txt",
        "vlm_prompt": "vlm_prompt.md",
        "vlm_input": "vlm_input.json",
        "action_set": normalized_action_set,
        "global_sheet_count": len(global_sheets),
        "local_candidate_sheet_count": len(local_candidate_sheets),
        "reference_boundary_count": len(reference_boundaries),
        "global_sheets": global_sheets,
        "local_candidate_sheets": local_candidate_sheets,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_prompt_markdown(prompt_text: str, vlm_input: dict[str, Any]) -> str:
    global_sheets = "\n".join(
        f"- {item['path']}: samples {item['start_sample_index']}..{item['end_sample_index']}, "
        f"frames {item['start_frame_id']}..{item['end_frame_id']}"
        for item in vlm_input["contact_sheets"]["global"]
    )
    local_sheets = "\n".join(
        f"- {item['path']}: candidate frame {item['candidate_boundary_frame_id']}, "
        f"context samples {item['start_sample_index']}..{item['end_sample_index']}"
        for item in vlm_input["contact_sheets"]["local_candidates"]
    )
    heuristic_boundaries = [
        item["boundary_frame_id"] for item in vlm_input["heuristic_boundary_candidates"]
    ]
    visual_reference = "\n".join(vlm_input["visual_reference"]["top_adjacent_frame_changes"])

    return (
        "# Codex-as-VLM TAS Decision Pack\n\n"
        "## System Prompt\n\n"
        f"```text\n{prompt_text.strip()}\n```\n\n"
        "## Episode\n\n"
        f"- episode_id: `{vlm_input['episode_id']}`\n"
        f"- video: `{vlm_input['video']['path']}`\n"
        f"- fps: `{vlm_input['video']['fps']}`\n"
        f"- total_frames: `{vlm_input['video']['total_frames']}`\n"
        f"- sample_fps: `{vlm_input['video']['sample_fps']}`\n\n"
        "## Contact Sheets\n\n"
        "Inspect the global sheets in order first, then inspect local candidate sheets.\n\n"
        f"{global_sheets}\n\n"
        "## Local Candidate Sheets\n\n"
        f"{local_sheets or '- none'}\n\n"
        "## Visual Reference\n\n"
        "These are non-binding low-level hints. Reject peaks without semantic image evidence.\n\n"
        f"heuristic_boundary_frames: `{heuristic_boundaries}`\n\n"
        f"```text\n{visual_reference or 'none'}\n```\n\n"
        "## Full Machine-Readable Input\n\n"
        "Use `vlm_input.json` for exact frame IDs, timestamps, heuristic candidates, and sheet paths.\n\n"
        "Return the JSON schema requested by the system prompt. Do not read task_info or ground truth before locking the decision.\n"
    )
