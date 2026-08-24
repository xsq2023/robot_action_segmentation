from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from robot_tas.cache import ensure_dir, read_json, write_json
from robot_tas.sampler import (
    _sample_video_frames_ffmpeg,
    _sample_video_frames_opencv,
    compute_sample_frame_ids,
)
from robot_tas.schemas import SampledFrame, VideoMetadata
from robot_tas.trajectory_reference import build_trajectory_reference
from robot_tas.video_io import read_video_metadata
from robot_tas.visual_reference import transition_change_score


TRI_VIEW_SHEET_COLUMNS = ("head_color", "hand_left_color", "hand_right_color")
CONTACT_SHEET_LAYOUT = "enlarged_tri_view_frame_rows"

DEFAULT_VIEW_ORDER = [
    "head_color",
    "hand_left_color",
    "hand_right_color",
]

VIEW_WEIGHTS = {
    "head_color": 1.15,
    "hand_left_color": 1.25,
    "hand_right_color": 1.25,
}

VISION_SOURCE_WEIGHTS = {
    "overhead_video": 0.50,
    "tri_view_video": 0.25,
    "pixel_cv_prior": 0.25,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare synchronized multi-view contact sheets and fused CV boundary priors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing a videos/ subdirectory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for sampled views and contact sheets.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Synchronized sampling FPS.")
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEW_ORDER),
        help="Comma-separated camera names without .mp4.",
    )
    parser.add_argument("--global-sheet-size", type=int, default=12, help="Multi-view sample tiles per global sheet.")
    parser.add_argument(
        "--global-columns",
        type=int,
        default=3,
        help="Deprecated compatibility argument; enlarged tri-view sheets are always frame rows.",
    )
    parser.add_argument("--local-radius", type=int, default=4, help="Sample radius around each fused CV candidate.")
    parser.add_argument(
        "--local-columns",
        type=int,
        default=3,
        help="Deprecated compatibility argument; enlarged tri-view sheets are always frame rows.",
    )
    parser.add_argument("--view-thumb-width", type=int, default=360, help="Width of each enlarged view column cell.")
    parser.add_argument("--view-thumb-height", type=int, default=270, help="Height of each enlarged view column cell.")
    parser.add_argument("--candidate-count", type=int, default=16, help="Number of fused CV candidate neighborhoods.")
    parser.add_argument("--candidate-min-gap", type=int, default=5, help="Minimum sample gap between fused CV candidates.")
    parser.add_argument(
        "--overhead-view",
        default="head_color",
        help="Primary overhead/time-axis view. Its visual evidence contributes 50%% to the visual prior.",
    )
    parser.add_argument(
        "--tri-view-views",
        default="head_color,hand_left_color,hand_right_color",
        help="Comma-separated tri-view set used as the 25%% multi-view visual source.",
    )
    parser.add_argument(
        "--trajectory-csv",
        default=None,
        help="Optional proprio/trajectory timeseries.csv used for candidate recall, timing refinement, and evidence.",
    )
    parser.add_argument(
        "--trajectory-candidate-count",
        type=int,
        default=12,
        help="Maximum trajectory-derived candidate neighborhoods to add.",
    )
    parser.add_argument(
        "--trajectory-candidate-min-gap-frames",
        type=int,
        default=30,
        help="Minimum raw-frame gap between trajectory candidates.",
    )
    parser.add_argument(
        "--decoder",
        choices=["ffmpeg", "opencv-fallback"],
        default="ffmpeg",
        help="Frame decoder. ffmpeg avoids noisy OpenCV AV1 hardware-decoder warnings on this dataset.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute sampled frames and contact sheets.")
    return parser.parse_args()


def _video_path(episode_dir: Path, view: str) -> Path:
    return episode_dir / "videos" / f"{view}.mp4"


def _load_cached_view(metadata_path: Path, frame_ids: list[int]) -> tuple[VideoMetadata, list[SampledFrame]] | None:
    if not metadata_path.exists():
        return None
    cached = read_json(metadata_path)
    sampled_frames = [SampledFrame.model_validate(item) for item in cached["sampled_frames"]]
    if [frame.original_frame_id for frame in sampled_frames] != frame_ids:
        return None
    return VideoMetadata.model_validate(cached["video"]), sampled_frames


def _sample_view(
    video_path: Path,
    view_dir: Path,
    sample_fps: float,
    frame_ids: list[int],
    force: bool,
    decoder: str,
) -> tuple[VideoMetadata, list[SampledFrame]]:
    metadata_path = view_dir / "metadata.json"
    if not force:
        cached = _load_cached_view(metadata_path, frame_ids)
        if cached is not None:
            return cached

    if force and (view_dir / "sampled_frames").exists():
        shutil.rmtree(view_dir / "sampled_frames")

    metadata = read_video_metadata(video_path=video_path, sample_fps=sample_fps)
    sampled_frames_dir = ensure_dir(view_dir / "sampled_frames")
    if decoder == "ffmpeg":
        sampled_frames = _sample_video_frames_ffmpeg(
            video_path=video_path,
            metadata=metadata,
            sampled_frames_dir=sampled_frames_dir,
            target_frame_ids=frame_ids,
        )
    else:
        try:
            sampled_frames = _sample_video_frames_opencv(
                video_path=video_path,
                metadata=metadata,
                sampled_frames_dir=sampled_frames_dir,
                target_frame_ids=frame_ids,
            )
        except RuntimeError:
            sampled_frames = _sample_video_frames_ffmpeg(
                video_path=video_path,
                metadata=metadata,
                sampled_frames_dir=sampled_frames_dir,
                target_frame_ids=frame_ids,
            )

    write_json(
        metadata_path,
        {
            "video": metadata.model_dump(mode="json"),
            "sampled_frames": [frame.model_dump(mode="json") for frame in sampled_frames],
        },
    )
    return metadata, sampled_frames


def _resolve_frame_path(view_dir: Path, frame: SampledFrame) -> Path:
    path = view_dir / frame.image_path
    if not path.exists():
        raise FileNotFoundError(f"Missing sampled frame image: {path}")
    return path


def _sheet_view_columns(frames_by_view: dict[str, list[SampledFrame]]) -> list[str]:
    missing = [view for view in TRI_VIEW_SHEET_COLUMNS if view not in frames_by_view]
    if missing:
        raise ValueError(
            "Enlarged VLM contact sheets require the fixed tri-view columns "
            f"{list(TRI_VIEW_SHEET_COLUMNS)}; missing {missing}."
        )
    return list(TRI_VIEW_SHEET_COLUMNS)


def _make_frame_row(
    sample_position: int,
    view_names: list[str],
    frames_by_view: dict[str, list[SampledFrame]],
    view_dirs: dict[str, Path],
    view_thumb_size: tuple[int, int],
    row_note: str = "",
) -> Image.Image:
    thumb_width, thumb_height = view_thumb_size
    sample_label_height = 34
    row_width = len(view_names) * thumb_width
    row_height = sample_label_height + thumb_height
    tile = Image.new("RGB", (row_width, row_height), (245, 245, 245))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()

    reference = frames_by_view[view_names[0]][sample_position]
    draw.rectangle([0, 0, row_width - 1, sample_label_height - 1], fill=(20, 20, 20))
    note = f" {row_note}" if row_note else ""
    draw.text(
        (5, 7),
        f"s={reference.sample_index:03d} f={reference.original_frame_id:04d} t={reference.timestamp_seconds:.2f}s{note}",
        fill="white",
        font=font,
    )

    for view_index, view in enumerate(view_names):
        frame = frames_by_view[view][sample_position]
        image = Image.open(_resolve_frame_path(view_dirs[view], frame)).convert("RGB")
        image.thumbnail(view_thumb_size, Image.Resampling.LANCZOS)

        x = view_index * thumb_width
        y = sample_label_height
        cell = Image.new("RGB", view_thumb_size, (235, 235, 235))
        cell.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        tile.paste(cell, (x, y))
        draw.rectangle([x, y, x + thumb_width - 1, y + thumb_height - 1], outline=(35, 35, 35))

    return tile


def _expected_sheet_size(
    sample_count: int,
    view_count: int,
    view_thumb_size: tuple[int, int],
) -> tuple[int, int]:
    thumb_width, thumb_height = view_thumb_size
    header_height = 42
    row_label_height = 34
    margin = 8
    return (
        margin * 2 + view_count * thumb_width,
        header_height + sample_count * (row_label_height + thumb_height) + margin,
    )


def _needs_contact_sheet_rewrite(
    output_path: Path,
    sample_count: int,
    view_count: int,
    view_thumb_size: tuple[int, int],
) -> bool:
    if not output_path.exists():
        return True
    expected_size = _expected_sheet_size(
        sample_count=sample_count,
        view_count=view_count,
        view_thumb_size=view_thumb_size,
    )
    try:
        with Image.open(output_path) as image:
            return image.size != expected_size
    except OSError:
        return True


def _write_multiview_contact_sheet(
    sample_positions: list[int],
    view_names: list[str],
    frames_by_view: dict[str, list[SampledFrame]],
    view_dirs: dict[str, Path],
    output_path: Path,
    view_thumb_size: tuple[int, int],
    columns: int,
) -> Path:
    if not sample_positions:
        raise ValueError("sample_positions must not be empty")
    ensure_dir(output_path.parent)
    _ = columns

    thumb_width, thumb_height = view_thumb_size
    header_height = 42
    row_label_height = 34
    margin = 8
    sheet_width, sheet_height = _expected_sheet_size(
        sample_count=len(sample_positions),
        view_count=len(view_names),
        view_thumb_size=view_thumb_size,
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, 8), f"{CONTACT_SHEET_LAYOUT}: rows=sampled_frames columns=tri-view", fill=(0, 0, 0), font=font)
    for view_index, view in enumerate(view_names):
        x = margin + view_index * thumb_width + 6
        draw.text((x, header_height - 24), view, fill=(0, 0, 0), font=font)

    y = header_height
    for position in sample_positions:
        row = _make_frame_row(
            sample_position=position,
            view_names=view_names,
            frames_by_view=frames_by_view,
            view_dirs=view_dirs,
            view_thumb_size=view_thumb_size,
            row_note="",
        )
        sheet.paste(row, (margin, y))
        y += row_label_height + thumb_height

    sheet.save(output_path, quality=92)
    return output_path


def _pixel_prior_score(per_view_scores: dict[str, float]) -> tuple[float, float, float]:
    weighted_total = 0.0
    weight_total = 0.0
    for view, score in per_view_scores.items():
        weight = VIEW_WEIGHTS.get(view, 0.75)
        weighted_total += score * weight
        weight_total += weight

    sorted_scores = sorted(per_view_scores.values(), reverse=True)
    top3_mean = sum(sorted_scores[:3]) / min(3, len(sorted_scores))
    weighted_mean = weighted_total / weight_total if weight_total else 0.0
    pixel_score = 0.65 * weighted_mean + 0.35 * top3_mean
    return pixel_score, weighted_mean, top3_mean


def _vision_prior_records(
    view_names: list[str],
    frames_by_view: dict[str, list[SampledFrame]],
    overhead_view: str,
    tri_view_names: list[str],
) -> list[dict[str, Any]]:
    sample_count = min(len(frames) for frames in frames_by_view.values())
    records: list[dict[str, Any]] = []
    if overhead_view not in frames_by_view:
        raise ValueError(f"Overhead view {overhead_view!r} is not in selected views: {view_names}")
    available_tri_views = [view for view in tri_view_names if view in frames_by_view]
    if not available_tri_views:
        raise ValueError(f"None of the tri-view views are available: {tri_view_names}")

    for position in range(1, sample_count):
        per_view_scores: dict[str, float] = {}
        for view in view_names:
            score = transition_change_score(frames_by_view[view][position - 1], frames_by_view[view][position])
            per_view_scores[view] = score

        overhead_score = per_view_scores[overhead_view]
        tri_view_score = sum(per_view_scores[view] for view in available_tri_views) / len(available_tri_views)
        pixel_score, weighted_mean, top3_mean = _pixel_prior_score(per_view_scores)
        fused_score = (
            VISION_SOURCE_WEIGHTS["overhead_video"] * overhead_score
            + VISION_SOURCE_WEIGHTS["tri_view_video"] * tri_view_score
            + VISION_SOURCE_WEIGHTS["pixel_cv_prior"] * pixel_score
        )
        frame = frames_by_view[view_names[0]][position]
        records.append(
            {
                "sample_index": frame.sample_index,
                "original_frame_id": frame.original_frame_id,
                "time_seconds": frame.timestamp_seconds,
                "fused_score": fused_score,
                "source_scores": {
                    "overhead_video": overhead_score,
                    "tri_view_video": tri_view_score,
                    "pixel_cv_prior": pixel_score,
                },
                "weighted_mean_score": weighted_mean,
                "top3_mean_score": top3_mean,
                "per_view_scores": per_view_scores,
                "top_views": [
                    {"view": view, "score": score}
                    for view, score in sorted(per_view_scores.items(), key=lambda item: item[1], reverse=True)[:3]
                ],
            }
        )
    return records


def _select_candidate_records(records: list[dict[str, Any]], candidate_count: int, min_gap: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (-item["fused_score"], item["sample_index"])):
        if all(abs(record["sample_index"] - chosen["sample_index"]) >= min_gap for chosen in candidates):
            candidates.append(record)
        if len(candidates) >= candidate_count:
            break
    return sorted(candidates, key=lambda item: item["sample_index"])


def _merge_candidate_sources(
    visual_candidates: list[dict[str, Any]],
    trajectory_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}

    for candidate in visual_candidates:
        sample_index = int(candidate["sample_index"])
        merged[sample_index] = {
            "sample_index": sample_index,
            "original_frame_id": int(candidate["original_frame_id"]),
            "time_seconds": float(candidate["time_seconds"]),
            "candidate_sources": ["visual_prior"],
            "visual_prior": candidate,
            "trajectory_reference": None,
        }

    for candidate in trajectory_candidates:
        sample_index = int(candidate["candidate_sample_index"])
        existing = merged.get(sample_index)
        if existing is None:
            merged[sample_index] = {
                "sample_index": sample_index,
                "original_frame_id": int(candidate["candidate_frame_id"]),
                "time_seconds": float(candidate["candidate_time_seconds"]),
                "candidate_sources": ["trajectory"],
                "visual_prior": None,
                "trajectory_reference": candidate,
            }
            continue

        existing["candidate_sources"].append("trajectory")
        existing["trajectory_reference"] = candidate

    return [merged[index] for index in sorted(merged)]


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    contact_dir = ensure_dir(output_dir / "contact_sheets")
    views_dir = ensure_dir(output_dir / "views")
    view_names = [view.strip() for view in args.views.split(",") if view.strip()]
    if not view_names:
        raise ValueError("At least one view is required.")
    tri_view_names = [view.strip() for view in args.tri_view_views.split(",") if view.strip()]

    video_paths = {view: _video_path(episode_dir, view) for view in view_names}
    missing = [str(path) for path in video_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing video files: {missing}")

    metadata_by_view = {view: read_video_metadata(video_path=path, sample_fps=args.sample_fps) for view, path in video_paths.items()}
    fps_values = {round(metadata.fps, 6) for metadata in metadata_by_view.values()}
    if len(fps_values) != 1:
        raise ValueError(f"Expected synchronized videos with one FPS, got: {sorted(fps_values)}")

    reference_metadata = metadata_by_view[view_names[0]]
    min_total_frames = min(metadata.total_frames for metadata in metadata_by_view.values())
    frame_ids = compute_sample_frame_ids(
        total_frames=min_total_frames,
        native_fps=reference_metadata.fps,
        sample_fps=args.sample_fps,
    )

    frames_by_view: dict[str, list[SampledFrame]] = {}
    view_dirs: dict[str, Path] = {}
    for view in view_names:
        view_dir = ensure_dir(views_dir / view)
        view_dirs[view] = view_dir
        _, frames_by_view[view] = _sample_view(
            video_path=video_paths[view],
            view_dir=view_dir,
            sample_fps=args.sample_fps,
            frame_ids=frame_ids,
            force=args.force,
            decoder=args.decoder,
        )

    sheet_view_names = _sheet_view_columns(frames_by_view)
    sheet_thumb_size = (args.view_thumb_width, args.view_thumb_height)

    global_sheets: list[dict[str, Any]] = []
    sample_count = len(frame_ids)
    for start in range(0, sample_count, args.global_sheet_size):
        positions = list(range(start, min(sample_count, start + args.global_sheet_size)))
        first = frames_by_view[view_names[0]][positions[0]]
        last = frames_by_view[view_names[0]][positions[-1]]
        output_path = contact_dir / f"multiview_global_{first.sample_index:03d}_{last.sample_index:03d}.jpg"
        if args.force or _needs_contact_sheet_rewrite(
            output_path=output_path,
            sample_count=len(positions),
            view_count=len(sheet_view_names),
            view_thumb_size=sheet_thumb_size,
        ):
            _write_multiview_contact_sheet(
                sample_positions=positions,
                view_names=sheet_view_names,
                frames_by_view=frames_by_view,
                view_dirs=view_dirs,
                output_path=output_path,
                view_thumb_size=sheet_thumb_size,
                columns=args.global_columns,
            )
        global_sheets.append(
            {
                "path": str(output_path.relative_to(output_dir)),
                "layout": CONTACT_SHEET_LAYOUT,
                "columns": sheet_view_names,
                "row_axis": "sampled_frames",
                "view_thumb_size": [args.view_thumb_width, args.view_thumb_height],
                "start_sample_index": first.sample_index,
                "end_sample_index": last.sample_index,
                "start_frame_id": first.original_frame_id,
                "end_frame_id": last.original_frame_id,
            }
        )

    cv_records = _vision_prior_records(
        view_names=view_names,
        frames_by_view=frames_by_view,
        overhead_view=args.overhead_view,
        tri_view_names=tri_view_names,
    )
    visual_candidates = _select_candidate_records(
        records=cv_records,
        candidate_count=args.candidate_count,
        min_gap=args.candidate_min_gap,
    )

    trajectory_reference: dict[str, Any] | None = None
    trajectory_candidates: list[dict[str, Any]] = []
    if args.trajectory_csv:
        trajectory_reference = build_trajectory_reference(
            Path(args.trajectory_csv).resolve(),
            frame_ids=frame_ids,
            fps=reference_metadata.fps,
            candidate_count=args.trajectory_candidate_count,
            min_gap_frames=args.trajectory_candidate_min_gap_frames,
        )
        trajectory_candidates = list(trajectory_reference["selected_candidates"])

    combined_candidates = _merge_candidate_sources(
        visual_candidates=visual_candidates,
        trajectory_candidates=trajectory_candidates,
    )

    local_sheets: list[dict[str, Any]] = []
    for candidate in combined_candidates:
        center = candidate["sample_index"]
        start = max(0, center - args.local_radius)
        end = min(sample_count - 1, center + args.local_radius)
        positions = list(range(start, end + 1))
        output_path = contact_dir / f"multiview_local_candidate_sample_{center:03d}_frames_{frame_ids[start]:04d}_{frame_ids[end]:04d}.jpg"
        if args.force or _needs_contact_sheet_rewrite(
            output_path=output_path,
            sample_count=len(positions),
            view_count=len(sheet_view_names),
            view_thumb_size=sheet_thumb_size,
        ):
            _write_multiview_contact_sheet(
                sample_positions=positions,
                view_names=sheet_view_names,
                frames_by_view=frames_by_view,
                view_dirs=view_dirs,
                output_path=output_path,
                view_thumb_size=sheet_thumb_size,
                columns=args.local_columns,
            )
        local_sheets.append(
            {
                "path": str(output_path.relative_to(output_dir)),
                "layout": CONTACT_SHEET_LAYOUT,
                "columns": sheet_view_names,
                "row_axis": "sampled_frames",
                "view_thumb_size": [args.view_thumb_width, args.view_thumb_height],
                "candidate_sample_index": center,
                "candidate_frame_id": candidate["original_frame_id"],
                "candidate_time_seconds": candidate["time_seconds"],
                "candidate_sources": candidate["candidate_sources"],
                "fused_score": (
                    candidate["visual_prior"]["fused_score"]
                    if candidate["visual_prior"] is not None
                    else None
                ),
                "source_scores": (
                    candidate["visual_prior"]["source_scores"]
                    if candidate["visual_prior"] is not None
                    else None
                ),
                "top_views": (
                    candidate["visual_prior"]["top_views"]
                    if candidate["visual_prior"] is not None
                    else []
                ),
                "trajectory_event_type": (
                    candidate["trajectory_reference"]["event_type"]
                    if candidate["trajectory_reference"] is not None
                    else None
                ),
                "trajectory_score": (
                    candidate["trajectory_reference"]["trajectory_score"]
                    if candidate["trajectory_reference"] is not None
                    else None
                ),
                "trajectory_evidence": (
                    candidate["trajectory_reference"]["evidence"]
                    if candidate["trajectory_reference"] is not None
                    else []
                ),
                "start_sample_index": start,
                "end_sample_index": end,
            }
        )

    write_json(
        output_dir / "fused_cv_reference.json",
        {
            "description": (
                "Visual adjacent-frame changes across synchronized camera views. "
                "Scores are non-binding recall hints; semantic visual evidence is still required."
            ),
            "vision_source_weights": VISION_SOURCE_WEIGHTS,
            "overhead_view": args.overhead_view,
            "tri_view_views": tri_view_names,
            "pixel_prior_view_weights": VIEW_WEIGHTS,
            "top_records": sorted(cv_records, key=lambda item: (-item["fused_score"], item["sample_index"]))[: max(args.candidate_count, 20)],
            "selected_candidates": visual_candidates,
        },
    )
    if trajectory_reference is not None:
        write_json(
            output_dir / "trajectory_reference.json",
            trajectory_reference,
        )
    write_json(
        output_dir / "candidate_reference.json",
        {
            "description": "Merged visual and trajectory candidate sources used to build local sheets.",
            "visual_candidate_count": len(visual_candidates),
            "trajectory_candidate_count": len(trajectory_candidates),
            "combined_candidate_count": len(combined_candidates),
            "combined_candidates": combined_candidates,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "episode_id": episode_dir.name,
            "episode_dir": str(episode_dir),
            "sample_fps": args.sample_fps,
            "fps": reference_metadata.fps,
            "min_total_frames": min_total_frames,
            "sample_count": sample_count,
            "views": view_names,
            "contact_sheet_layout": CONTACT_SHEET_LAYOUT,
            "contact_sheet_columns": sheet_view_names,
            "contact_sheet_row_axis": "sampled_frames",
            "contact_sheet_view_thumb_size": [args.view_thumb_width, args.view_thumb_height],
            "vision_source_weights": VISION_SOURCE_WEIGHTS,
            "overhead_view": args.overhead_view,
            "tri_view_views": tri_view_names,
            "trajectory_reference": "trajectory_reference.json" if trajectory_reference is not None else None,
            "candidate_reference": "candidate_reference.json",
            "global_sheets": global_sheets,
            "local_candidate_sheets": local_sheets,
        },
    )
    print(
        f"Prepared multiview pack: views={len(view_names)} samples={sample_count} "
        f"global_sheets={len(global_sheets)} candidates={len(combined_candidates)}"
    )


if __name__ == "__main__":
    main()
