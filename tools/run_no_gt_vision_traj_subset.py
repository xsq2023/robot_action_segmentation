#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_tas.cli.evaluate_agibot_task import find_episode_dirs
from robot_tas.cli.evaluate_multiview_codex_task import (
    _requested_views,
    ensure_decision,
    existing_episode_views,
    run_apply_decision,
    run_base_head_pipeline,
    run_multiview_pack,
    validate_final_output,
)
from robot_tas.cache import write_json


DEFAULT_VIEWS = "head_color,hand_left_color,hand_right_color"


@dataclass(slots=True)
class EpisodeStatus:
    task_id: str
    episode_id: str
    status: str
    reason: str
    episode_dir: str
    output_dir: str
    trajectory_csv: str | None
    trajectory_mode: str
    final_segments: str | None = None
    annotated_video: str | None = None
    timeline_html: str | None = None
    flat_annotated_video: str | None = None
    flat_timeline_html: str | None = None
    segment_count: int | None = None
    boundary_count: int | None = None
    reused_from: str | None = None


@dataclass(slots=True)
class TaskStatus:
    task_id: str
    requested_episodes: int
    available_episodes: int
    selected_episodes: list[str]
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the no-GT vision+trajectory pipeline for a fixed number of local episodes per AgiBot task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations-root", default="dataset/AgiBotWorld-Alpha/observations")
    parser.add_argument("--proprio-root", default="dataset/AgiBotWorld-Alpha/proprio_stats_extracted")
    parser.add_argument("--output-root", default="outputs/codex_vlm_fresh_multiview")
    parser.add_argument("--reuse-output-root", default="")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids. Defaults to every local observations task.")
    parser.add_argument("--camera", default="head_color.mp4")
    parser.add_argument("--model", default="codex-local")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--views", default=DEFAULT_VIEWS)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--candidate-min-gap", type=int, default=5)
    parser.add_argument("--overhead-view", default="head_color")
    parser.add_argument("--tri-view-views", default="head_color,hand_left_color,hand_right_color")
    parser.add_argument("--trajectory-candidate-count", type=int, default=12)
    parser.add_argument("--trajectory-candidate-min-gap-frames", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-bootstrap-decisions",
        action="store_true",
        help=(
            "Allow deterministic placeholder decisions when decision/codex_vlm_decision.json is missing. "
            "This is for pack inspection only and is not a final semantic result."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def resolve_api_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def hardlink_or_copy(src: Path | str, dst: Path | str) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.samefile(dst):
                return
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copytree_linked(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst, copy_function=hardlink_or_copy)


def read_counts(final_segments_path: Path) -> tuple[int, int]:
    payload = json.loads(final_segments_path.read_text(encoding="utf-8"))
    return len(payload.get("segments", [])), len(payload.get("boundaries", []))


def task_ids_from_root(observations_root: Path, raw_task_ids: str | None) -> list[str]:
    if raw_task_ids:
        return [item.strip() for item in raw_task_ids.split(",") if item.strip()]
    return sorted((path.name for path in observations_root.iterdir() if path.is_dir()), key=lambda item: int(item))


def trajectory_csv(proprio_root: Path, task_id: str, episode_id: str) -> Path | None:
    path = proprio_root / task_id / episode_id / "timeseries.csv"
    return path if path.is_file() else None


def selected_episode_dirs(observations_root: Path, task_id: str, limit: int) -> list[Path]:
    task_dir = observations_root / task_id
    if not task_dir.is_dir():
        return []
    return [
        episode_dir
        for episode_dir in find_episode_dirs(task_dir)
        if (episode_dir / "videos" / "head_color.mp4").is_file()
    ][:limit]


def final_paths(output_root: Path, task_id: str, episode_id: str) -> tuple[Path, Path, Path, Path, Path]:
    episode_output = output_root / f"task_{task_id}" / "episodes" / episode_id
    final_dir = episode_output / "final"
    flat_dir = output_root / "annotated_videos"
    prefix = f"task_{task_id}_episode_{episode_id}"
    return (
        episode_output,
        final_dir / "final_segments.json",
        final_dir / "annotated_actions.mp4",
        final_dir / "timeline.html",
        flat_dir / f"{prefix}_annotated_actions.mp4",
    )


def link_flat_outputs(output_root: Path, task_id: str, episode_id: str) -> tuple[Path | None, Path | None]:
    episode_output, final_segments, video, timeline, flat_video = final_paths(output_root, task_id, episode_id)
    del episode_output, final_segments
    flat_timeline = output_root / "annotated_videos" / f"task_{task_id}_episode_{episode_id}_timeline.html"
    if video.is_file():
        hardlink_or_copy(video, flat_video)
    if timeline.is_file():
        hardlink_or_copy(timeline, flat_timeline)
    return (flat_video if flat_video.is_file() else None, flat_timeline if flat_timeline.is_file() else None)


def reuse_existing_episode(
    reuse_root: Path | None,
    output_root: Path,
    task_id: str,
    episode_id: str,
) -> Path | None:
    if reuse_root is None:
        return None
    src_episode = reuse_root / f"task_{task_id}" / "episodes" / episode_id
    src_final = src_episode / "final" / "final_segments.json"
    if not src_final.is_file():
        return None
    dst_episode = output_root / f"task_{task_id}" / "episodes" / episode_id
    copytree_linked(src_episode, dst_episode)
    link_flat_outputs(output_root=output_root, task_id=task_id, episode_id=episode_id)
    return src_episode


def run_episode(
    *,
    api_dir: Path,
    output_root: Path,
    reuse_root: Path | None,
    proprio_root: Path,
    task_id: str,
    episode_dir: Path,
    args: argparse.Namespace,
    requested_views: list[str],
) -> EpisodeStatus:
    episode_id = episode_dir.name
    episode_output = output_root / f"task_{task_id}" / "episodes" / episode_id
    final_dir = episode_output / "final"
    final_segments = final_dir / "final_segments.json"
    trajectory = trajectory_csv(proprio_root, task_id, episode_id)
    trajectory_mode = "caption_plus_trajectory" if trajectory else "vision_only_no_timeseries"

    status = EpisodeStatus(
        task_id=task_id,
        episode_id=episode_id,
        status="failed",
        reason="",
        episode_dir=str(episode_dir),
        output_dir=str(episode_output),
        trajectory_csv=str(trajectory) if trajectory else None,
        trajectory_mode=trajectory_mode,
    )

    if final_segments.is_file() and not args.force:
        status.status = "completed"
        status.reason = "existing_final_reused"
    else:
        reused = None if args.force else reuse_existing_episode(reuse_root, output_root, task_id, episode_id)
        if reused is not None:
            status.status = "completed"
            status.reason = "copied_from_reuse_output_root"
            status.reused_from = str(reused)
        else:
            existing_views, missing_views = existing_episode_views(episode_dir=episode_dir, requested_views=requested_views)
            if "head_color" not in existing_views:
                status.status = "skipped"
                status.reason = f"missing head_color.mp4; missing_views={missing_views}"
                return status

            base_dir = episode_output / "base_head_pipeline"
            pack_dir = episode_output / "multiview_pack"
            decision_path = episode_output / "decision" / "codex_vlm_decision.json"
            try:
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
                    trajectory_csv=trajectory,
                    trajectory_candidate_count=args.trajectory_candidate_count,
                    trajectory_candidate_min_gap_frames=args.trajectory_candidate_min_gap_frames,
                    force=args.force,
                )
                decision_mode = ensure_decision(
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
                    allow_unreviewed_decision=decision_mode != "existing_codex_vlm_decision",
                )
                validate_final_output(final_dir=final_dir, pipeline_output_dir=base_dir)
            except Exception as exc:
                status.reason = str(exc)
                return status
            status.status = "completed"
            status.reason = decision_mode

    if final_segments.is_file():
        segments, boundaries = read_counts(final_segments)
        flat_video, flat_timeline = link_flat_outputs(output_root=output_root, task_id=task_id, episode_id=episode_id)
        status.final_segments = str(final_segments)
        status.annotated_video = str(final_dir / "annotated_actions.mp4")
        status.timeline_html = str(final_dir / "timeline.html")
        status.flat_annotated_video = str(flat_video) if flat_video else None
        status.flat_timeline_html = str(flat_timeline) if flat_timeline else None
        status.segment_count = segments
        status.boundary_count = boundaries
    return status


def main() -> None:
    args = parse_args()
    observations_root = resolve_repo_path(args.observations_root)
    proprio_root = resolve_repo_path(args.proprio_root)
    output_root = resolve_api_path(args.output_root)
    reuse_root = resolve_api_path(args.reuse_output_root) if args.reuse_output_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["ROBOT_TAS_SKIP_RUN_TAS_VISUALIZATION"] = "1"

    requested_views = _requested_views(args.views, args.camera)
    task_statuses: list[TaskStatus] = []
    episode_statuses: list[EpisodeStatus] = []

    for task_id in task_ids_from_root(observations_root, args.task_ids):
        episodes = selected_episode_dirs(observations_root, task_id, args.episodes_per_task)
        available_count = len(find_episode_dirs(observations_root / task_id)) if (observations_root / task_id).is_dir() else 0
        task_statuses.append(
            TaskStatus(
                task_id=task_id,
                requested_episodes=args.episodes_per_task,
                available_episodes=available_count,
                selected_episodes=[episode.name for episode in episodes],
                status="ok" if len(episodes) >= args.episodes_per_task else "partial_insufficient_local_episodes",
            )
        )
        for episode_dir in episodes:
            print(f"[run] task={task_id} episode={episode_dir.name}", flush=True)
            status = run_episode(
                api_dir=PROJECT_ROOT,
                output_root=output_root,
                reuse_root=reuse_root,
                proprio_root=proprio_root,
                task_id=task_id,
                episode_dir=episode_dir,
                args=args,
                requested_views=requested_views,
            )
            episode_statuses.append(status)
            print(f"[{status.status}] task={task_id} episode={episode_dir.name} {status.reason}", flush=True)

    completed = [status for status in episode_statuses if status.status == "completed"]
    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "flat_video_dir": str(output_root / "annotated_videos"),
        "use_gt": False,
        "requested_episodes_per_task": args.episodes_per_task,
        "task_count": len(task_statuses),
        "episode_count": len(episode_statuses),
        "completed": len(completed),
        "vision_plus_trajectory_completed": sum(
            1 for status in completed if status.trajectory_mode == "caption_plus_trajectory"
        ),
        "vision_only_completed": sum(
            1 for status in completed if status.trajectory_mode == "vision_only_no_timeseries"
        ),
        "config": {
            "camera": args.camera,
            "model": args.model,
            "sample_fps": args.sample_fps,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "views": args.views,
            "candidate_count": args.candidate_count,
            "candidate_min_gap": args.candidate_min_gap,
            "overhead_view": args.overhead_view,
            "tri_view_views": args.tri_view_views,
            "proprio_root": str(proprio_root),
            "reuse_output_root": str(reuse_root) if reuse_root else None,
        },
        "tasks": [asdict(status) for status in task_statuses],
        "episode_statuses": [asdict(status) for status in episode_statuses],
    }
    write_json(output_root / "no_gt_run_summary.json", summary)
    print(f"[done] completed={len(completed)}/{len(episode_statuses)} summary={output_root / 'no_gt_run_summary.json'}")


if __name__ == "__main__":
    main()
