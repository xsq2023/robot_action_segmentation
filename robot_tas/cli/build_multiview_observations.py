from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from robot_tas.cache import ensure_dir, write_json


REQUIRED_VIEWS = [
    "head_color",
    "hand_left_color",
    "hand_right_color",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build synchronized multi-view CV packs for every episode under an observations root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations-root", required=True, help="Root directory containing task observation folders.")
    parser.add_argument("--output-root", default="outputs/codex_multiview_probe", help="Output root for per-episode packs.")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--candidate-count", type=int, default=18)
    parser.add_argument("--candidate-min-gap", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Rebuild packs even when manifest.json exists.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Record incomplete episodes instead of failing before the run.")
    return parser.parse_args()


def find_episode_video_dirs(observations_root: Path) -> list[Path]:
    return sorted(path for path in observations_root.rglob("videos") if path.is_dir())


def missing_required_videos(videos_dir: Path) -> list[str]:
    return [f"{view}.mp4" for view in REQUIRED_VIEWS if not (videos_dir / f"{view}.mp4").exists()]


def task_id_for_video_dir(observations_root: Path, videos_dir: Path) -> str:
    relative = videos_dir.relative_to(observations_root)
    if not relative.parts:
        raise ValueError(f"Cannot infer task id for {videos_dir}")
    return relative.parts[0]


def run_builder(
    episode_dir: Path,
    output_dir: Path,
    sample_fps: float,
    candidate_count: int,
    candidate_min_gap: int,
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
        "--candidate-count",
        str(candidate_count),
        "--candidate-min-gap",
        str(candidate_min_gap),
        "--decoder",
        "ffmpeg",
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    observations_root = Path(args.observations_root).resolve()
    output_root = ensure_dir(Path(args.output_root).resolve())

    all_videos_dirs = find_episode_video_dirs(observations_root)
    incomplete: list[dict[str, Any]] = []
    videos_dirs: list[Path] = []
    for videos_dir in all_videos_dirs:
        missing = missing_required_videos(videos_dir)
        if missing:
            incomplete.append(
                {
                    "task_id": task_id_for_video_dir(observations_root, videos_dir),
                    "episode_id": videos_dir.parent.name,
                    "episode_dir": str(videos_dir.parent),
                    "missing_videos": missing,
                }
            )
        else:
            videos_dirs.append(videos_dir)
    if incomplete and not args.allow_incomplete:
        raise ValueError(f"Found incomplete episode video directories: {incomplete[:5]}")

    episode_ids = [path.parent.name for path in videos_dirs]
    duplicates = sorted({episode_id for episode_id in episode_ids if episode_ids.count(episode_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate episode ids under observations root; use task-aware output layout: {duplicates}")

    inventory: list[dict[str, Any]] = []

    def write_inventory() -> None:
        write_json(
            output_root / "observations_multiview_inventory.json",
            {
                "observations_root": str(observations_root),
                "output_root": str(output_root),
                "episode_count": len(inventory),
                "complete_episode_count": len(videos_dirs),
                "incomplete_episode_count": len(incomplete),
                "sample_fps": args.sample_fps,
                "candidate_count": args.candidate_count,
                "candidate_min_gap": args.candidate_min_gap,
                "episodes": inventory,
                "incomplete_episodes": incomplete,
            },
        )

    for index, videos_dir in enumerate(videos_dirs, start=1):
        episode_dir = videos_dir.parent
        episode_id = episode_dir.name
        task_id = task_id_for_video_dir(observations_root, videos_dir)
        output_dir = output_root / episode_id
        manifest_path = output_dir / "manifest.json"
        status = "skipped"
        if args.force or not manifest_path.exists():
            print(f"[{index}/{len(videos_dirs)}] building task={task_id} episode={episode_id}", flush=True)
            try:
                run_builder(
                    episode_dir=episode_dir,
                    output_dir=output_dir,
                    sample_fps=args.sample_fps,
                    candidate_count=args.candidate_count,
                    candidate_min_gap=args.candidate_min_gap,
                    force=args.force,
                )
                status = "built"
            except subprocess.CalledProcessError as error:
                status = "failed"
                print(f"[{index}/{len(videos_dirs)}] failed task={task_id} episode={episode_id}: {error}", flush=True)
        else:
            print(f"[{index}/{len(videos_dirs)}] skip existing task={task_id} episode={episode_id}", flush=True)

        inventory.append(
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "episode_dir": str(episode_dir),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "status": status,
            }
        )
        write_inventory()

    write_inventory()
    print(f"wrote {output_root / 'observations_multiview_inventory.json'}", flush=True)


if __name__ == "__main__":
    main()
