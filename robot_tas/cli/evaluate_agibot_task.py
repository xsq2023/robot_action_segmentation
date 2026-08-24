from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from robot_tas.evaluation import EvalSegment, mean_absolute_boundary_error, segmental_f1
from robot_tas.normalization import normalize_label


@dataclass(slots=True)
class EpisodeEval:
    episode_id: int
    video_path: str
    predicted_output_dir: str
    gt_action_count: int
    gt_boundary_count: int
    predicted_segment_count: int
    predicted_boundary_count: int
    boundary_tp: int
    boundary_fp: int
    boundary_fn: int
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_mae_frames: float | None
    coarse_framewise_accuracy: float
    coarse_segment_f1_10: float
    coarse_segment_f1_25: float
    coarse_segment_f1_50: float
    predicted_labels: list[str]
    predicted_coarse_labels: list[str]
    gt_skills: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Robot TAS on AgiBotWorld observations and compare against task_info ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations-dir", required=True, help="Observation task directory, e.g. dataset/.../observations/327")
    parser.add_argument("--task-info", required=True, help="Task info JSON path, e.g. dataset/.../task_info/task_327.json")
    parser.add_argument("--camera", default="head_color.mp4", help="Video filename to use under each episode's videos/ directory.")
    parser.add_argument("--model", default="codex-local", help="Pipeline model name to pass to run_tas.py.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Sampling FPS for run_tas.py.")
    parser.add_argument("--window-size", type=int, default=16, help="Sliding window size for run_tas.py.")
    parser.add_argument("--window-stride", type=int, default=8, help="Sliding window stride for run_tas.py.")
    parser.add_argument("--pipeline-boundary-tolerance", type=int, default=None, help="Boundary clustering tolerance passed to run_tas.py.")
    parser.add_argument("--min-segment-samples", type=int, default=None, help="Minimum visual segment length passed to run_tas.py.")
    parser.add_argument("--min-boundary-confidence", type=float, default=None, help="Minimum boundary confidence passed to run_tas.py.")
    parser.add_argument(
        "--preset-actions",
        default=None,
        help="Comma-separated action prior passed to run_tas.py.",
    )
    parser.add_argument(
        "--preset-boundary-mode",
        choices=["off", "fill-missing", "force"],
        default="off",
        help="Preset boundary mode passed to run_tas.py.",
    )
    parser.add_argument(
        "--preset-boundary-ratios",
        default=None,
        help="Comma-separated cumulative boundary ratios passed to run_tas.py.",
    )
    parser.add_argument(
        "--learn-boundary-ratios-from-task-info",
        action="store_true",
        help="Use average action boundary ratios from task_info entries not present under observations-dir.",
    )
    parser.add_argument("--boundary-tolerance", type=int, default=30, help="Boundary tolerance in frames for evaluation.")
    parser.add_argument("--output-dir", default="outputs/agibot_task_eval", help="Directory under api/ for run outputs and summaries.")
    parser.add_argument("--force", action="store_true", help="Force rerunning per-episode pipeline outputs.")
    return parser.parse_args()


def load_task_info(task_info_path: Path) -> dict[int, dict]:
    data = json.loads(task_info_path.read_text(encoding="utf-8"))
    return {int(item["episode_id"]): item for item in data}


def find_episode_dirs(observations_dir: Path) -> list[Path]:
    episode_dirs: list[Path] = []
    for child in observations_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "videos").is_dir():
            episode_dirs.append(child)
            continue
        episode_dirs.extend(
            episode_dir
            for episode_dir in child.iterdir()
            if episode_dir.is_dir() and (episode_dir / "videos").is_dir()
        )
    return sorted(episode_dirs)


def coarse_label_from_prediction(action_label: str) -> str:
    return normalize_label(action_label)


def gt_segments_and_boundaries(gt_item: dict) -> tuple[list[EvalSegment], list[int]]:
    action_config = gt_item["label_info"]["action_config"]
    gt_segments = [
        EvalSegment(
            start_frame_id=int(action["start_frame"]),
            end_frame_id=int(action["end_frame"]),
            label=str(action["skill"]),
        )
        for action in action_config
    ]
    gt_boundaries = [int(action["start_frame"]) for action in action_config[1:]]
    return gt_segments, gt_boundaries


def predicted_segments_and_boundaries(final_segments: dict) -> tuple[list[EvalSegment], list[int], list[str], list[str]]:
    segments = final_segments["segments"]
    predicted_segments = [
        EvalSegment(
            start_frame_id=int(segment["start_frame_id"]),
            end_frame_id=int(segment["end_frame_id"]),
            label=coarse_label_from_prediction(segment["action_label"]),
        )
        for segment in segments
    ]
    predicted_boundaries = [int(boundary["boundary_frame_id"]) for boundary in final_segments["boundaries"]]
    predicted_labels = [str(segment["action_label"]) for segment in segments]
    predicted_coarse_labels = [segment.label for segment in predicted_segments]
    return predicted_segments, predicted_boundaries, predicted_labels, predicted_coarse_labels


def boundary_counts(predicted_boundaries: Iterable[int], gt_boundaries: Iterable[int], tolerance: int) -> tuple[int, int, int]:
    predicted = list(predicted_boundaries)
    truth = list(gt_boundaries)
    matched_truth: set[int] = set()
    tp = 0
    for boundary in predicted:
        match_index = None
        best_delta = None
        for truth_index, truth_boundary in enumerate(truth):
            if truth_index in matched_truth:
                continue
            delta = abs(boundary - truth_boundary)
            if delta <= tolerance and (best_delta is None or delta < best_delta):
                match_index = truth_index
                best_delta = delta
        if match_index is not None:
            matched_truth.add(match_index)
            tp += 1
    fp = len(predicted) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def framewise_accuracy_labeled_window(predicted_segments: list[EvalSegment], gt_segments: list[EvalSegment]) -> float:
    if not gt_segments:
        return 0.0
    start_frame = min(segment.start_frame_id for segment in gt_segments)
    end_frame = max(segment.end_frame_id for segment in gt_segments)
    if end_frame <= start_frame:
        return 0.0

    total = 0
    matches = 0
    for frame_id in range(start_frame, end_frame):
        gt_label = "Other"
        for segment in gt_segments:
            if segment.start_frame_id <= frame_id < segment.end_frame_id:
                gt_label = normalize_label(segment.label)
                break

        predicted_label = "Other"
        for segment in predicted_segments:
            if segment.start_frame_id <= frame_id < segment.end_frame_id:
                predicted_label = normalize_label(segment.label)
                break

        total += 1
        if gt_label == predicted_label:
            matches += 1
    return matches / total if total else 0.0


def run_episode_pipeline(
    api_dir: Path,
    video_path: Path,
    output_dir: Path,
    model: str,
    sample_fps: float,
    window_size: int,
    window_stride: int,
    pipeline_boundary_tolerance: int | None,
    min_segment_samples: int | None,
    min_boundary_confidence: float | None,
    preset_actions: str | None,
    preset_boundary_mode: str,
    preset_boundary_ratios: str | None,
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
        "--preset-boundary-mode",
        preset_boundary_mode,
    ]
    if pipeline_boundary_tolerance is not None:
        command.extend(["--boundary-tolerance", str(pipeline_boundary_tolerance)])
    if min_segment_samples is not None:
        command.extend(["--min-segment-samples", str(min_segment_samples)])
    if min_boundary_confidence is not None:
        command.extend(["--min-boundary-confidence", str(min_boundary_confidence)])
    if preset_actions:
        command.extend(["--preset-actions", preset_actions])
    if preset_boundary_ratios:
        command.extend(["--preset-boundary-ratios", preset_boundary_ratios])
    if force:
        command.append("--force")
    result = subprocess.run(command, cwd=api_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed for {video_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def evaluate_episode(
    episode_id: int,
    video_path: Path,
    predicted_output_dir: Path,
    gt_item: dict,
    tolerance: int,
) -> EpisodeEval:
    final_segments = json.loads((predicted_output_dir / "final_segments.json").read_text(encoding="utf-8"))
    gt_segments, gt_boundaries = gt_segments_and_boundaries(gt_item)
    predicted_segments, predicted_boundaries, predicted_labels, predicted_coarse_labels = predicted_segments_and_boundaries(final_segments)

    tp, fp, fn = boundary_counts(predicted_boundaries, gt_boundaries, tolerance=tolerance)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)

    coarse_segment_scores = segmental_f1(predicted_segments, gt_segments)
    return EpisodeEval(
        episode_id=episode_id,
        video_path=str(video_path),
        predicted_output_dir=str(predicted_output_dir),
        gt_action_count=len(gt_segments),
        gt_boundary_count=len(gt_boundaries),
        predicted_segment_count=len(predicted_segments),
        predicted_boundary_count=len(predicted_boundaries),
        boundary_tp=tp,
        boundary_fp=fp,
        boundary_fn=fn,
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        boundary_mae_frames=mean_absolute_boundary_error(predicted_boundaries, gt_boundaries),
        coarse_framewise_accuracy=framewise_accuracy_labeled_window(predicted_segments, gt_segments),
        coarse_segment_f1_10=coarse_segment_scores["f1@0.10"],
        coarse_segment_f1_25=coarse_segment_scores["f1@0.25"],
        coarse_segment_f1_50=coarse_segment_scores["f1@0.50"],
        predicted_labels=predicted_labels,
        predicted_coarse_labels=predicted_coarse_labels,
        gt_skills=[segment.label for segment in gt_segments],
    )


def learned_boundary_ratios(task_lookup: dict[int, dict], excluded_episode_ids: set[int]) -> list[float]:
    """Learn average cumulative action ratios from non-evaluated task_info entries."""

    ratio_rows: list[list[float]] = []
    for episode_id, item in task_lookup.items():
        if episode_id in excluded_episode_ids:
            continue
        action_config = item["label_info"]["action_config"]
        if len(action_config) <= 1:
            continue
        final_end = float(action_config[-1]["end_frame"])
        if final_end <= 0.0:
            continue
        ratio_rows.append([float(action["start_frame"]) / final_end for action in action_config[1:]])

    if not ratio_rows:
        return []
    expected_len = len(ratio_rows[0])
    if any(len(row) != expected_len for row in ratio_rows):
        raise ValueError("Cannot learn one boundary-ratio prior from task_info with variable action counts.")
    return [statistics.mean(row[index] for row in ratio_rows) for index in range(expected_len)]


def main() -> None:
    args = parse_args()
    api_dir = Path(__file__).resolve().parents[2]
    repo_root = api_dir.parent
    observations_dir = (repo_root / args.observations_dir).resolve()
    task_info_path = (repo_root / args.task_info).resolve()
    base_output_dir = (repo_root / "api" / args.output_dir / Path(args.camera).stem).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    task_lookup = load_task_info(task_info_path)
    episode_dirs = find_episode_dirs(observations_dir)
    observed_episode_ids = {int(episode_dir.name) for episode_dir in episode_dirs}
    preset_boundary_ratios = args.preset_boundary_ratios
    if args.learn_boundary_ratios_from_task_info:
        learned_ratios = learned_boundary_ratios(task_lookup, excluded_episode_ids=observed_episode_ids)
        preset_boundary_ratios = ",".join(f"{ratio:.6f}" for ratio in learned_ratios)
    per_episode: list[EpisodeEval] = []

    for episode_dir in episode_dirs:
        episode_id = int(episode_dir.name)
        if episode_id not in task_lookup:
            continue
        video_path = episode_dir / "videos" / args.camera
        if not video_path.exists():
            continue
        predicted_output_dir = base_output_dir / str(episode_id)
        print(f"[run] episode={episode_id} camera={args.camera}")
        run_episode_pipeline(
            api_dir=api_dir,
            video_path=video_path,
            output_dir=predicted_output_dir,
            model=args.model,
            sample_fps=args.sample_fps,
            window_size=args.window_size,
            window_stride=args.window_stride,
            pipeline_boundary_tolerance=args.pipeline_boundary_tolerance,
            min_segment_samples=args.min_segment_samples,
            min_boundary_confidence=args.min_boundary_confidence,
            preset_actions=args.preset_actions,
            preset_boundary_mode=args.preset_boundary_mode,
            preset_boundary_ratios=preset_boundary_ratios,
            force=args.force,
        )
        per_episode.append(
            evaluate_episode(
                episode_id=episode_id,
                video_path=video_path,
                predicted_output_dir=predicted_output_dir,
                gt_item=task_lookup[episode_id],
                tolerance=args.boundary_tolerance,
            )
        )

    total_tp = total_fp = total_fn = 0
    maes = [item.boundary_mae_frames for item in per_episode if item.boundary_mae_frames is not None]
    for item in per_episode:
        total_tp += item.boundary_tp
        total_fp += item.boundary_fp
        total_fn += item.boundary_fn

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        0.0
        if micro_precision + micro_recall == 0.0
        else (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
    )

    summary = {
        "observations_dir": str(observations_dir),
        "task_info": str(task_info_path),
        "camera": args.camera,
        "provider": "codex",
        "model": args.model,
        "sample_fps": args.sample_fps,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "pipeline_boundary_tolerance": args.pipeline_boundary_tolerance,
        "min_segment_samples": args.min_segment_samples,
        "min_boundary_confidence": args.min_boundary_confidence,
        "preset_actions": args.preset_actions,
        "preset_boundary_mode": args.preset_boundary_mode,
        "preset_boundary_ratios": preset_boundary_ratios,
        "learn_boundary_ratios_from_task_info": args.learn_boundary_ratios_from_task_info,
        "boundary_tolerance_frames": args.boundary_tolerance,
        "episodes_evaluated": len(per_episode),
        "micro_boundary_precision": micro_precision,
        "micro_boundary_recall": micro_recall,
        "micro_boundary_f1": micro_f1,
        "mean_boundary_mae_frames": statistics.mean(maes) if maes else None,
        "mean_coarse_framewise_accuracy": statistics.mean(item.coarse_framewise_accuracy for item in per_episode) if per_episode else 0.0,
        "mean_coarse_segment_f1_10": statistics.mean(item.coarse_segment_f1_10 for item in per_episode) if per_episode else 0.0,
        "mean_coarse_segment_f1_25": statistics.mean(item.coarse_segment_f1_25 for item in per_episode) if per_episode else 0.0,
        "mean_coarse_segment_f1_50": statistics.mean(item.coarse_segment_f1_50 for item in per_episode) if per_episode else 0.0,
        "mean_predicted_segment_count": statistics.mean(item.predicted_segment_count for item in per_episode) if per_episode else 0.0,
        "mean_predicted_boundary_count": statistics.mean(item.predicted_boundary_count for item in per_episode) if per_episode else 0.0,
    }

    summary_path = base_output_dir / "summary.json"
    per_episode_path = base_output_dir / "per_episode.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    per_episode_path.write_text(
        json.dumps([asdict(item) for item in per_episode], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
