from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from robot_tas.aggregation import merge_verified_boundaries
from robot_tas.action_set import format_action_set_contract, merge_action_sets, parse_action_set
from robot_tas.action_priors import (
    align_boundaries_with_prior,
    construct_prior_segments,
    label_segments_with_prior,
    parse_boundary_ratios,
    parse_preset_actions,
)
from robot_tas.api.base import MultimodalClient
from robot_tas.api.codex_client import CodexMultimodalClient
from robot_tas.cache import ensure_dir, read_json, write_json
from robot_tas.config import PipelineConfig, apply_overrides, load_config
from robot_tas.global_check import run_global_consistency_check
from robot_tas.logging_utils import configure_logging
from robot_tas.proposal import run_local_boundary_proposals
from robot_tas.sampler import sample_video_frames
from robot_tas.schemas import (
    FinalOutput,
    GlobalCheckResult,
    LabeledSegment,
    LocalBoundaryProposal,
    MergedBoundary,
    RawSegment,
    SampledFrame,
    VideoMetadata,
    Window,
)
from robot_tas.segmentation import construct_segments
from robot_tas.semantic_labeling import run_segment_labeling
from robot_tas.verification import run_boundary_verification
from robot_tas.video_io import read_video_metadata
from robot_tas.visualization import write_annotated_action_video, write_timeline_html
from robot_tas.windows import build_sliding_windows


LOGGER = logging.getLogger(__name__)


def api_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_api_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (api_root() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Robot TAS MVP pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input robot ego-view video.")
    parser.add_argument("--output-dir", required=True, help="Directory for all pipeline outputs.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--provider", choices=["codex"], default=None, help="Only supported visual-judgment provider.")
    parser.add_argument("--model", default=None, help="Model identifier for the selected provider.")
    parser.add_argument("--sample-fps", type=float, default=None, help="Sampling FPS.")
    parser.add_argument("--window-size", type=int, default=None, help="Sliding window size in sampled frames.")
    parser.add_argument("--window-stride", type=int, default=None, help="Sliding window stride in sampled frames.")
    parser.add_argument("--boundary-tolerance", type=int, default=None, help="Boundary clustering tolerance in sampled frames.")
    parser.add_argument("--min-segment-samples", type=int, default=None, help="Minimum visual segment length in sampled frames.")
    parser.add_argument("--min-boundary-confidence", type=float, default=None, help="Minimum merged boundary confidence.")
    parser.add_argument(
        "--preset-actions",
        default=None,
        help="Comma-separated action prior, e.g. pick,place,pick,place,pick,place.",
    )
    parser.add_argument(
        "--action-set",
        default=None,
        help="Comma-separated allowed action labels. Defaults to a broad robot manipulation action set.",
    )
    parser.add_argument(
        "--preset-boundary-mode",
        choices=["off", "fill-missing", "force", "align"],
        default="off",
        help="Use preset actions to align, fill, or force grounded segments.",
    )
    parser.add_argument(
        "--preset-boundary-ratios",
        default=None,
        help="Comma-separated cumulative boundary ratios, e.g. 0.24,0.35,0.56,0.67,0.87.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute artifacts instead of reusing caches.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_prompt(prompt_path: Path) -> tuple[str, str]:
    return prompt_path.stem, prompt_path.read_text(encoding="utf-8")


def append_action_contract(prompt_text: str, preset_actions: list[str]) -> str:
    if not preset_actions:
        return prompt_text
    sequence = " -> ".join(preset_actions)
    allowed = ", ".join(sorted(set(preset_actions)))
    return (
        f"{prompt_text}\n\n"
        "Task action contract:\n"
        f"- Expected coarse action sequence for the whole episode: {sequence}.\n"
        f"- The only allowed final action labels are: {allowed}.\n"
        "- Candidate boundaries must be between adjacent actions in this sequence.\n"
        "- If the visual evidence is ambiguous, prefer fewer coarse boundaries over fine-grained over-segmentation.\n"
        "- Do not use object names as action labels; keep object names in evidence/description fields.\n"
    )


def append_action_set_contract(prompt_text: str, action_set: list[str]) -> str:
    return f"{prompt_text}\n\n{format_action_set_contract(action_set)}\n"


def build_client(config: PipelineConfig, output_dir: Path) -> MultimodalClient:
    provider = config.provider.lower()
    if provider == "codex":
        return CodexMultimodalClient(
            model=config.model,
            temperature=config.temperature,
            artifact_root=output_dir,
        )
    raise ValueError(f"Unsupported provider: {config.provider}")


def prepare_metadata_and_samples(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    force: bool,
) -> tuple[VideoMetadata, list[SampledFrame]]:
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not force:
        cached = read_json(metadata_path)
        metadata = VideoMetadata.model_validate(cached["video"])
        sampled_frames = [SampledFrame.model_validate(item) for item in cached["sampled_frames"]]
        return metadata, sampled_frames

    metadata = read_video_metadata(video_path=video_path, sample_fps=sample_fps)
    sampled_frames = sample_video_frames(video_path=video_path, metadata=metadata, output_dir=output_dir)
    write_json(
        metadata_path,
        {
            "video": metadata.model_dump(mode="json"),
            "sampled_frames": [frame.model_dump(mode="json") for frame in sampled_frames],
        },
    )
    return metadata, sampled_frames


def prepare_windows(
    sampled_frames: list[SampledFrame],
    output_dir: Path,
    window_size: int,
    window_stride: int,
    force: bool,
) -> list[Window]:
    stage_path = output_dir / "windows.json"
    if stage_path.exists() and not force:
        return [Window.model_validate(item) for item in read_json(stage_path)]

    windows = build_sliding_windows(
        sampled_frames=sampled_frames,
        window_size=window_size,
        stride=window_stride,
    )
    write_json(stage_path, [window.model_dump(mode="json") for window in windows])
    return windows


def write_stage(path: Path, items: list[LocalBoundaryProposal] | list[MergedBoundary] | list[RawSegment] | list[LabeledSegment]) -> None:
    write_json(path, [item.model_dump(mode="json") for item in items])


def load_stage(path: Path, model_type: type[LocalBoundaryProposal] | type[MergedBoundary] | type[RawSegment] | type[LabeledSegment]) -> list[LocalBoundaryProposal] | list[MergedBoundary] | list[RawSegment] | list[LabeledSegment]:
    return [model_type.model_validate(item) for item in read_json(path)]


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    video_path = Path(args.video).resolve()
    config_path = resolve_api_path(args.config)

    config = load_config(config_path)
    config = apply_overrides(
        config,
        provider=args.provider,
        model=args.model,
        sample_fps=args.sample_fps,
        window_size=args.window_size,
        window_stride=args.window_stride,
        boundary_tolerance=args.boundary_tolerance,
        min_segment_samples=args.min_segment_samples,
        min_boundary_confidence=args.min_boundary_confidence,
        force=args.force,
    )

    prompts_dir = api_root() / "prompts"
    prompt_versions = {}
    local_version, local_prompt = load_prompt(prompts_dir / "local_boundary_v1.txt")
    verify_version, verify_prompt = load_prompt(prompts_dir / "verify_boundary_v1.txt")
    label_version, label_prompt = load_prompt(prompts_dir / "label_segment_v1.txt")
    global_version, global_prompt = load_prompt(prompts_dir / "global_check_v1.txt")
    prompt_versions["local_boundary"] = local_version
    prompt_versions["verify_boundary"] = verify_version
    prompt_versions["label_segment"] = label_version
    prompt_versions["global_check"] = global_version
    preset_actions = parse_preset_actions(args.preset_actions)
    action_set = merge_action_sets(parse_action_set(args.action_set), preset_actions)
    preset_boundary_ratios = parse_boundary_ratios(args.preset_boundary_ratios)
    local_prompt = append_action_set_contract(local_prompt, action_set)
    verify_prompt = append_action_set_contract(verify_prompt, action_set)
    label_prompt = append_action_set_contract(label_prompt, action_set)
    global_prompt = append_action_set_contract(global_prompt, action_set)
    local_prompt = append_action_contract(local_prompt, preset_actions)
    verify_prompt = append_action_contract(verify_prompt, preset_actions)
    label_prompt = append_action_contract(label_prompt, preset_actions)
    global_prompt = append_action_contract(global_prompt, preset_actions)
    prompt_versions["action_set"] = ",".join(action_set)
    if preset_actions:
        prompt_versions["action_prior"] = ",".join(preset_actions)
    if preset_boundary_ratios:
        prompt_versions["action_prior_boundary_ratios"] = ",".join(
            f"{ratio:.6f}" for ratio in preset_boundary_ratios
        )

    metadata, sampled_frames = prepare_metadata_and_samples(
        video_path=video_path,
        output_dir=output_dir,
        sample_fps=config.sample_fps,
        force=config.force,
    )
    LOGGER.info("Prepared %s sampled frames", len(sampled_frames))

    windows = prepare_windows(
        sampled_frames=sampled_frames,
        output_dir=output_dir,
        window_size=config.window_size,
        window_stride=config.window_stride,
        force=config.force,
    )
    LOGGER.info("Prepared %s sliding windows", len(windows))

    client = build_client(config=config, output_dir=output_dir)

    local_proposals = run_local_boundary_proposals(
        windows=windows,
        client=client,
        prompt_text=local_prompt,
        prompt_version=local_version,
        output_dir=output_dir,
        force=config.force,
    )

    verified_boundaries = run_boundary_verification(
        sampled_frames=sampled_frames,
        proposals=local_proposals,
        client=client,
        prompt_text=verify_prompt,
        prompt_version=verify_version,
        output_dir=output_dir,
        verification_radius=config.verification_radius,
        force=config.force,
    )

    merged_boundaries_path = output_dir / "merged_boundaries.json"
    if merged_boundaries_path.exists() and not config.force:
        merged_boundaries = load_stage(merged_boundaries_path, MergedBoundary)
    else:
        merged_boundaries = merge_verified_boundaries(
            verified_boundaries=verified_boundaries,
            tolerance=config.boundary_tolerance,
            min_boundary_confidence=config.min_boundary_confidence,
            min_segment_samples=config.min_segment_samples,
            total_sample_count=len(sampled_frames),
        )
        write_stage(merged_boundaries_path, merged_boundaries)
    LOGGER.info("Merged boundaries down to %s final candidates", len(merged_boundaries))

    if preset_actions and args.preset_boundary_mode == "align":
        aligned_boundaries = align_boundaries_with_prior(
            boundaries=merged_boundaries,
            metadata=metadata,
            sampled_frames=sampled_frames,
            preset_actions=preset_actions,
            boundary_ratios=preset_boundary_ratios,
        )
        if aligned_boundaries:
            merged_boundaries = aligned_boundaries
            write_stage(merged_boundaries_path, merged_boundaries)
            LOGGER.info(
                "Aligned boundaries to action prior: %s/%s transitions kept",
                len(merged_boundaries),
                len(preset_actions) - 1,
            )

    use_preset_segments = False
    if preset_actions:
        expected_boundary_count = max(0, len(preset_actions) - 1)
        if args.preset_boundary_mode == "force":
            use_preset_segments = True
        elif args.preset_boundary_mode == "fill-missing" and len(merged_boundaries) < expected_boundary_count:
            use_preset_segments = True

    if use_preset_segments:
        merged_boundaries, raw_segments = construct_prior_segments(
            metadata=metadata,
            sampled_frames=sampled_frames,
            preset_actions=preset_actions,
            boundary_ratios=preset_boundary_ratios,
        )
        write_stage(merged_boundaries_path, merged_boundaries)
        write_stage(output_dir / "segments_raw.json", raw_segments)
        labeled_segments = label_segments_with_prior(raw_segments=raw_segments, preset_actions=preset_actions)
        write_stage(output_dir / "segments_labeled.json", labeled_segments)
        LOGGER.info(
            "Applied preset action prior: %s segments, %s boundaries",
            len(labeled_segments),
            len(merged_boundaries),
        )
    else:
        raw_segments_path = output_dir / "segments_raw.json"
        if raw_segments_path.exists() and not config.force:
            raw_segments = load_stage(raw_segments_path, RawSegment)
        else:
            raw_segments = construct_segments(
                metadata=metadata,
                sampled_frames=sampled_frames,
                boundaries=merged_boundaries,
            )
            write_stage(raw_segments_path, raw_segments)

        if preset_actions and len(raw_segments) == len(preset_actions):
            labeled_segments = label_segments_with_prior(raw_segments=raw_segments, preset_actions=preset_actions)
            write_stage(output_dir / "segments_labeled.json", labeled_segments)
            LOGGER.info("Applied preset action labels to %s visual segments", len(labeled_segments))
        else:
            labeled_segments = run_segment_labeling(
                segments=raw_segments,
                sampled_frames=sampled_frames,
                client=client,
                prompt_text=label_prompt,
                prompt_version=label_version,
                output_dir=output_dir,
                force=config.force,
            )

    if preset_actions and len(labeled_segments) == len(preset_actions):
        global_result = GlobalCheckResult(
            issues=[],
            applied_issues=[],
            prompt_version=global_version,
            summary="Skipped because a fixed action prior already defines the coarse sequence.",
        )
        write_json(output_dir / "global_check.json", global_result.model_dump(mode="json"))
        final_segments = labeled_segments
        final_boundaries = merged_boundaries
    else:
        global_result, final_segments, final_boundaries = run_global_consistency_check(
            metadata=metadata,
            sampled_frames=sampled_frames,
            boundaries=merged_boundaries,
            segments=labeled_segments,
            client=client,
            prompt_text=global_prompt,
            prompt_version=global_version,
            output_dir=output_dir,
            confidence_threshold=config.global_edit_confidence,
            force=config.force,
        )

    final_output = FinalOutput(
        video=metadata,
        segments=final_segments,
        boundaries=final_boundaries,
        prompt_versions=prompt_versions,
    )
    write_json(output_dir / "final_segments.json", final_output.model_dump(mode="json"))
    if os.environ.get("ROBOT_TAS_SKIP_RUN_TAS_VISUALIZATION") != "1":
        write_timeline_html(
            metadata=metadata,
            sampled_frames=sampled_frames,
            boundaries=final_boundaries,
            segments=final_segments,
            output_path=output_dir / "timeline.html",
        )
        write_annotated_action_video(
            metadata=metadata,
            segments=final_segments,
            output_path=output_dir / "annotated_actions.mp4",
        )

    LOGGER.info(
        "Pipeline complete: %s segments, %s boundaries, %s applied global issues",
        len(final_segments),
        len(final_boundaries),
        len(global_result.applied_issues),
    )


if __name__ == "__main__":
    main()
