from __future__ import annotations

from robot_tas.schemas import SampledFrame


def frame_reference_line(frame: SampledFrame) -> str:
    """Compact frame label plus low-level visual cues for VLM prompts."""

    return (
        f"FRAME sample_index={frame.sample_index} "
        f"original_frame_id={frame.original_frame_id} "
        f"time={frame.timestamp_seconds:.3f}s "
        f"motion_score={frame.motion_score:.4f} "
        f"mean_luma={frame.mean_luma:.4f}"
    )


def transition_change_score(previous: SampledFrame, current: SampledFrame) -> float:
    """Score adjacent-frame change using cheap cues as a non-binding visual reference."""

    luma_delta = abs(current.mean_luma - previous.mean_luma)
    return min(1.0, 0.75 * current.motion_score + 0.25 * luma_delta)


def transition_reference_lines(frames: list[SampledFrame], top_k: int = 6) -> list[str]:
    """Return ranked adjacent-frame change hints for prompt context."""

    if len(frames) < 2 or top_k <= 0:
        return []

    references: list[tuple[float, SampledFrame, SampledFrame]] = []
    for previous, current in zip(frames, frames[1:]):
        references.append((transition_change_score(previous, current), previous, current))

    ranked = sorted(references, key=lambda item: (-item[0], item[2].sample_index))[:top_k]
    lines: list[str] = []
    for score, previous, current in ranked:
        luma_delta = abs(current.mean_luma - previous.mean_luma)
        lines.append(
            "candidate_boundary "
            f"sample_index={current.sample_index} "
            f"original_frame_id={current.original_frame_id} "
            f"time={current.timestamp_seconds:.3f}s "
            f"previous_sample_index={previous.sample_index} "
            f"visual_change_score={score:.4f} "
            f"motion_score={current.motion_score:.4f} "
            f"luma_delta={luma_delta:.4f}"
        )
    return lines
