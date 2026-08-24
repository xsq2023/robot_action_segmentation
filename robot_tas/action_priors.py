from __future__ import annotations

from robot_tas.normalization import normalize_label, normalize_transition
from robot_tas.schemas import LabeledSegment, MergedBoundary, RawSegment, SampledFrame, VideoMetadata
from robot_tas.segmentation import construct_segments


def parse_preset_actions(raw_actions: str | None) -> list[str]:
    """Parse a comma-separated action prior into normalized labels."""

    if not raw_actions:
        return []
    return [normalize_label(action) for action in raw_actions.split(",") if normalize_label(action)]


def parse_boundary_ratios(raw_ratios: str | None) -> list[float]:
    """Parse cumulative boundary ratios such as 0.25,0.50,0.75."""

    if not raw_ratios:
        return []
    ratios = [float(ratio.strip()) for ratio in raw_ratios.split(",") if ratio.strip()]
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("Preset boundary ratios must be within (0, 1).")
    if ratios != sorted(ratios):
        raise ValueError("Preset boundary ratios must be strictly increasing.")
    if len(set(ratios)) != len(ratios):
        raise ValueError("Preset boundary ratios must not contain duplicates.")
    return ratios


def build_equal_prior_boundaries(
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    preset_actions: list[str],
    boundary_ratios: list[float] | None = None,
) -> list[MergedBoundary]:
    """Create prior boundaries snapped to real sampled frames."""

    if len(preset_actions) <= 1 or len(sampled_frames) <= 1:
        return []

    boundaries: list[MergedBoundary] = []
    used_sample_indices: set[int] = set()
    last_sample_position = len(sampled_frames) - 1
    if boundary_ratios:
        if len(boundary_ratios) != len(preset_actions) - 1:
            raise ValueError("Preset boundary ratios must have len(preset_actions) - 1 values.")
        cumulative_ratios = boundary_ratios
        source = "preset ratio"
    else:
        cumulative_ratios = [
            boundary_number / len(preset_actions)
            for boundary_number in range(1, len(preset_actions))
        ]
        source = "equal temporal spacing"

    for boundary_number, ratio in enumerate(cumulative_ratios, start=1):
        position = round(ratio * last_sample_position)
        position = min(max(1, position), last_sample_position)
        frame = sampled_frames[position]
        if frame.sample_index in used_sample_indices:
            continue
        used_sample_indices.add(frame.sample_index)
        before_action = preset_actions[boundary_number - 1]
        after_action = preset_actions[boundary_number]
        boundaries.append(
            MergedBoundary(
                boundary_sample_index=frame.sample_index,
                boundary_frame_id=frame.original_frame_id,
                boundary_time=frame.timestamp_seconds,
                before_action=before_action,
                after_action=after_action,
                transition_type=f"{before_action}_to_{after_action}",
                visual_evidence=[
                    (
                        f"Boundary generated from a task action prior with {source}; "
                        "this is not a visual boundary detection result."
                    )
                ],
                supporting_windows=[],
                confidence=0.5,
                source_proposal_ids=[f"preset_equal_boundary_{boundary_number}"],
            )
        )

    return boundaries


def _target_frame_id(metadata: VideoMetadata, boundary_number: int, action_count: int, boundary_ratios: list[float] | None) -> int:
    if boundary_ratios:
        ratio = boundary_ratios[boundary_number - 1]
    else:
        ratio = boundary_number / action_count
    return round(ratio * (metadata.total_frames - 1))


def _matches_transition(boundary: MergedBoundary, before_action: str, after_action: str) -> bool:
    expected_transition = normalize_transition(before_action, after_action)
    return (
        normalize_label(boundary.before_action) == before_action
        and normalize_label(boundary.after_action) == after_action
    ) or normalize_transition(boundary.before_action, boundary.after_action, boundary.transition_type) == expected_transition


def align_boundaries_with_prior(
    boundaries: list[MergedBoundary],
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    preset_actions: list[str],
    boundary_ratios: list[float] | None = None,
) -> list[MergedBoundary]:
    """Keep or fill one boundary per expected prior transition, in chronological order."""

    if len(preset_actions) <= 1:
        return []
    if boundary_ratios and len(boundary_ratios) != len(preset_actions) - 1:
        raise ValueError("Preset boundary ratios must have len(preset_actions) - 1 values.")

    ordered = sorted(boundaries, key=lambda boundary: boundary.boundary_sample_index)
    prior_boundaries = build_equal_prior_boundaries(
        metadata=metadata,
        sampled_frames=sampled_frames,
        preset_actions=preset_actions,
        boundary_ratios=boundary_ratios,
    )
    max_distance_frames = max(1, round(metadata.total_frames / (len(preset_actions) * 1.25)))
    selected: list[MergedBoundary] = []
    last_sample_index = -1
    for boundary_number, (before_action, after_action) in enumerate(zip(preset_actions, preset_actions[1:]), start=1):
        prior_boundary = prior_boundaries[boundary_number - 1]
        matching = [
            boundary
            for boundary in ordered
            if boundary.boundary_sample_index > last_sample_index
            and _matches_transition(boundary, before_action, after_action)
        ]
        target = _target_frame_id(
            metadata=metadata,
            boundary_number=boundary_number,
            action_count=len(preset_actions),
            boundary_ratios=boundary_ratios,
        )
        chosen = None
        if matching:
            candidate = min(
                matching,
                key=lambda boundary: (
                    abs(boundary.boundary_frame_id - target),
                    -boundary.confidence,
                    boundary.boundary_sample_index,
                ),
            )
            if abs(candidate.boundary_frame_id - target) <= max_distance_frames:
                chosen = candidate
        if chosen is None:
            chosen = prior_boundary
        selected.append(
            chosen.model_copy(
                update={
                    "visual_evidence": [
                        *chosen.visual_evidence,
                        (
                            "Boundary kept by fixed action-sequence alignment "
                            f"for expected transition {before_action}_to_{after_action}."
                        ),
                    ]
                }
            )
        )
        last_sample_index = chosen.boundary_sample_index

    return selected


def construct_prior_segments(
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    preset_actions: list[str],
    boundary_ratios: list[float] | None = None,
) -> tuple[list[MergedBoundary], list[RawSegment]]:
    """Construct segments directly from preset action priors."""

    boundaries = build_equal_prior_boundaries(
        metadata=metadata,
        sampled_frames=sampled_frames,
        preset_actions=preset_actions,
        boundary_ratios=boundary_ratios,
    )
    return boundaries, construct_segments(metadata=metadata, sampled_frames=sampled_frames, boundaries=boundaries)


def label_segments_with_prior(
    raw_segments: list[RawSegment],
    preset_actions: list[str],
) -> list[LabeledSegment]:
    """Assign preset action labels to already-grounded segments."""

    labeled_segments: list[LabeledSegment] = []
    for segment, action_label in zip(raw_segments, preset_actions):
        if action_label == "pick":
            description = "The robot retrieves an item from the shelf."
            actor_motion = "pick"
            contact_state = "grasp_or_lift"
            object_motion = "item_removed_from_shelf"
            goal = "retrieve an item"
        elif action_label == "place":
            description = "The robot places the held item into the shopping cart bag."
            actor_motion = "place"
            contact_state = "release_or_set_down"
            object_motion = "item_placed_into_container"
            goal = "place an item into the bag"
        else:
            description = f"The robot performs {action_label.replace('_', ' ')}."
            actor_motion = action_label
            contact_state = "unknown"
            object_motion = "unknown"
            goal = action_label.replace("_", " ")

        raw_fields = {field_name: getattr(segment, field_name) for field_name in RawSegment.model_fields}
        labeled_segments.append(
            LabeledSegment(
                **raw_fields,
                action_label=action_label,
                description=description,
                primary_object="supermarket item",
                secondary_objects=["shelf", "shopping cart bag"],
                actor_motion=actor_motion,
                contact_state=contact_state,
                object_motion=object_motion,
                goal=goal,
                confidence=0.5,
            )
        )
    return labeled_segments
