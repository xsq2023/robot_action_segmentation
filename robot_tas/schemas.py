from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    fps: float = Field(gt=0.0)
    total_frames: int = Field(ge=1)
    duration_seconds: float = Field(ge=0.0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    sample_fps: float = Field(gt=0.0)
    timestamp_source: str = "original_frame_id/native_fps"
    sampling_note: str = ""


class SampledFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(ge=0)
    original_frame_id: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    image_path: str
    image_sha256: str
    motion_score: float = Field(ge=0.0, le=1.0, default=0.0)
    mean_luma: float = Field(ge=0.0, le=1.0, default=0.0)


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: int = Field(ge=0)
    start_sample_index: int = Field(ge=0)
    end_sample_index: int = Field(ge=0)
    frames: list[SampledFrame]


class ActionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    actor_motion: str
    contact_state: str
    object_motion: str
    target_object: str


class BoundaryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary_sample_index: int = Field(ge=0)
    boundary_original_frame_id: int = Field(ge=0)
    boundary_timestamp: float = Field(ge=0.0)
    before_state: ActionState
    after_state: ActionState
    transition_type: str
    visual_evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class LocalBoundaryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: int = Field(ge=0)
    window_summary: str
    boundary_candidates: list[BoundaryCandidate]
    prompt_version: str


class VerifiedBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    window_id: int = Field(ge=0)
    status: Literal["accept", "shift_to_neighbor_frame", "reject"]
    original_boundary_sample_index: int = Field(ge=0)
    verified_boundary_sample_index: int | None = Field(default=None, ge=0)
    verified_original_frame_id: int | None = Field(default=None, ge=0)
    verified_timestamp: float | None = Field(default=None, ge=0.0)
    before_action: str
    after_action: str
    transition_type: str
    visual_evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        legacy = {
            "accepted": "accept",
            "corrected": "shift_to_neighbor_frame",
            "rejected": "reject",
        }
        return legacy.get(value, value)


class MergedBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary_sample_index: int = Field(ge=0)
    boundary_frame_id: int = Field(ge=0)
    boundary_time: float = Field(ge=0.0)
    before_action: str
    after_action: str
    transition_type: str
    visual_evidence: list[str]
    selected_views: list[str] = Field(default_factory=list)
    view_evidence: list[dict[str, str]] = Field(default_factory=list)
    trajectory_evidence: list[str] = Field(default_factory=list)
    supporting_windows: list[int]
    confidence: float = Field(ge=0.0, le=1.0)
    source_proposal_ids: list[str] = []


class RawSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: int = Field(ge=0)
    start_sample_index: int = Field(ge=0)
    end_sample_index: int = Field(ge=0)
    start_frame_id: int = Field(ge=0)
    end_frame_id: int = Field(ge=0)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)


class SegmentLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_label: str
    description: str
    primary_object: str
    secondary_objects: list[str]
    actor_motion: str
    contact_state: str
    object_motion: str
    goal: str
    confidence: float = Field(ge=0.0, le=1.0)


class LabeledSegment(RawSegment):
    action_label: str
    description: str
    primary_object: str
    secondary_objects: list[str]
    actor_motion: str
    contact_state: str
    object_motion: str
    goal: str
    selected_views: list[str] = Field(default_factory=list)
    left_hand_state: str = "unclear"
    right_hand_state: str = "unclear"
    held_objects_by_hand: dict[str, list[str]] = Field(default_factory=lambda: {"left": [], "right": []})
    visual_evidence: list[str] = Field(default_factory=list)
    trajectory_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GlobalIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["merge_adjacent_segments", "relabel_segment"]
    segment_ids: list[int] = []
    segment_id: int | None = Field(default=None, ge=0)
    new_label: str | None = None
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class GlobalCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[GlobalIssue]
    applied_issues: list[GlobalIssue] = []
    prompt_version: str
    summary: str = ""


class FinalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: VideoMetadata
    segments: list[LabeledSegment]
    boundaries: list[MergedBoundary]
    prompt_versions: dict[str, str]
