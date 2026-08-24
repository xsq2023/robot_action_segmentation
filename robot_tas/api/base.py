from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from robot_tas.schemas import (
    BoundaryCandidate,
    GlobalCheckResult,
    LabeledSegment,
    LocalBoundaryProposal,
    MergedBoundary,
    RawSegment,
    SampledFrame,
    SegmentLabel,
    VerifiedBoundary,
    VideoMetadata,
    Window,
)


T = TypeVar("T")


@dataclass(slots=True)
class ClientCallResult(Generic[T]):
    """Structured result for a single client call."""

    parsed: T
    raw_request: dict[str, Any]
    raw_response: dict[str, Any]
    cache_key: str


class MultimodalClient(ABC):
    """Abstract interface for multimodal boundary and labeling calls."""

    def __init__(self, provider: str, model: str, temperature: float, artifact_root: Path) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.artifact_root = artifact_root

    @abstractmethod
    def propose_boundaries(
        self,
        window: Window,
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[LocalBoundaryProposal]:
        raise NotImplementedError

    @abstractmethod
    def verify_boundary(
        self,
        proposal_id: str,
        candidate: BoundaryCandidate,
        neighborhood: list[SampledFrame],
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[VerifiedBoundary]:
        raise NotImplementedError

    @abstractmethod
    def label_segment(
        self,
        segment: RawSegment,
        representative_frames: list[SampledFrame],
        total_segments: int,
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[SegmentLabel]:
        raise NotImplementedError

    @abstractmethod
    def check_global_consistency(
        self,
        metadata: VideoMetadata,
        segments: list[LabeledSegment],
        boundaries: list[MergedBoundary],
        boundary_frames: dict[int, list[SampledFrame]],
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[GlobalCheckResult]:
        raise NotImplementedError

