from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np

from robot_tas.action_set import DEFAULT_ACTION_SET, normalize_action_set
from robot_tas.api.base import ClientCallResult, MultimodalClient
from robot_tas.cache import deterministic_key
from robot_tas.normalization import normalize_label, normalize_transition
from robot_tas.schemas import (
    ActionState,
    BoundaryCandidate,
    GlobalCheckResult,
    GlobalIssue,
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


@dataclass(slots=True)
class FrameAnalysis:
    sample_index: int
    original_frame_id: int
    timestamp_seconds: float
    mean_rgb: tuple[float, float, float]
    center_rgb: tuple[float, float, float]
    mean_luma: float
    edge_density: float
    motion_score: float
    color_name: str


class CodexMultimodalClient(MultimodalClient):
    """Local Codex-only visual helper for deterministic pipeline stages."""

    def __init__(self, model: str, temperature: float, artifact_root: Path) -> None:
        super().__init__(provider="codex", model=model, temperature=temperature, artifact_root=artifact_root)
        self._analysis_cache: dict[str, FrameAnalysis] = {}

    def _resolve_image_path(self, frame: SampledFrame) -> Path:
        return self.artifact_root / frame.image_path

    def _read_frame(self, frame: SampledFrame) -> np.ndarray:
        image = cv2.imread(str(self._resolve_image_path(frame)))
        if image is None:
            raise FileNotFoundError(f"Unable to read sampled frame image: {frame.image_path}")
        return image

    def _color_name(self, bgr: np.ndarray) -> str:
        hsv = cv2.cvtColor(np.uint8([[bgr.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
        hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if val < 50:
            return "dark"
        if sat < 40:
            return "gray"
        if hue < 10 or hue >= 170:
            return "red"
        if hue < 25:
            return "orange"
        if hue < 40:
            return "yellow"
        if hue < 85:
            return "green"
        if hue < 130:
            return "blue"
        return "magenta"

    def _analyze_frame(self, frame: SampledFrame) -> FrameAnalysis:
        if frame.image_sha256 in self._analysis_cache:
            return self._analysis_cache[frame.image_sha256]

        image = self._read_frame(frame)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        y0, y1 = height // 4, (height * 3) // 4
        x0, x1 = width // 4, (width * 3) // 4
        center = image[y0:y1, x0:x1]
        center_rgb = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
        edges = cv2.Canny(gray, 75, 175)

        analysis = FrameAnalysis(
            sample_index=frame.sample_index,
            original_frame_id=frame.original_frame_id,
            timestamp_seconds=frame.timestamp_seconds,
            mean_rgb=tuple(float(value) for value in np.mean(rgb.reshape(-1, 3), axis=0)),
            center_rgb=tuple(float(value) for value in np.mean(center_rgb.reshape(-1, 3), axis=0)),
            mean_luma=float(np.mean(gray) / 255.0),
            edge_density=float(np.mean(edges > 0)),
            motion_score=frame.motion_score,
            color_name=self._color_name(np.mean(center.reshape(-1, 3), axis=0)),
        )
        self._analysis_cache[frame.image_sha256] = analysis
        return analysis

    def _transition_scores(self, frames: list[SampledFrame]) -> list[float]:
        analyses = [self._analyze_frame(frame) for frame in frames]
        scores: list[float] = []
        for previous, current in zip(analyses, analyses[1:]):
            color_delta = float(
                np.linalg.norm(np.array(previous.center_rgb) - np.array(current.center_rgb)) / 441.67295593
            )
            luma_delta = abs(previous.mean_luma - current.mean_luma)
            edge_delta = abs(previous.edge_density - current.edge_density)
            motion_bonus = max(previous.motion_score, current.motion_score)
            score = min(1.0, 0.35 * color_delta + 0.2 * luma_delta + 0.15 * edge_delta + 0.3 * motion_bonus)
            scores.append(float(score))
        return scores

    def _find_peak_positions(
        self,
        scores: list[float],
        threshold: float,
        max_peaks: int = 2,
        min_distance: int = 2,
    ) -> list[int]:
        if not scores:
            return []

        peak_candidates: list[tuple[int, float]] = []
        for index, score in enumerate(scores):
            left = scores[index - 1] if index > 0 else -1.0
            right = scores[index + 1] if index + 1 < len(scores) else -1.0
            if score >= threshold and score >= left and score >= right:
                peak_candidates.append((index + 1, score))

        if not peak_candidates:
            max_index = int(np.argmax(scores))
            if scores[max_index] >= threshold * 0.9:
                peak_candidates.append((max_index + 1, scores[max_index]))

        selected: list[int] = []
        for position, _score in sorted(peak_candidates, key=lambda item: item[1], reverse=True):
            if all(abs(position - kept) >= min_distance for kept in selected):
                selected.append(position)
            if len(selected) >= max_peaks:
                break
        return sorted(selected)

    def _adaptive_threshold(self, scores: list[float], percentile: float = 75.0) -> float:
        if not scores:
            return 1.0
        arr = np.array(scores, dtype=np.float64)
        percentile_threshold = float(np.percentile(arr, percentile))
        robust_threshold = float(np.median(arr) + 1.4826 * np.median(np.abs(arr - np.median(arr))))
        return min(max(percentile_threshold, robust_threshold), float(arr.max()))

    def _adaptive_confidence(self, score: float, scores: list[float]) -> float:
        if not scores:
            return 0.55
        arr = np.array(scores, dtype=np.float64)
        spread = max(float(arr.max() - arr.min()), 1e-6)
        relative = (score - float(arr.min())) / spread
        return min(0.92, 0.58 + 0.30 * relative)

    def _actor_motion(self, motion_score: float) -> str:
        if motion_score < 0.035:
            return "steady"
        if motion_score < 0.09:
            return "approach"
        if motion_score < 0.16:
            return "reposition"
        return "manipulate"

    def _allowed_actions_from_prompt(self, prompt_text: str) -> list[str]:
        match = re.search(r"Allowed action labels:\s*([a-zA-Z0-9_,\s]+)\.", prompt_text)
        if not match:
            return list(DEFAULT_ACTION_SET)
        return normalize_action_set(match.group(1).split(","))

    def _choose_allowed_action(self, prompt_text: str, preferred: list[str]) -> str:
        allowed = self._allowed_actions_from_prompt(prompt_text)
        for action in preferred:
            normalized = normalize_label(action)
            if normalized in allowed:
                return normalized
        return allowed[0] if allowed else "unknown"

    def _state_from_analysis(self, analysis: FrameAnalysis, prompt_text: str = "") -> ActionState:
        if analysis.motion_score < 0.035:
            action = self._choose_allowed_action(prompt_text, ["wait", "hold"])
            object_motion = "stationary"
            contact_state = "no_clear_contact"
        elif analysis.motion_score < 0.09:
            action = self._choose_allowed_action(prompt_text, ["reach_for", "move", "pick"])
            object_motion = "stationary"
            contact_state = "no_clear_contact"
        elif analysis.motion_score < 0.16:
            action = self._choose_allowed_action(prompt_text, ["move", "pull", "push", "retract"])
            object_motion = "moving_with_actor"
            contact_state = "possible_contact"
        else:
            action = self._choose_allowed_action(prompt_text, ["grasp", "pick", "close", "release", "lift"])
            object_motion = "moving_with_actor"
            contact_state = "maintained_contact"

        return ActionState(
            action=action,
            actor_motion=self._actor_motion(analysis.motion_score),
            contact_state=contact_state,
            object_motion=object_motion,
            target_object=f"{analysis.color_name} object",
        )

    def _summary_from_window(self, frames: list[SampledFrame]) -> str:
        analyses = [self._analyze_frame(frame) for frame in frames]
        colors = Counter(analysis.color_name for analysis in analyses)
        dominant_color = colors.most_common(1)[0][0]
        avg_motion = fmean(frame.motion_score for frame in frames)
        if avg_motion < 0.05:
            activity = "mostly steady observation"
        elif avg_motion < 0.11:
            activity = "controlled repositioning"
        else:
            activity = "active manipulation"
        return f"The window shows {activity} around a {dominant_color} workspace object."

    def propose_boundaries(
        self,
        window: Window,
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[LocalBoundaryProposal]:
        scores = self._transition_scores(window.frames)
        threshold = self._adaptive_threshold(scores, percentile=75.0)
        peak_positions = self._find_peak_positions(scores, threshold, max_peaks=2, min_distance=2)

        candidates: list[BoundaryCandidate] = []
        for position in peak_positions:
            before_analysis = self._analyze_frame(window.frames[position - 1])
            after_analysis = self._analyze_frame(window.frames[position])
            before_state = self._state_from_analysis(before_analysis, prompt_text=prompt_text)
            after_state = self._state_from_analysis(after_analysis, prompt_text=prompt_text)
            score = scores[position - 1]
            transition = normalize_transition(before_state.action, after_state.action)
            candidates.append(
                BoundaryCandidate(
                    boundary_sample_index=window.frames[position].sample_index,
                    boundary_original_frame_id=window.frames[position].original_frame_id,
                    boundary_timestamp=window.frames[position].timestamp_seconds,
                    before_state=before_state,
                    after_state=after_state,
                    transition_type=transition,
                    visual_evidence=[
                        f"Appearance change peaks at sample {window.frames[position].sample_index} with score {score:.3f}.",
                        f"The central region remains focused on a {after_analysis.color_name} object while the motion phase changes.",
                    ],
                    confidence=self._adaptive_confidence(score, scores),
                )
            )

        parsed = LocalBoundaryProposal(
            window_id=window.window_id,
            window_summary=self._summary_from_window(window.frames),
            boundary_candidates=candidates,
            prompt_version=prompt_version,
        )
        raw_request = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": prompt_version,
            "prompt_preview": prompt_text[:200],
            "window_id": window.window_id,
            "frames": [
                {
                    "sample_index": frame.sample_index,
                    "original_frame_id": frame.original_frame_id,
                    "timestamp_seconds": frame.timestamp_seconds,
                }
                for frame in window.frames
            ],
            "threshold": threshold,
        }
        raw_response = parsed.model_dump(mode="json")
        cache_key = deterministic_key(raw_request)
        return ClientCallResult(parsed=parsed, raw_request=raw_request, raw_response=raw_response, cache_key=cache_key)

    def verify_boundary(
        self,
        proposal_id: str,
        candidate: BoundaryCandidate,
        neighborhood: list[SampledFrame],
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[VerifiedBoundary]:
        scores = self._transition_scores(neighborhood)
        threshold = self._adaptive_threshold(scores, percentile=70.0)
        peak_positions = self._find_peak_positions(scores, threshold, max_peaks=1, min_distance=2)

        if not peak_positions:
            parsed = VerifiedBoundary(
                proposal_id=proposal_id,
                window_id=0,
                status="reject",
                original_boundary_sample_index=candidate.boundary_sample_index,
                verified_boundary_sample_index=None,
                verified_original_frame_id=None,
                verified_timestamp=None,
                before_action=candidate.before_state.action,
                after_action=candidate.after_state.action,
                transition_type=candidate.transition_type,
                visual_evidence=["No meaningful local transition peak was found in the verification neighborhood."],
                confidence=0.35,
            )
        else:
            best_position = peak_positions[0]
            best_score = scores[best_position - 1]
            before_analysis = self._analyze_frame(neighborhood[best_position - 1])
            after_analysis = self._analyze_frame(neighborhood[best_position])
            status = (
                "accept"
                if neighborhood[best_position].sample_index == candidate.boundary_sample_index
                else "shift_to_neighbor_frame"
            )
            before_state = self._state_from_analysis(before_analysis, prompt_text=prompt_text)
            after_state = self._state_from_analysis(after_analysis, prompt_text=prompt_text)
            parsed = VerifiedBoundary(
                proposal_id=proposal_id,
                window_id=0,
                status=status,
                original_boundary_sample_index=candidate.boundary_sample_index,
                verified_boundary_sample_index=neighborhood[best_position].sample_index,
                verified_original_frame_id=neighborhood[best_position].original_frame_id,
                verified_timestamp=neighborhood[best_position].timestamp_seconds,
                before_action=before_state.action,
                after_action=after_state.action,
                transition_type=normalize_transition(
                    before_state.action,
                    after_state.action,
                    candidate.transition_type,
                ),
                visual_evidence=[
                    f"Neighborhood change peaks at sample {neighborhood[best_position].sample_index} with score {best_score:.3f}.",
                    "The chosen frame is the earliest supplied frame where the new local motion phase is clearer.",
                ],
                confidence=self._adaptive_confidence(best_score, scores),
            )

        raw_request = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": prompt_version,
            "prompt_preview": prompt_text[:200],
            "proposal_id": proposal_id,
            "candidate": candidate.model_dump(mode="json"),
            "neighborhood_frame_ids": [frame.sample_index for frame in neighborhood],
            "threshold": threshold,
        }
        raw_response = parsed.model_dump(mode="json")
        cache_key = deterministic_key(raw_request)
        return ClientCallResult(parsed=parsed, raw_request=raw_request, raw_response=raw_response, cache_key=cache_key)

    def label_segment(
        self,
        segment: RawSegment,
        representative_frames: list[SampledFrame],
        total_segments: int,
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[SegmentLabel]:
        analyses = [self._analyze_frame(frame) for frame in representative_frames]
        avg_motion = fmean(frame.motion_score for frame in representative_frames)
        dominant_color = Counter(analysis.color_name for analysis in analyses).most_common(1)[0][0]
        duration = max(0.0, segment.end_time - segment.start_time)

        if avg_motion < 0.035:
            action_label = self._choose_allowed_action(prompt_text, ["wait", "hold"])
            actor_motion = "steady"
            contact_state = "no_clear_contact"
            object_motion = "stationary"
            goal = "monitor the workspace"
        elif avg_motion < 0.09:
            action_label = self._choose_allowed_action(
                prompt_text,
                ["reach_for", "move", "pick"] if segment.segment_id == 0 else ["move", "reach_for", "retract"],
            )
            actor_motion = "approach"
            contact_state = "no_clear_contact"
            object_motion = "stationary"
            goal = "align the gripper with the target region"
        elif avg_motion < 0.16:
            action_label = self._choose_allowed_action(
                prompt_text,
                ["move", "pull", "push", "place", "lower"] if segment.segment_id < total_segments - 1 else ["retract", "move"],
            )
            actor_motion = "transport"
            contact_state = "possible_contact"
            object_motion = "moving_with_actor"
            goal = "move the manipulated object or end-effector to a new position"
        else:
            action_label = self._choose_allowed_action(prompt_text, ["grasp", "pick", "close", "release", "lift"])
            actor_motion = "manipulate"
            contact_state = "maintained_contact"
            object_motion = "moving_with_actor"
            goal = "change the interaction state around the object"

        if duration < 1.5 and avg_motion >= 0.12:
            action_label = self._choose_allowed_action(prompt_text, ["move", "retract"])
            actor_motion = "reposition"
            goal = "perform a short adjustment before the next phase"

        primary_object = f"{dominant_color} object"
        parsed = SegmentLabel(
            action_label=normalize_label(action_label),
            description=f"The robot performs {normalize_label(action_label).replace('_', ' ')} around a {primary_object}.",
            primary_object=primary_object,
            secondary_objects=["workspace surface"],
            actor_motion=actor_motion,
            contact_state=contact_state,
            object_motion=object_motion,
            goal=goal,
            confidence=min(0.93, 0.55 + min(0.25, avg_motion * 1.5) + min(0.08, len(representative_frames) * 0.01)),
        )
        raw_request = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": prompt_version,
            "prompt_preview": prompt_text[:200],
            "segment": segment.model_dump(mode="json"),
            "representative_frame_ids": [frame.sample_index for frame in representative_frames],
            "total_segments": total_segments,
        }
        raw_response = parsed.model_dump(mode="json")
        cache_key = deterministic_key(raw_request)
        return ClientCallResult(parsed=parsed, raw_request=raw_request, raw_response=raw_response, cache_key=cache_key)

    def check_global_consistency(
        self,
        metadata: VideoMetadata,
        segments: list[LabeledSegment],
        boundaries: list[MergedBoundary],
        boundary_frames: dict[int, list[SampledFrame]],
        prompt_text: str,
        prompt_version: str,
    ) -> ClientCallResult[GlobalCheckResult]:
        issues: list[GlobalIssue] = []
        for previous, current in zip(segments, segments[1:]):
            if normalize_label(previous.action_label) == normalize_label(current.action_label):
                issues.append(
                    GlobalIssue(
                        type="merge_adjacent_segments",
                        segment_ids=[previous.segment_id, current.segment_id],
                        reason="Adjacent segments have the same normalized action label and no explicit semantic break.",
                        confidence=min(previous.confidence, current.confidence),
                    )
                )

        summary = "No explicit consistency problems detected." if not issues else f"Detected {len(issues)} constrained edit candidates."
        parsed = GlobalCheckResult(issues=issues, prompt_version=prompt_version, summary=summary)
        raw_request = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": prompt_version,
            "prompt_preview": prompt_text[:200],
            "video": metadata.model_dump(mode="json"),
            "segments": [segment.model_dump(mode="json") for segment in segments],
            "boundaries": [boundary.model_dump(mode="json") for boundary in boundaries],
            "boundary_contexts": {key: [frame.sample_index for frame in value] for key, value in boundary_frames.items()},
        }
        raw_response = parsed.model_dump(mode="json")
        cache_key = deterministic_key(raw_request)
        return ClientCallResult(parsed=parsed, raw_request=raw_request, raw_response=raw_response, cache_key=cache_key)
