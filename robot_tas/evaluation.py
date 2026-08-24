from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from robot_tas.normalization import normalize_label
from robot_tas.schemas import LabeledSegment


@dataclass(slots=True)
class EvalSegment:
    start_frame_id: int
    end_frame_id: int
    label: str


def _iou(a: EvalSegment, b: EvalSegment) -> float:
    inter_start = max(a.start_frame_id, b.start_frame_id)
    inter_end = min(a.end_frame_id, b.end_frame_id)
    intersection = max(0, inter_end - inter_start)
    union = max(a.end_frame_id, b.end_frame_id) - min(a.start_frame_id, b.start_frame_id)
    if union <= 0:
        return 0.0
    return intersection / union


def boundary_precision_recall_f1(
    predicted_boundaries: Iterable[int],
    ground_truth_boundaries: Iterable[int],
    tolerance_frames: int,
) -> dict[str, float]:
    predicted = list(predicted_boundaries)
    truth = list(ground_truth_boundaries)
    matched_truth: set[int] = set()
    true_positive = 0
    for boundary in predicted:
        for truth_index, truth_boundary in enumerate(truth):
            if truth_index in matched_truth:
                continue
            if abs(boundary - truth_boundary) <= tolerance_frames:
                matched_truth.add(truth_index)
                true_positive += 1
                break

    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_absolute_boundary_error(
    predicted_boundaries: Iterable[int],
    ground_truth_boundaries: Iterable[int],
) -> float | None:
    predicted = list(predicted_boundaries)
    truth = list(ground_truth_boundaries)
    if not predicted or not truth:
        return None
    errors = [min(abs(boundary - target) for target in truth) for boundary in predicted]
    return sum(errors) / len(errors)


def framewise_accuracy(
    predicted_segments: list[EvalSegment],
    ground_truth_segments: list[EvalSegment],
    total_frames: int,
) -> float:
    if total_frames <= 0:
        return 0.0

    predicted_labels = ["background"] * total_frames
    truth_labels = ["background"] * total_frames

    for segment in predicted_segments:
        for frame_id in range(max(0, segment.start_frame_id), min(total_frames, segment.end_frame_id)):
            predicted_labels[frame_id] = normalize_label(segment.label)
    for segment in ground_truth_segments:
        for frame_id in range(max(0, segment.start_frame_id), min(total_frames, segment.end_frame_id)):
            truth_labels[frame_id] = normalize_label(segment.label)

    matches = sum(1 for pred, truth in zip(predicted_labels, truth_labels) if pred == truth)
    return matches / total_frames


def segmental_f1(
    predicted_segments: list[EvalSegment],
    ground_truth_segments: list[EvalSegment],
    thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict[str, float]:
    results: dict[str, float] = {}
    for threshold in thresholds:
        matched_truth: set[int] = set()
        true_positive = 0
        for prediction in predicted_segments:
            best_match_index = None
            best_iou = 0.0
            for truth_index, truth in enumerate(ground_truth_segments):
                if truth_index in matched_truth:
                    continue
                if normalize_label(prediction.label) != normalize_label(truth.label):
                    continue
                overlap = _iou(prediction, truth)
                if overlap > best_iou:
                    best_iou = overlap
                    best_match_index = truth_index
            if best_match_index is not None and best_iou >= threshold:
                matched_truth.add(best_match_index)
                true_positive += 1
        precision = true_positive / len(predicted_segments) if predicted_segments else 0.0
        recall = true_positive / len(ground_truth_segments) if ground_truth_segments else 0.0
        f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)
        results[f"f1@{threshold:.2f}"] = f1
    return results


def edit_score(predicted_labels: list[str], ground_truth_labels: list[str]) -> float:
    normalized_pred = [normalize_label(label) for label in predicted_labels]
    normalized_truth = [normalize_label(label) for label in ground_truth_labels]
    rows = len(normalized_pred) + 1
    cols = len(normalized_truth) + 1
    dp = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if normalized_pred[row - 1] == normalized_truth[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + cost,
            )
    distance = dp[-1][-1]
    normalizer = max(len(normalized_pred), len(normalized_truth), 1)
    return 1.0 - (distance / normalizer)


def over_under_segmentation_ratio(
    predicted_count: int,
    ground_truth_count: int,
) -> dict[str, float]:
    if ground_truth_count <= 0:
        return {"over_segmentation_ratio": 0.0, "under_segmentation_ratio": 0.0}
    return {
        "over_segmentation_ratio": max(0.0, (predicted_count - ground_truth_count) / ground_truth_count),
        "under_segmentation_ratio": max(0.0, (ground_truth_count - predicted_count) / ground_truth_count),
    }


def segments_from_labeled(predicted_segments: list[LabeledSegment]) -> list[EvalSegment]:
    return [
        EvalSegment(
            start_frame_id=segment.start_frame_id,
            end_frame_id=segment.end_frame_id,
            label=segment.action_label,
        )
        for segment in predicted_segments
    ]
