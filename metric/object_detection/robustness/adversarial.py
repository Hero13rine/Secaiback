"""Adversarial robustness metrics for object detection models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from metric.object_detection.basic.detection import DetectionSample, ObjectDetectionEvaluator


PredictionLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
GroundTruthLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
RotationKey = Union[int, float, str]


@dataclass(frozen=True)
class RotationRobustnessMetrics:
    """Grouped robustness metrics for a single rotation view."""

    map_drop_rate: float
    miss_rate: float
    false_detection_rate: float


@dataclass(frozen=True)
class AttackEvaluationResult:
    """Structured result for an adversarial attack evaluation."""

    attack_name: str
    overall: RotationRobustnessMetrics
    by_rotation: Dict[float, RotationRobustnessMetrics]


class AdversarialRobustnessEvaluator:
    """Compute robustness indicators for adversarially attacked detectors."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)
        self._detector = ObjectDetectionEvaluator([self.iou_threshold])

    def evaluate_attack(
        self,
        attack_name: str,
        baseline_predictions: Mapping[float, Sequence[PredictionLike]],
        adversarial_predictions: Mapping[float, Sequence[PredictionLike]],
        ground_truths: Mapping[float, Sequence[GroundTruthLike]],
        metrics_to_report: Optional[Iterable[str]] = None,
    ) -> AttackEvaluationResult:
        """Evaluate one adversarial attack under multiple rotation angles.

        Args:
            attack_name: Identifier of the adversarial attack.
            baseline_predictions: Mapping from rotation angle to the detector
                predictions without perturbations.
            adversarial_predictions: Mapping from rotation angle to the detector
                predictions after perturbations.
            ground_truths: Mapping from rotation angle to ground-truth
                annotations.
            metrics_to_report: Optional iterable restricting the metrics to be
                returned. Supported names: ``map_drop_rate``, ``miss_rate`` and
                ``false_detection_rate``.

        Returns:
            AttackEvaluationResult containing aggregated and per-rotation
            statistics.
        """

        normalized_metrics = self._normalize_metric_selection(metrics_to_report)
        rotation_union = self._collect_rotation_keys(
            baseline_predictions, adversarial_predictions, ground_truths
        )
        rotation_results: Dict[float, RotationRobustnessMetrics] = {}

        baseline_maps: List[float] = []
        adversarial_maps: List[float] = []
        total_misses = 0
        total_false_positives = 0
        total_gt = 0
        total_predictions = 0

        for rotation in rotation_union:
            base_samples = self._to_samples(
                baseline_predictions.get(rotation, []),
                sample_type="prediction",
            )
            adv_samples = self._to_samples(
                adversarial_predictions.get(rotation, []),
                sample_type="prediction",
            )
            gt_samples = self._to_samples(
                ground_truths.get(rotation, []),
                sample_type="ground_truth",
            )
            if not gt_samples:
                raise ValueError(
                    "Ground truth annotations must be provided for every rotation"
                )
            if len(base_samples) != len(gt_samples):
                raise ValueError(
                    f"Baseline predictions ({len(base_samples)}) and ground truths "
                    f"({len(gt_samples)}) count mismatch for rotation {rotation}"
                )
            if len(adv_samples) != len(gt_samples):
                raise ValueError(
                    f"Adversarial predictions ({len(adv_samples)}) and ground truths "
                    f"({len(gt_samples)}) count mismatch for rotation {rotation}"
                )

            base_map = self._compute_map(base_samples, gt_samples)
            adv_map = self._compute_map(adv_samples, gt_samples)
            baseline_maps.append(base_map)
            adversarial_maps.append(adv_map)

            misses, false_positives, gt_count, pred_count = self._compute_detection_errors(
                adv_samples, gt_samples
            )
            total_misses += misses
            total_false_positives += false_positives
            total_gt += gt_count
            total_predictions += pred_count

            rotation_results[rotation] = self._compose_metrics(
                base_map,
                adv_map,
                misses,
                false_positives,
                gt_count,
                pred_count,
                normalized_metrics,
            )

        overall = self._compose_metrics(
            float(np.mean(baseline_maps)) if baseline_maps else 0.0,
            float(np.mean(adversarial_maps)) if adversarial_maps else 0.0,
            total_misses,
            total_false_positives,
            total_gt,
            total_predictions,
            normalized_metrics,
        )

        return AttackEvaluationResult(
            attack_name=attack_name,
            overall=overall,
            by_rotation=dict(sorted(rotation_results.items())),
        )

    def _compose_metrics(
        self,
        baseline_map: float,
        adversarial_map: float,
        misses: int,
        false_positives: int,
        ground_truth_total: int,
        prediction_total: int,
        metric_filter: Optional[Mapping[str, None]],
    ) -> RotationRobustnessMetrics:
        """Create a metrics container applying optional filtering."""

        map_drop_rate = self._map_drop(baseline_map, adversarial_map)
        miss_rate = (
            misses / ground_truth_total if ground_truth_total > 0 else 0.0
        )
        false_detection_rate = (
            false_positives / prediction_total if prediction_total > 0 else 0.0
        )

        metric_values: Dict[str, float] = {
            "map_drop_rate": map_drop_rate,
            "miss_rate": miss_rate,
            "false_detection_rate": false_detection_rate,
        }
        if metric_filter:
            metric_values = {
                key: metric_values[key]
                for key in metric_values
                if key in metric_filter
            }
        return RotationRobustnessMetrics(
            map_drop_rate=metric_values.get("map_drop_rate", 0.0),
            miss_rate=metric_values.get("miss_rate", 0.0),
            false_detection_rate=metric_values.get("false_detection_rate", 0.0),
        )

    def _compute_map(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> float:
        evaluation = self._detector.evaluate(list(predictions), list(ground_truths))
        details = evaluation.get(self.iou_threshold, {})
        return float(details.get("map", 0.0))

    def _compute_detection_errors(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> Tuple[int, int, int, int]:
        """Calculate missed and false detections using greedy IoU matching."""

        misses = 0
        false_positives = 0
        total_gt = 0
        total_predictions = 0

        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.boxes
            pred_boxes = pred.boxes
            total_gt += gt_boxes.shape[0]
            total_predictions += pred_boxes.shape[0]
            matches = _greedy_iou_match(pred_boxes, gt_boxes, self.iou_threshold)
            misses += gt_boxes.shape[0] - len(matches)
            false_positives += pred_boxes.shape[0] - len(matches)

        return misses, false_positives, total_gt, total_predictions

    def _to_samples(
        self,
        entries: Sequence[PredictionLike],
        sample_type: str,
    ) -> List[DetectionSample]:
        """Normalize arbitrary input predictions/targets into samples."""

        samples: List[DetectionSample] = []
        for entry in entries:
            if isinstance(entry, DetectionSample):
                samples.append(entry)
            else:
                if sample_type == "prediction":
                    samples.append(DetectionSample.from_prediction(dict(entry)))
                elif sample_type == "ground_truth":
                    samples.append(DetectionSample.from_ground_truth(dict(entry)))
                else:
                    raise ValueError(f"Unsupported sample type: {sample_type}")
        return samples

    @staticmethod
    def _normalize_metric_selection(
        metrics_to_report: Optional[Iterable[str]],
    ) -> Optional[Mapping[str, None]]:
        if metrics_to_report is None:
            return None
        filtered: Dict[str, None] = {}
        for name in metrics_to_report:
            if not isinstance(name, str):
                continue
            normalized = name.strip().lower()
            if normalized in {"map_drop_rate", "miss_rate", "false_detection_rate"}:
                filtered[normalized] = None
        return filtered or None

    @staticmethod
    def _collect_rotation_keys(
        *mappings: Mapping[RotationKey, Sequence[PredictionLike]]
    ) -> List[float]:
        keys: List[float] = []
        seen: Dict[float, None] = {}
        for mapping in mappings:
            for raw_key in mapping.keys():
                try:
                    key = float(raw_key)
                except (TypeError, ValueError):
                    continue
                if key not in seen:
                    seen[key] = None
                    keys.append(key)
        keys.sort()
        return keys

    @staticmethod
    def _map_drop(baseline_map: float, adversarial_map: float) -> float:
        if baseline_map <= 0:
            return 0.0
        drop = (baseline_map - adversarial_map) / baseline_map
        return float(max(0.0, drop))


def _greedy_iou_match(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float
) -> List[Tuple[int, int]]:
    """Perform greedy IoU matching between prediction and ground-truth boxes."""

    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return []

    iou_matrix = _pairwise_iou(pred_boxes, gt_boxes)
    matches: List[Tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    while True:
        remaining = [
            (i, j, iou_matrix[i, j])
            for i in range(iou_matrix.shape[0])
            for j in range(iou_matrix.shape[1])
            if i not in used_pred and j not in used_gt
        ]
        if not remaining:
            break
        best = max(remaining, key=lambda item: item[2])
        i, j, iou = best
        if iou < threshold:
            break
        matches.append((i, j))
        used_pred.add(i)
        used_gt.add(j)

    return matches


def _pairwise_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for two sets of boxes."""

    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    pred = pred_boxes.astype(np.float32)
    gt = gt_boxes.astype(np.float32)

    pred = pred.reshape(-1, 4)
    gt = gt.reshape(-1, 4)

    iou = np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float32)
    for i, p in enumerate(pred):
        px1, py1, px2, py2 = p
        p_area = max(px2 - px1, 0) * max(py2 - py1, 0)
        for j, g in enumerate(gt):
            gx1, gy1, gx2, gy2 = g
            g_area = max(gx2 - gx1, 0) * max(gy2 - gy1, 0)
            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            inter_w = max(ix2 - ix1, 0)
            inter_h = max(iy2 - iy1, 0)
            intersection = inter_w * inter_h
            union = p_area + g_area - intersection
            iou[i, j] = intersection / union if union > 0 else 0.0
    return iou

