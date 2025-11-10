"""目标检测模型的对抗鲁棒性指标."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from metric.object_detection.basic.detection import DetectionSample, ObjectDetectionEvaluator


# 类型定义
PredictionLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
GroundTruthLike = Union[DetectionSample, Mapping[str, Sequence[float]]]


@dataclass(frozen=True)
class RobustnessMetrics:
    """目标检测鲁棒性指标容器."""

    map_drop_rate: float
    miss_rate: float
    false_detection_rate: float


@dataclass(frozen=True)
class AttackEvaluationResult:
    """对抗攻击评估的结构化结果."""

    attack_name: str
    metrics: RobustnessMetrics


class AdversarialRobustnessEvaluator:
    """计算对抗攻击检测器的鲁棒性指标."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """初始化对抗鲁棒性评估器."""

        self.iou_threshold = float(iou_threshold)
        self._detector = ObjectDetectionEvaluator([self.iou_threshold])

    def evaluate_attack(
        self,
        attack_name: str,
        baseline_predictions: Sequence[PredictionLike],
        adversarial_predictions: Sequence[PredictionLike],
        ground_truths: Sequence[GroundTruthLike],
        metrics_to_report: Optional[Iterable[str]] = None,
    ) -> AttackEvaluationResult:
        """评估单个对抗攻击."""

        normalized_metrics = self._normalize_metric_selection(metrics_to_report)

        base_samples = self._to_samples(
            baseline_predictions,
            sample_type="prediction",
        )
        adv_samples = self._to_samples(
            adversarial_predictions,
            sample_type="prediction",
        )
        gt_samples = self._to_samples(
            ground_truths,
            sample_type="ground_truth",
        )

        if not gt_samples:
            raise ValueError("必须提供真实标注")
        if len(base_samples) != len(gt_samples):
            raise ValueError("基线预测数量与真实标注数量不匹配")
        if len(adv_samples) != len(gt_samples):
            raise ValueError("对抗预测数量与真实标注数量不匹配")

        print(f"  进度: 计算基线mAP...")
        base_map = self._compute_map(base_samples, gt_samples)
        print(f"  进度: 计算对抗攻击后mAP...")
        adv_map = self._compute_map(adv_samples, gt_samples)

        print(f"  进度: 计算检测错误...")
        misses, false_positives, gt_count, pred_count = self._compute_detection_errors(
            adv_samples,
            gt_samples,
        )

        metrics = self._compose_metrics(
            base_map,
            adv_map,
            misses,
            false_positives,
            gt_count,
            pred_count,
            normalized_metrics,
        )

        print(f"  进度: 攻击 '{attack_name}' 评估完成")
        return AttackEvaluationResult(
            attack_name=attack_name,
            metrics=metrics,
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
    ) -> RobustnessMetrics:
        """创建应用可选过滤的指标容器."""

        map_drop_rate = self._map_drop(baseline_map, adversarial_map)
        miss_rate = misses / ground_truth_total if ground_truth_total > 0 else 0.0
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

        return RobustnessMetrics(
            map_drop_rate=metric_values.get("map_drop_rate", 0.0),
            miss_rate=metric_values.get("miss_rate", 0.0),
            false_detection_rate=metric_values.get("false_detection_rate", 0.0),
        )

    def _compute_map(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> float:
        """计算mAP指标."""

        evaluation = self._detector.evaluate(list(predictions), list(ground_truths))
        details = evaluation.get(self.iou_threshold, {})
        return float(details.get("map", 0.0))

    def _compute_detection_errors(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> Tuple[int, int, int, int]:
        """使用贪婪IoU匹配计算漏检和误检."""

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
        """将任意输入的预测/目标标准化为样本."""

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
                    raise ValueError(f"不支持的样本类型: {sample_type}")
        return samples

    @staticmethod
    def _normalize_metric_selection(
        metrics_to_report: Optional[Iterable[str]],
    ) -> Optional[Mapping[str, None]]:
        """标准化指标选择."""

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
    def _map_drop(baseline_map: float, adversarial_map: float) -> float:
        """计算mAP下降率."""

        if baseline_map <= 0:
            return 0.0
        drop = (baseline_map - adversarial_map) / baseline_map
        return float(max(0.0, drop))


def _greedy_iou_match(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float
) -> List[Tuple[int, int]]:
    """Greedy IoU matching for axis-aligned boxes."""

    return _greedy_iou_match_generic(
        pred_boxes, gt_boxes, threshold, _pairwise_iou_axis_aligned
    )


def _greedy_iou_match_generic(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float,
    pairwise_iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> List[Tuple[int, int]]:
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return []

    iou_matrix = pairwise_iou_fn(pred_boxes, gt_boxes)
    if iou_matrix.size == 0:
        return []

    matches: List[Tuple[int, int]] = []

    while True:
        flat_index = int(np.argmax(iou_matrix))
        best_iou = float(iou_matrix.flat[flat_index])
        if not np.isfinite(best_iou) or best_iou < threshold or best_iou <= 0.0:
            break
        pred_idx, gt_idx = divmod(flat_index, iou_matrix.shape[1])
        matches.append((pred_idx, gt_idx))
        iou_matrix[pred_idx, :] = -1.0
        iou_matrix[:, gt_idx] = -1.0

    return matches


def _pairwise_iou_axis_aligned(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Vectorised pairwise IoU for axis-aligned bounding boxes."""

    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    pred = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    pred_exp = pred[:, None, :]
    gt_exp = gt[None, :, :]

    ix1 = np.maximum(pred_exp[..., 0], gt_exp[..., 0])
    iy1 = np.maximum(pred_exp[..., 1], gt_exp[..., 1])
    ix2 = np.minimum(pred_exp[..., 2], gt_exp[..., 2])
    iy2 = np.minimum(pred_exp[..., 3], gt_exp[..., 3])

    inter_w = np.maximum(ix2 - ix1, 0.0)
    inter_h = np.maximum(iy2 - iy1, 0.0)
    intersection = inter_w * inter_h

    pred_area = np.maximum(pred_exp[..., 2] - pred_exp[..., 0], 0.0) * np.maximum(
        pred_exp[..., 3] - pred_exp[..., 1], 0.0
    )
    gt_area = np.maximum(gt_exp[..., 2] - gt_exp[..., 0], 0.0) * np.maximum(
        gt_exp[..., 3] - gt_exp[..., 1], 0.0
    )
    union = pred_area + gt_area - intersection
    union = np.maximum(union, np.finfo(np.float32).eps)

    return (intersection / union).astype(np.float32)


__all__ = [
    "AttackEvaluationResult",
    "AdversarialRobustnessEvaluator",
    "PredictionLike",
    "RobustnessMetrics",
]
