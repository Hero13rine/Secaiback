"""目标检测模型的对抗鲁棒性指标."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from metric.object_detection.basic.detection import DetectionSample, ObjectDetectionEvaluator
from metric.object_detection.robustness.matching import compute_detection_errors


# 类型定义
PredictionLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
GroundTruthLike = Union[DetectionSample, Mapping[str, Sequence[float]]]


@dataclass(frozen=True)
class RobustnessMetrics:
    """目标检测鲁棒性指标容器.
    Attributes:
    map_drop_rate(float): mAP下降率，表示模型性能下降的程度
    miss_rate(float): 漏检率，表示未检测到的真实目标比例
    false_detection_rate(float): 误检率，表示错误检测的比例
    map_drop_rate: float
    miss_rate: float
    false_detection_rate: float
"""

@dataclass(frozen=True)
class AttackEvaluationResult:
    """对抗攻击评估的结构化结果.
     Attributes:
        attack_name (str): 对抗攻击的标识符
        overall (RotationRobustnessMetrics): 整体鲁棒性指标
    """

    attack_name: str
    metrics: RobustnessMetrics


class AdversarialRobustnessEvaluator:
    """计算对抗攻击检测器的鲁棒性指标."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """初始化对抗鲁棒性评估器.
         Args:
            iou_threshold (float): IoU阈值，用于匹配预测框和真实框，默认为0.5
        """

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
        misses, false_positives, gt_count, pred_count = compute_detection_errors(
            adv_samples,
            gt_samples,
            self.iou_threshold,
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
__all__ = [
    "AttackEvaluationResult",
    "AdversarialRobustnessEvaluator",
    "PredictionLike",
    "RobustnessMetrics",
]
