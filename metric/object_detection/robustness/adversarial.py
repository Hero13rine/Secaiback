"""目标检测模型的对抗鲁棒性指标."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from metric.object_detection.basic.detection import DetectionSample, ObjectDetectionEvaluator
from metric.object_detection.robustness.robustUtils import compute_detection_errors


# 类型定义
PredictionLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
GroundTruthLike = Union[DetectionSample, Mapping[str, Sequence[float]]]


# 允许的指标配置（在 evaluate_robustness 中共用）
SCALAR_ROBUSTNESS_METRICS = frozenset({"map_drop_rate", "miss_rate", "false_detection_rate"})
PER_CLASS_METRIC_NAME = "per_class_map"
PER_CLASS_METRIC_ALIASES = frozenset(
    {"per_class_map", "per_class_ap", "per_class_clean_map", "per_class_adversarial_map"}
)
ALL_METRIC_ALIASES = {
    **{metric: metric for metric in SCALAR_ROBUSTNESS_METRICS},
    **{alias: PER_CLASS_METRIC_NAME for alias in PER_CLASS_METRIC_ALIASES},
}


@dataclass(frozen=True)
class RobustnessMetrics:
    """目标检测鲁棒性指标容器."""

    map_drop_rate: float
    miss_rate: float
    false_detection_rate: float
    per_class_clean_map: Mapping[str, float] = field(default_factory=dict)
    per_class_adversarial_map: Mapping[str, float] = field(default_factory=dict)
    clean_map: float = 0.0
    adversarial_map: float = 0.0
    clean_miss_rate: float = 0.0
    clean_false_detection_rate: float = 0.0


@dataclass(frozen=True)
class AttackEvaluationResult:
    """对抗攻击评估的结构化结果.

    Attributes:
        attack_name (str): 对抗攻击的标识符
        metrics (RobustnessMetrics): 整体鲁棒性指标
        metadata (Mapping[str, Any]): 附加的上下文信息, 例如调度参数
    """

    attack_name: str
    metrics: RobustnessMetrics
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
        metadata: Optional[Mapping[str, Any]] = None,
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
        base_map, base_per_class = self._compute_map_with_details(base_samples, gt_samples)
        print(f"  进度: 计算对抗攻击后mAP...")
        adv_map, adv_per_class = self._compute_map_with_details(adv_samples, gt_samples)

        print(f"  进度: 计算基线检测错误...")
        baseline_errors = compute_detection_errors(
            base_samples,
            gt_samples,
            self.iou_threshold,
        )

        print(f"  进度: 计算对抗检测错误...")
        adversarial_errors = compute_detection_errors(
            adv_samples,
            gt_samples,
            self.iou_threshold,
        )

        metrics = self._compose_metrics(
            normalized_metrics,
            base_map,
            adv_map,
            baseline_errors,
            adversarial_errors,
            base_per_class,
            adv_per_class,
        )

        print(f"  进度: 攻击 '{attack_name}' 评估完成")
        return AttackEvaluationResult(
            attack_name=attack_name,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )

    def _compose_metrics(
        self,
        metric_filter: Optional[Mapping[str, None]],
        baseline_map: float,
        adversarial_map: float,
        baseline_errors: Tuple[int, int, int, int],
        adversarial_errors: Tuple[int, int, int, int],
        baseline_per_class: Optional[Mapping[str, float]] = None,
        adversarial_per_class: Optional[Mapping[str, float]] = None,
    ) -> RobustnessMetrics:
        """创建应用可选过滤的指标容器."""

        map_drop_rate = self._map_drop(baseline_map, adversarial_map)

        (
            baseline_misses,
            baseline_false_positives,
            baseline_ground_truth_total,
            baseline_prediction_total,
        ) = baseline_errors
        (
            adversarial_misses,
            adversarial_false_positives,
            adversarial_ground_truth_total,
            adversarial_prediction_total,
        ) = adversarial_errors

        ground_truth_total = (
            adversarial_ground_truth_total or baseline_ground_truth_total
        )
        prediction_total = (
            adversarial_prediction_total or baseline_prediction_total
        )

        miss_rate = (
            adversarial_misses / ground_truth_total if ground_truth_total > 0 else 0.0
        )
        clean_miss_rate = (
            baseline_misses / baseline_ground_truth_total
            if baseline_ground_truth_total > 0
            else 0.0
        )

        false_detection_rate = (
            adversarial_false_positives / prediction_total if prediction_total > 0 else 0.0
        )
        clean_false_detection_rate = (
            baseline_false_positives / baseline_prediction_total
            if baseline_prediction_total > 0
            else 0.0
        )

        metric_values: Dict[str, float] = {
            "map_drop_rate": map_drop_rate,
            "miss_rate": miss_rate,
            "false_detection_rate": false_detection_rate,
        }

        include_per_class_metrics = (
            metric_filter is None or PER_CLASS_METRIC_NAME in metric_filter
        )

        if metric_filter:
            metric_values = {
                key: metric_values[key]
                for key in metric_values
                if key in metric_filter
            }

        per_class_clean = baseline_per_class if include_per_class_metrics else None
        per_class_adv = adversarial_per_class if include_per_class_metrics else None

        return RobustnessMetrics(
            map_drop_rate=metric_values.get("map_drop_rate", 0.0),
            miss_rate=metric_values.get("miss_rate", 0.0),
            false_detection_rate=metric_values.get("false_detection_rate", 0.0),
            per_class_clean_map=per_class_clean or {},
            per_class_adversarial_map=per_class_adv or {},
            clean_map=baseline_map,
            adversarial_map=adversarial_map,
            clean_miss_rate=clean_miss_rate,
            clean_false_detection_rate=clean_false_detection_rate,
        )

    def _compute_map(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> float:
        """计算mAP指标."""

        map_value, _ = self._compute_map_with_details(predictions, ground_truths)
        return map_value

    def _compute_map_with_details(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> Tuple[float, Mapping[str, float]]:
        """计算mAP并返回逐类AP."""

        evaluation = self._detector.evaluate(list(predictions), list(ground_truths))
        details = evaluation.get(self.iou_threshold, {})
        per_class = self._normalize_per_class(details.get("per_class"))
        return float(details.get("map", 0.0)), per_class

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
    def _normalize_metric_selection(  # TODO 添加perclass 的参数filter
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
            canonical = ALL_METRIC_ALIASES.get(normalized)
            if canonical:
                filtered[canonical] = None
        return filtered or None

    @staticmethod
    def _map_drop(baseline_map: float, adversarial_map: float) -> float:
        """计算mAP下降率."""

        if baseline_map <= 0:
            return 0.0
        drop = (baseline_map - adversarial_map) / baseline_map
        return float(max(0.0, drop))

    @staticmethod
    def _normalize_per_class(per_class_payload: Any) -> Mapping[str, float]:
        if not isinstance(per_class_payload, Mapping):
            return {}
        normalized: Dict[str, float] = {}
        for key, value in per_class_payload.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

__all__ = [
    "AttackEvaluationResult",
    "AdversarialRobustnessEvaluator",
    "PredictionLike",
    "RobustnessMetrics",
    "ALL_METRIC_ALIASES",
    "PER_CLASS_METRIC_NAME",
    "SCALAR_ROBUSTNESS_METRICS",
]
