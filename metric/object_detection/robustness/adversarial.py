"""目标检测模型的对抗鲁棒性指标."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from metric.object_detection.basic.detection import DetectionSample, ObjectDetectionEvaluator


# 类型定义
PredictionLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
GroundTruthLike = Union[DetectionSample, Mapping[str, Sequence[float]]]
RotationKey = Union[int, float, str]


@dataclass(frozen=True)
class RotationRobustnessMetrics:
    """单个旋转视角的分组鲁棒性指标.

    Attributes:
        map_drop_rate (float): mAP下降率，表示模型性能下降的程度
        miss_rate (float): 漏检率，表示未检测到的真实目标比例
        false_detection_rate (float): 误检率，表示错误检测的比例
    """

    map_drop_rate: float
    miss_rate: float
    false_detection_rate: float


@dataclass(frozen=True)
class AttackEvaluationResult:
    """对抗攻击评估的结构化结果.

    Attributes:
        attack_name (str): 对抗攻击的标识符
        overall (RotationRobustnessMetrics): 整体鲁棒性指标
        by_rotation (Dict[float, RotationRobustnessMetrics]): 按旋转角度分组的鲁棒性指标
    """

    attack_name: str
    overall: RotationRobustnessMetrics
    by_rotation: Dict[float, RotationRobustnessMetrics]


class AdversarialRobustnessEvaluator:
    """计算对抗攻击检测器的鲁棒性指标."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """初始化对抗鲁棒性评估器.

        Args:
            iou_threshold (float): IoU阈值，用于匹配预测框和真实框，默认为0.5
        """
        self.iou_threshold = float(iou_threshold)
        # 初始化目标检测评估器，用于计算mAP等指标
        self._detector = ObjectDetectionEvaluator([self.iou_threshold])

    def evaluate_attack(
        self,
        attack_name: str,
        baseline_predictions: Mapping[float, Sequence[PredictionLike]],
        adversarial_predictions: Mapping[float, Sequence[PredictionLike]],
        ground_truths: Mapping[float, Sequence[GroundTruthLike]],
        metrics_to_report: Optional[Iterable[str]] = None,
    ) -> AttackEvaluationResult:
        """评估在多个旋转角度下的单个对抗攻击.

        Args:
            attack_name: 对抗攻击的标识符
            baseline_predictions: 从旋转角度到无扰动检测器预测的映射
            adversarial_predictions: 从旋转角度到扰动后检测器预测的映射
            ground_truths: 从旋转角度到真实标注的映射
            metrics_to_report: 可选的可迭代对象，限制返回的指标。
                支持的名称：``map_drop_rate``、``miss_rate`` 和 ``false_detection_rate``

        Returns:
            AttackEvaluationResult 包含聚合的和按旋转角度统计的结果

        Raises:
            ValueError: 当真实标注未提供或预测与真实标注数量不匹配时
        """

        # 标准化需要报告的指标
        normalized_metrics = self._normalize_metric_selection(metrics_to_report)
        # 收集所有旋转角度键
        rotation_union = self._collect_rotation_keys(
            baseline_predictions, adversarial_predictions, ground_truths
        )
        rotation_results: Dict[float, RotationRobustnessMetrics] = {}

        # 存储所有旋转角度的mAP值
        baseline_maps: List[float] = []
        adversarial_maps: List[float] = []
        
        # 累计统计信息
        total_misses = 0
        total_false_positives = 0
        total_gt = 0
        total_predictions = 0

        # 遍历每个旋转角度进行评估
        for rotation in rotation_union:
            # 获取基线预测样本
            base_samples = self._to_samples(
                baseline_predictions.get(rotation, []),
                sample_type="prediction",
            )
            # 获取对抗攻击后的预测样本
            adv_samples = self._to_samples(
                adversarial_predictions.get(rotation, []),
                sample_type="prediction",
            )
            # 获取真实标注样本
            gt_samples = self._to_samples(
                ground_truths.get(rotation, []),
                sample_type="ground_truth",
            )
            
            # 验证数据完整性
            if not gt_samples:
                raise ValueError(
                    "必须为每个旋转角度提供真实标注"
                )
            if len(base_samples) != len(gt_samples):
                raise ValueError(
                    f"基线预测 ({len(base_samples)}) 和真实标注 "
                    f"({len(gt_samples)}) 数量不匹配，旋转角度 {rotation}"
                )
            if len(adv_samples) != len(gt_samples):
                raise ValueError(
                    f"对抗预测 ({len(adv_samples)}) 和真实标注 "
                    f"({len(gt_samples)}) 数量不匹配，旋转角度 {rotation}"
                )

            # 计算基线和对抗攻击后的mAP
            base_map = self._compute_map(base_samples, gt_samples)
            adv_map = self._compute_map(adv_samples, gt_samples)
            baseline_maps.append(base_map)
            adversarial_maps.append(adv_map)

            # 计算检测错误（漏检和误检）
            misses, false_positives, gt_count, pred_count = self._compute_detection_errors(
                adv_samples, gt_samples
            )
            total_misses += misses
            total_false_positives += false_positives
            total_gt += gt_count
            total_predictions += pred_count

            # 组合该旋转角度的指标
            rotation_results[rotation] = self._compose_metrics(
                base_map,
                adv_map,
                misses,
                false_positives,
                gt_count,
                pred_count,
                normalized_metrics,
            )

        # 计算整体指标（所有旋转角度的平均值）
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
        """创建应用可选过滤的指标容器.

        Args:
            baseline_map: 基线mAP值
            adversarial_map: 对抗攻击后的mAP值
            misses: 漏检数量
            false_positives: 误检数量
            ground_truth_total: 真实目标总数
            prediction_total: 预测总数
            metric_filter: 指标过滤器

        Returns:
            RotationRobustnessMetrics 包含计算的指标
        """

        # 计算mAP下降率
        map_drop_rate = self._map_drop(baseline_map, adversarial_map)
        # 计算漏检率
        miss_rate = (
            misses / ground_truth_total if ground_truth_total > 0 else 0.0
        )
        # 计算误检率
        false_detection_rate = (
            false_positives / prediction_total if prediction_total > 0 else 0.0
        )

        # 构建指标字典
        metric_values: Dict[str, float] = {
            "map_drop_rate": map_drop_rate,
            "miss_rate": miss_rate,
            "false_detection_rate": false_detection_rate,
        }
        
        # 应用指标过滤器
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
        """计算mAP指标
        
        Args:
            predictions: 预测样本序列
            ground_truths: 真实标注样本序列

        Returns:
            float: 计算得到的mAP值
        """
        evaluation = self._detector.evaluate(list(predictions), list(ground_truths))
        details = evaluation.get(self.iou_threshold, {})
        return float(details.get("map", 0.0))

    def _compute_detection_errors(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> Tuple[int, int, int, int]:
        """使用贪婪IoU匹配计算漏检和误检.

        Args:
            predictions: 预测样本序列
            ground_truths: 真实标注样本序列

        Returns:
            Tuple[int, int, int, int]: 漏检数、误检数、真实目标总数、预测总数
        """

        misses = 0
        false_positives = 0
        total_gt = 0
        total_predictions = 0

        # 遍历每对预测和真实标注
        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.boxes
            pred_boxes = pred.boxes
            total_gt += gt_boxes.shape[0]
            total_predictions += pred_boxes.shape[0]
            # 使用贪婪IoU匹配算法匹配预测框和真实框
            matches = _greedy_iou_match(pred_boxes, gt_boxes, self.iou_threshold)
            # 计算漏检数（真实框数 - 匹配数）
            misses += gt_boxes.shape[0] - len(matches)
            # 计算误检数（预测框数 - 匹配数）
            false_positives += pred_boxes.shape[0] - len(matches)

        return misses, false_positives, total_gt, total_predictions

    def _to_samples(
        self,
        entries: Sequence[PredictionLike],
        sample_type: str,
    ) -> List[DetectionSample]:
        """将任意输入的预测/目标标准化为样本.

        Args:
            entries: 预测或真实标注条目序列
            sample_type: 样本类型，"prediction" 或 "ground_truth"

        Returns:
            List[DetectionSample]: 标准化后的检测样本列表

        Raises:
            ValueError: 当样本类型不支持时
        """

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
        """标准化指标选择
        
        Args:
            metrics_to_report: 需要报告的指标列表

        Returns:
            Optional[Mapping[str, None]]: 标准化后的指标字典
        """
        if metrics_to_report is None:
            return None
        filtered: Dict[str, None] = {}
        for name in metrics_to_report:
            if not isinstance(name, str):
                continue
            normalized = name.strip().lower()
            # 只保留支持的指标名称
            if normalized in {"map_drop_rate", "miss_rate", "false_detection_rate"}:
                filtered[normalized] = None
        return filtered or None

    @staticmethod
    def _collect_rotation_keys(
        *mappings: Mapping[RotationKey, Sequence[PredictionLike]]
    ) -> List[float]:
        """收集旋转角度键
        
        Args:
            mappings: 多个映射参数

        Returns:
            List[float]: 排序后的旋转角度列表
        """
        keys: List[float] = []
        seen: Dict[float, None] = {}
        # 遍历所有映射，收集旋转角度键
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
        """计算mAP下降率
        
        Args:
            baseline_map: 基线mAP值
            adversarial_map: 对抗攻击后的mAP值

        Returns:
            float: mAP下降率
        """
        if baseline_map <= 0:
            return 0.0
        # 计算下降率：(基线 - 对抗) / 基线
        drop = (baseline_map - adversarial_map) / baseline_map
        return float(max(0.0, drop))


def _greedy_iou_match(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float
) -> List[Tuple[int, int]]:
    """在预测框和真实框之间执行贪婪IoU匹配.

    Args:
        pred_boxes: 预测框数组，形状为(N, 4)
        gt_boxes: 真实框数组，形状为(M, 4)
        threshold: IoU阈值

    Returns:
        List[Tuple[int, int]]: 匹配的索引对列表，每个元组为(预测框索引, 真实框索引)
    """

    # 处理空数组情况
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return []

    # 计算所有预测框和真实框之间的IoU矩阵
    iou_matrix = _pairwise_iou(pred_boxes, gt_boxes)
    matches: List[Tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    # 贪婪匹配过程
    while True:
        # 找到所有未使用的预测框和真实框之间的IoU
        remaining = [
            (i, j, iou_matrix[i, j])
            for i in range(iou_matrix.shape[0])
            for j in range(iou_matrix.shape[1])
            if i not in used_pred and j not in used_gt
        ]
        if not remaining:
            break
        # 选择IoU最大的一对
        best = max(remaining, key=lambda item: item[2])
        i, j, iou = best
        # 如果IoU低于阈值，则停止匹配
        if iou < threshold:
            break
        # 记录匹配结果
        matches.append((i, j))
        used_pred.add(i)
        used_gt.add(j)

    return matches


def _pairwise_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """计算两组框之间的成对IoU.

    Args:
        pred_boxes: 预测框数组，形状为(N, 4)
        gt_boxes: 真实框数组，形状为(M, 4)

    Returns:
        np.ndarray: IoU矩阵，形状为(N, M)
    """

    # 处理空数组情况
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    # 数据类型转换
    pred = pred_boxes.astype(np.float32)
    gt = gt_boxes.astype(np.float32)

    # 重塑为(N, 4)和(M, 4)形状
    pred = pred.reshape(-1, 4)
    gt = gt.reshape(-1, 4)

    # 初始化IoU矩阵
    iou = np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float32)
    
    # 计算每对框之间的IoU
    for i, p in enumerate(pred):
        px1, py1, px2, py2 = p
        # 计算预测框面积
        p_area = max(px2 - px1, 0) * max(py2 - py1, 0)
        for j, g in enumerate(gt):
            gx1, gy1, gx2, gy2 = g
            # 计算真实框面积
            g_area = max(gx2 - gx1, 0) * max(gy2 - gy1, 0)
            # 计算交集区域坐标
            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            # 计算交集区域宽高
            inter_w = max(ix2 - ix1, 0)
            inter_h = max(iy2 - iy1, 0)
            # 计算交集面积
            intersection = inter_w * inter_h
            # 计算并集面积
            union = p_area + g_area - intersection
            # 计算IoU
            iou[i, j] = intersection / union if union > 0 else 0.0
    return iou