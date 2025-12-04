"""Evaluation utilities for object detection models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np


from utils.SecAISender import  ResultSender
# from utils.sender import ConsoleResultSender as ResultSender # 本地调试时使用



ArrayLike = Union[np.ndarray, Sequence[float]]

_METRIC_ALIAS_MAP = {
    "accuracy": "map_50",
    "map50": "map_50",
    "map@50": "map_50",
}


def _canonical_metric_name(name: Any) -> Any:
    """Normalize a metric name so aliases share the same handling."""

    if not isinstance(name, str):
        return name
    normalized = name.strip()
    if not normalized:
        return normalized
    alias = _METRIC_ALIAS_MAP.get(normalized.lower())
    return alias if alias is not None else normalized.lower()


def _normalize_metrics_request(metrics) -> Tuple[List[str], Dict]:
    """Extract metric names and auxiliary config from user configuration."""

    def _extract_names(candidate: Any) -> List[str]:
        if not candidate:
            return []
        if isinstance(candidate, str):
            return [candidate]
        if isinstance(candidate, dict):
            inner = candidate.get("metrics")
            if inner:
                return _extract_names(inner)
            return []
        return list(candidate)

    if metrics is None:
        return [], {}

    config: Dict = {}
    names: List[str] = []

    if isinstance(metrics, dict):
        performance_section = metrics.get("performance_testing")
        if isinstance(performance_section, dict):
            names = _extract_names(performance_section.get("metrics"))
            config = (
                performance_section.get("performance_testing_config")
                or performance_section.get("config")
                or metrics.get("performance_testing_config")
                or metrics.get("config")
                or {}
            )
        else:
            candidate_lists = [
                metrics.get("performance_testing"),
                metrics.get("basic"),
                metrics.get("metrics"),
            ]
            for candidate in candidate_lists:
                names = _extract_names(candidate)
                if names:
                    break
            config = (
                metrics.get("performance_testing_config")
                or metrics.get("config")
                or {}
            )
    elif isinstance(metrics, (list, tuple, set)):
        names = list(metrics)
    else:
        names = [metrics]

    return names, dict(config) if isinstance(config, dict) else {}


@dataclass
class DetectionSample:
    """标准化的目标检测数据表示，用于单张图像."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_prediction(cls, prediction: Dict[str, ArrayLike]) -> "DetectionSample":
        """
        从模型预测结果创建 DetectionSample 实例.

        Args:
            prediction: 包含预测框、得分和标签的字典

        Returns:
            DetectionSample 实例
        """
        # 提取边界框、得分和标签，并确保正确的形状
        boxes = _to_numpy(prediction.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        scores = _to_numpy(prediction.get("scores", []), dtype=np.float32).reshape(-1)
        labels = _to_numpy(prediction.get("labels", []), dtype=np.int64).reshape(-1)

        # 如果模型没有输出置信度得分，则为所有检测结果默认设置得分为1.0
        if scores.size == 0 and boxes.size > 0:
            scores = np.ones((boxes.shape[0],), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels.size == 0 and boxes.size > 0:
            labels = np.zeros((boxes.shape[0],), dtype=np.int64)
        return cls(boxes=boxes, scores=scores, labels=labels)

    @classmethod
    def from_ground_truth(cls, target: Dict[str, ArrayLike]) -> "DetectionSample":
        """
        从真实标签创建 DetectionSample 实例.

        Args:
            target: 包含真实边界框和标签的字典

        Returns:
            DetectionSample 实例
        """
        # 提取真实边界框和标签，并确保正确的形状
        boxes = _to_numpy(target.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        labels = _to_numpy(target.get("labels", []), dtype=np.int64).reshape(-1)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels.size == 0 and boxes.size > 0:
            labels = np.zeros((boxes.shape[0],), dtype=np.int64)
        # 真实标签的置信度默认为1.0
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
        return cls(boxes=boxes, scores=scores, labels=labels)


class ObjectDetectionEvaluator:
    """封装目标检测模型的评估逻辑."""

    def __init__(self, iou_thresholds: Iterable[float]):
        """
        初始化评估器.

        Args:
            iou_thresholds: 用于评估的 IoU 阈值列表
        """
        self.iou_thresholds = sorted(set(float(th) for th in iou_thresholds))

    def evaluate(
        self, predictions: List[DetectionSample], ground_truths: List[DetectionSample]
    ) -> Dict[float, Dict[str, Union[float, Dict[str, float]]]]:
        """
        执行目标检测评估.

        Args:
            predictions: 模型预测结果列表
            ground_truths: 真实标签列表

        Returns:
            包含不同 IoU 阈值下评估结果的字典

        Raises:
            ValueError: 当预测结果和真实标签数量不匹配时抛出
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Prediction and ground truth counts do not match")
        results: Dict[float, Dict[str, Union[float, Dict[str, float]]]] = {}
        # 收集所有类别标签
        classes = _collect_all_labels(ground_truths, predictions)
        # 对每个 IoU 阈值进行评估
        for threshold in self.iou_thresholds:
            # 存储每个类别的平均精度
            per_class_ap: Dict[str, float] = {}
            ap_values: List[float] = []
            tp_total = 0.0
            fp_total = 0.0
            gt_total = 0.0
            # 对每个类别进行评估
            for class_id in classes:
                stats = _evaluate_single_class(predictions, ground_truths, class_id, threshold)
                ap, tp, fp, npos = stats.ap, stats.tp, stats.fp, stats.npos
                if not np.isnan(ap):
                    per_class_ap[str(class_id)] = float(ap)
                    ap_values.append(float(ap))
                else:
                    per_class_ap[str(class_id)] = 0.0
                tp_total += tp
                fp_total += fp
                gt_total += npos

            # 计算总体 mAP 值
            map_value = float(np.mean(ap_values)) if ap_values else 0.0
            # 计算总体精度和召回率
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall = tp_total / gt_total if gt_total > 0 else 0.0
            results[threshold] = {
                "map": map_value,
                "per_class": per_class_ap,
                "precision": precision,
                "recall": recall,
            }
        return results


@dataclass
class _ClassEvaluationStats:
    """用于存储单个类别的评估统计信息."""
    ap: float
    tp: float
    fp: float
    npos: int


def cal_basic(estimator, test_loader, metrics: Dict[str, Union[List[str], Dict]]):
    """
    目标检测任务的评估入口点.

    Args:
        estimator: 评估的模型估计器
        test_loader: 测试数据加载器
        metrics: 评估指标配置
    """
    try:

        ResultSender.send_log("进度", "开始收集检测模型预测结果")
        predictions: List[DetectionSample] = []
        ground_truths: List[DetectionSample] = []

        # 遍历测试数据集，收集预测结果和真实标签
        for images, targets in test_loader:
            outputs = estimator.predict(images)
            normalized_outputs = _ensure_list(outputs)
            normalized_targets = _ensure_list(targets)
            if len(normalized_outputs) != len(normalized_targets):
                raise ValueError("预测输出与标注数量不匹配")
            for pred, target in zip(normalized_outputs, normalized_targets):
                predictions.append(DetectionSample.from_prediction(_ensure_mapping(pred)))
                ground_truths.append(DetectionSample.from_ground_truth(_ensure_mapping(target)))

        if not predictions:
            raise ValueError("测试集为空，无法进行评测")

        # 解析评估指标配置
        metric_names, config = _normalize_metrics_request(metrics)

        # 确定需要评估的 IoU 阈值
        requested_thresholds = set()
        if isinstance(config, dict):
            requested_thresholds.update(
                float(th) for th in config.get("iou_thresholds", [])
            )
        derived_thresholds = _derive_thresholds_from_metric_names(metric_names)
        requested_thresholds.update(derived_thresholds)
        if not requested_thresholds:
            requested_thresholds.add(0.5)

        # 执行评估
        evaluator = ObjectDetectionEvaluator(requested_thresholds)
        evaluation_results = evaluator.evaluate(predictions, ground_truths)


        ResultSender.send_log("进度", "评测计算完成，开始汇总结果")
        _dispatch_results(metric_names, config, evaluation_results)



        ResultSender.send_status("成功")

        ResultSender.send_log("进度", "目标检测评测结果已写回数据库")
    except Exception as exc:  # pylint: disable=broad-except

        ResultSender.send_status("失败")
        ResultSender.send_log("错误", str(exc))


def _dispatch_results(metric_names: Iterable[str], config: Dict, evaluation_results: Dict[float, Dict[str, Union[float, Dict[str, float]]]]):
    """
    分发评估结果到结果发送器.

    Args:
        metric_names: 需要发送的指标名称列表
        config: 评估配置
        evaluation_results: 评估结果
    """
    if not metric_names:
        metric_names = ["map_50"]

    for name in metric_names:
        canonical = _canonical_metric_name(name)
        display_name = name if isinstance(name, str) else str(name)

        if canonical in ("map", "map_50"):
            # 发送 IoU 阈值为 0.5 时的 mAP
            _send_map_for_threshold(
                evaluation_results,
                0.5,
                display_name if display_name else canonical,
            )
        elif isinstance(canonical, str) and canonical.startswith("map_") and canonical not in ("map_50", "map_5095"):
            # 发送指定 IoU 阈值的 mAP
            try:
                threshold = float(canonical.split("_")[1]) / 100.0
            except (IndexError, ValueError):
                continue
            key = display_name if display_name else f"map_{int(threshold * 100)}"
            _send_map_for_threshold(evaluation_results, threshold, key)
        elif canonical == "map_5095":
            # 计算 IoU 阈值从 0.5 到 0.95（步长 0.05）的平均 mAP
            values = [details["map"] for thr, details in evaluation_results.items() if 0.5 <= thr <= 0.95]
            values = [value for value in values if not np.isnan(value)]
            if values:

                result_key = display_name if display_name else "map_5095"

                ResultSender.send_result("map_5095", float(np.mean(values)))
        elif canonical == "per_class_ap":
            # 发送每个类别的 AP 值
            if 0.5 in evaluation_results:
                result_key = display_name if display_name else "per_class_ap_50"

                ResultSender.send_result("per_class_ap_50", evaluation_results[0.5]["per_class"])
        elif canonical == "precision":
            # 发送指定 IoU 阈值的精度
            threshold = float(config.get("precision_iou_threshold", 0.5)) if isinstance(config, dict) else 0.5
            display_key = display_name or None
            _send_scalar_metric(evaluation_results, threshold, "precision", display_key)
        elif canonical == "recall":
            # 发送指定 IoU 阈值的召回率
            threshold = float(config.get("recall_iou_threshold", 0.5)) if isinstance(config, dict) else 0.5
            display_key = display_name or None
            _send_scalar_metric(evaluation_results, threshold, "recall", display_key)


def _send_map_for_threshold(evaluation_results: Dict[float, Dict[str, Union[float, Dict[str, float]]]], threshold: float, key: str):
    """
    发送指定阈值的 mAP 结果.

    Args:
        evaluation_results: 评估结果
        threshold: IoU 阈值
        key: 结果键名
    """
    if threshold in evaluation_results:
        value = evaluation_results[threshold]["map"]
        if not np.isnan(value):


            ResultSender.send_result(key, float(value))


def _send_scalar_metric(
    evaluation_results: Dict[float, Dict[str, Union[float, Dict[str, float]]]],
    threshold: float,
    metric_key: str,
    display_name: Union[str, None] = None,
):
    """
    发送标量指标结果.

    Args:
        evaluation_results: 评估结果
        threshold: IoU 阈值
        metric_key: 指标键名
    """
    if threshold in evaluation_results:
        value = evaluation_results[threshold].get(metric_key)
        if value is not None and not np.isnan(value):

            label = display_name or f"{metric_key}_{int(threshold * 100)}"

            ResultSender.send_result(label, float(value))


def _derive_thresholds_from_metric_names(metric_names: Iterable[str]) -> set:
    """
    从指标名称中推导出需要的 IoU 阈值.

    Args:
        metric_names: 指标名称列表

    Returns:
        需要的 IoU 阈值集合
    """
    thresholds = set()
    for name in metric_names or []:
        canonical = _canonical_metric_name(name)
        if canonical in ("map", "map_50", "precision", "recall", "per_class_ap"):
            thresholds.add(0.5)
        elif isinstance(canonical, str) and canonical.startswith("map_") and canonical != "map_5095":
            try:
                threshold = float(canonical.split("_")[1]) / 100.0
                thresholds.add(threshold)
            except (IndexError, ValueError):
                continue
        elif canonical == "map_5095":
            # 添加从 0.5 到 0.95，步长为 0.05 的阈值
            thresholds.update(np.round(np.arange(0.5, 1.0, 0.05), 2))
    return thresholds


def _collect_all_labels(ground_truths: List[DetectionSample], predictions: List[DetectionSample]) -> List[int]:
    """
    收集所有样本中的类别标签.
    
    Args:
        ground_truths: 真实标签列表
        predictions: 预测结果列表
        
    Returns:
        所有类别标签的排序列表
    """
    labels = set()
    for sample in ground_truths + predictions:
        labels.update(sample.labels.tolist())
    labels.discard(-1)  # 移除背景类（如果存在）
    return sorted(labels)


def _evaluate_single_class(
    predictions: List[DetectionSample],
    ground_truths: List[DetectionSample],
    class_id: int,
    iou_threshold: float,
) -> _ClassEvaluationStats:
    """
    评估单个类别的性能.
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
        class_id: 类别ID
        iou_threshold: IoU 阈值
        
    Returns:
        该类别的评估统计信息
    """
    # 收集所有检测结果
    detections: List[Tuple[int, float, np.ndarray]] = []
    # 构建真实标签映射，跟踪匹配状态
    ground_truth_map: Dict[int, Dict[str, np.ndarray]] = {}
    for image_idx, (prediction, target) in enumerate(zip(predictions, ground_truths)):
        # 过滤出当前类别的预测框和真实框
        pred_mask = prediction.labels == class_id
        gt_mask = target.labels == class_id
        pred_boxes = prediction.boxes[pred_mask]
        pred_scores = prediction.scores[pred_mask]
        gt_boxes = target.boxes[gt_mask]
        # 初始化真实框的匹配状态
        ground_truth_map[image_idx] = {
            "boxes": gt_boxes,
            "matched": np.zeros(len(gt_boxes), dtype=bool),
        }
        # 将当前图像的检测结果添加到总列表中
        for box, score in zip(pred_boxes, pred_scores):
            detections.append((image_idx, float(score), box))

    # 按得分降序排列检测结果
    detections.sort(key=lambda item: item[1], reverse=True)
    # 初始化真正例和假正例数组
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    # 计算正例总数
    npos = sum(len(info["boxes"]) for info in ground_truth_map.values())

    # 对每个检测结果进行匹配判断
    for idx, (image_idx, _score, pred_box) in enumerate(detections):
        gt_info = ground_truth_map[image_idx]
        gt_boxes = gt_info["boxes"]
        # 如果当前图像没有该类别的真实框，则为假正例
        if gt_boxes.size == 0:
            fp[idx] = 1
            continue
        # 计算与所有真实框的 IoU
        ious = _compute_iou(pred_box, gt_boxes)
        # 找到 IoU 最大的真实框
        best_match = int(np.argmax(ious))
        best_iou = ious[best_match]
        # 如果 IoU 超过阈值且该真实框未被匹配，则为真正例
        if best_iou >= iou_threshold and not gt_info["matched"][best_match]:
            tp[idx] = 1
            gt_info["matched"][best_match] = True
        else:
            # 否则为假正例
            fp[idx] = 1

    # 计算累积的真正例和假正例
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    # 计算精度和召回率
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    recall = tp_cumsum / max(npos, np.finfo(np.float64).eps)
    # 计算平均精度
    ap = _compute_average_precision(recall, precision) if npos > 0 else float("nan")
    tp_total = float(tp_cumsum[-1]) if tp_cumsum.size > 0 else 0.0
    fp_total = float(fp_cumsum[-1]) if fp_cumsum.size > 0 else 0.0
    return _ClassEvaluationStats(ap=ap, tp=tp_total, fp=fp_total, npos=npos)


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    计算单个边界框与多个边界框之间的 IoU（交并比）.
    
    Args:
        box: 单个边界框 [x1, y1, x2, y2]
        boxes: 多个边界框数组，形状为 (N, 4)
        
    Returns:
        IoU 值数组
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    box = box.astype(np.float32)
    boxes = boxes.astype(np.float32)

    # 计算交集的坐标
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # 计算交集的宽和高
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    # 计算交集面积
    intersection = inter_w * inter_h

    # 计算两个框的面积
    box_area = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
    boxes_area = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) * np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
    # 计算并集面积
    union = box_area + boxes_area - intersection
    union = np.maximum(union, np.finfo(np.float32).eps)
    # 返回 IoU 值
    return intersection / union


def _compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    根据精度-召回率曲线计算平均精度（AP）.
    
    Args:
        recall: 召回率数组
        precision: 精度数组
        
    Returns:
        平均精度值
    """
    if recall.size == 0 or precision.size == 0:
        return float("nan")
    # 在曲线的开始和结束添加点，确保从0开始，到0结束
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # 对精度进行平滑处理，确保精度曲线单调递减
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # 找到召回率变化的点
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    # 计算平均精度
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _to_numpy(array: ArrayLike, dtype=None) -> np.ndarray:
    """
    将输入转换为 NumPy 数组.
    
    Args:
        array: 输入数组
        dtype: 目标数据类型
        
    Returns:
        NumPy 数组
    """
    if array is None:
        result = np.empty((0,), dtype=dtype or np.float32)
    elif isinstance(array, np.ndarray):
        result = array
    elif hasattr(array, "detach"):
        # 处理 PyTorch 张量
        result = array.detach().cpu().numpy()
    elif hasattr(array, "cpu") and hasattr(array.cpu(), "numpy"):
        # 处理其他具有 numpy 方法的张量
        result = array.cpu().numpy()
    else:
        result = np.array(array)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _ensure_list(value) -> List:
    """
    确保值为列表类型.
    
    Args:
        value: 输入值
        
    Returns:
        列表
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ensure_mapping(value) -> Dict:
    """
    确保值为字典类型.
    
    Args:
        value: 输入值
        
    Returns:
        字典
        
    Raises:
        TypeError: 当值无法转换为字典时抛出
    """
    if isinstance(value, dict):
        return value
    if hasattr(value, "_asdict"):
        return value._asdict()
    raise TypeError("目标检测评测需要字典格式的预测输出和标签")