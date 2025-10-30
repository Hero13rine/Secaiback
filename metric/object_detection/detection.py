"""Evaluation utilities for object detection models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from utils.SecAISender import ResultSender


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class DetectionSample:
    """Normalized representation of detection data for a single image."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_prediction(cls, prediction: Dict[str, ArrayLike]) -> "DetectionSample":
        boxes = _to_numpy(prediction.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        scores = _to_numpy(prediction.get("scores", []), dtype=np.float32).reshape(-1)
        labels = _to_numpy(prediction.get("labels", []), dtype=np.int64).reshape(-1)

        # If the model did not output confidence scores, default to 1.0 for all detections.
        if scores.size == 0 and boxes.size > 0:
            scores = np.ones((boxes.shape[0],), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels.size == 0 and boxes.size > 0:
            labels = np.zeros((boxes.shape[0],), dtype=np.int64)
        return cls(boxes=boxes, scores=scores, labels=labels)

    @classmethod
    def from_ground_truth(cls, target: Dict[str, ArrayLike]) -> "DetectionSample":
        boxes = _to_numpy(target.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        labels = _to_numpy(target.get("labels", []), dtype=np.int64).reshape(-1)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels.size == 0 and boxes.size > 0:
            labels = np.zeros((boxes.shape[0],), dtype=np.int64)
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
        return cls(boxes=boxes, scores=scores, labels=labels)


class ObjectDetectionEvaluator:
    """Encapsulates evaluation logic for object detection models."""

    def __init__(self, iou_thresholds: Iterable[float]):
        self.iou_thresholds = sorted(set(float(th) for th in iou_thresholds))

    def evaluate(
        self, predictions: List[DetectionSample], ground_truths: List[DetectionSample]
    ) -> Dict[float, Dict[str, Union[float, Dict[str, float]]]]:
        if len(predictions) != len(ground_truths):
            raise ValueError("Prediction and ground truth counts do not match")
        results: Dict[float, Dict[str, Union[float, Dict[str, float]]]] = {}
        classes = _collect_all_labels(ground_truths, predictions)
        for threshold in self.iou_thresholds:
            per_class_ap: Dict[str, float] = {}
            ap_values: List[float] = []
            tp_total = 0.0
            fp_total = 0.0
            gt_total = 0.0
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

            map_value = float(np.mean(ap_values)) if ap_values else 0.0
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
    ap: float
    tp: float
    fp: float
    npos: int


def cal_object_detection(estimator, test_loader, metrics: Dict[str, Union[List[str], Dict]]):
    """Entrypoint used by the evaluation pipeline for object detection tasks."""
    try:
        ResultSender.send_log("进度", "开始收集检测模型预测结果")
        predictions: List[DetectionSample] = []
        ground_truths: List[DetectionSample] = []

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

        metrics = metrics or {}
        metric_names = metrics.get("performance_testing", []) if isinstance(metrics, dict) else metrics
        if metric_names is None:
            metric_names = []
        config = metrics.get("performance_testing_config", {}) if isinstance(metrics, dict) else {}

        requested_thresholds = set(float(th) for th in config.get("iou_thresholds", []))
        derived_thresholds = _derive_thresholds_from_metric_names(metric_names)
        requested_thresholds.update(derived_thresholds)
        if not requested_thresholds:
            requested_thresholds.add(0.5)

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
    if not metric_names:
        metric_names = ["map_50"]

    for name in metric_names:
        if name in ("map", "map_50"):
            _send_map_for_threshold(evaluation_results, 0.5, "map_50")
        elif name.startswith("map_") and name not in ("map", "map_50", "map_5095"):
            try:
                threshold = float(name.split("_")[1]) / 100.0
            except (IndexError, ValueError):
                continue
            _send_map_for_threshold(evaluation_results, threshold, f"map_{int(threshold * 100)}")
        elif name == "map_5095":
            values = [details["map"] for thr, details in evaluation_results.items() if 0.5 <= thr <= 0.95]
            values = [value for value in values if not np.isnan(value)]
            if values:
                ResultSender.send_result("map_5095", float(np.mean(values)))
        elif name == "per_class_ap":
            if 0.5 in evaluation_results:
                ResultSender.send_result("per_class_ap_50", evaluation_results[0.5]["per_class"])
        elif name == "precision":
            threshold = float(config.get("precision_iou_threshold", 0.5))
            _send_scalar_metric(evaluation_results, threshold, "precision")
        elif name == "recall":
            threshold = float(config.get("recall_iou_threshold", 0.5))
            _send_scalar_metric(evaluation_results, threshold, "recall")


def _send_map_for_threshold(evaluation_results: Dict[float, Dict[str, Union[float, Dict[str, float]]]], threshold: float, key: str):
    if threshold in evaluation_results:
        value = evaluation_results[threshold]["map"]
        if not np.isnan(value):
            ResultSender.send_result(key, float(value))


def _send_scalar_metric(evaluation_results: Dict[float, Dict[str, Union[float, Dict[str, float]]]], threshold: float, metric_key: str):
    if threshold in evaluation_results:
        value = evaluation_results[threshold].get(metric_key)
        if value is not None and not np.isnan(value):
            ResultSender.send_result(f"{metric_key}_{int(threshold * 100)}", float(value))


def _derive_thresholds_from_metric_names(metric_names: Iterable[str]) -> set:
    thresholds = set()
    for name in metric_names or []:
        if name in ("map", "map_50", "precision", "recall", "per_class_ap"):
            thresholds.add(0.5)
        elif name.startswith("map_") and name != "map_5095":
            try:
                threshold = float(name.split("_")[1]) / 100.0
                thresholds.add(threshold)
            except (IndexError, ValueError):
                continue
        elif name == "map_5095":
            thresholds.update(np.round(np.arange(0.5, 1.0, 0.05), 2))
    return thresholds


def _collect_all_labels(ground_truths: List[DetectionSample], predictions: List[DetectionSample]) -> List[int]:
    labels = set()
    for sample in ground_truths + predictions:
        labels.update(sample.labels.tolist())
    labels.discard(-1)
    return sorted(labels)


def _evaluate_single_class(
    predictions: List[DetectionSample],
    ground_truths: List[DetectionSample],
    class_id: int,
    iou_threshold: float,
) -> _ClassEvaluationStats:
    detections: List[Tuple[int, float, np.ndarray]] = []
    ground_truth_map: Dict[int, Dict[str, np.ndarray]] = {}
    for image_idx, (prediction, target) in enumerate(zip(predictions, ground_truths)):
        pred_mask = prediction.labels == class_id
        gt_mask = target.labels == class_id
        pred_boxes = prediction.boxes[pred_mask]
        pred_scores = prediction.scores[pred_mask]
        gt_boxes = target.boxes[gt_mask]
        ground_truth_map[image_idx] = {
            "boxes": gt_boxes,
            "matched": np.zeros(len(gt_boxes), dtype=bool),
        }
        for box, score in zip(pred_boxes, pred_scores):
            detections.append((image_idx, float(score), box))

    detections.sort(key=lambda item: item[1], reverse=True)
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    npos = sum(len(info["boxes"]) for info in ground_truth_map.values())

    for idx, (image_idx, _score, pred_box) in enumerate(detections):
        gt_info = ground_truth_map[image_idx]
        gt_boxes = gt_info["boxes"]
        if gt_boxes.size == 0:
            fp[idx] = 1
            continue
        ious = _compute_iou(pred_box, gt_boxes)
        best_match = int(np.argmax(ious))
        best_iou = ious[best_match]
        if best_iou >= iou_threshold and not gt_info["matched"][best_match]:
            tp[idx] = 1
            gt_info["matched"][best_match] = True
        else:
            fp[idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    recall = tp_cumsum / max(npos, np.finfo(np.float64).eps)
    ap = _compute_average_precision(recall, precision) if npos > 0 else float("nan")
    tp_total = float(tp_cumsum[-1]) if tp_cumsum.size > 0 else 0.0
    fp_total = float(fp_cumsum[-1]) if fp_cumsum.size > 0 else 0.0
    return _ClassEvaluationStats(ap=ap, tp=tp_total, fp=fp_total, npos=npos)


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    box = box.astype(np.float32)
    boxes = boxes.astype(np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    box_area = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
    boxes_area = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) * np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
    union = box_area + boxes_area - intersection
    union = np.maximum(union, np.finfo(np.float32).eps)
    return intersection / union


def _compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0 or precision.size == 0:
        return float("nan")
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _to_numpy(array: ArrayLike, dtype=None) -> np.ndarray:
    if array is None:
        result = np.empty((0,), dtype=dtype or np.float32)
    elif isinstance(array, np.ndarray):
        result = array
    elif hasattr(array, "detach"):
        result = array.detach().cpu().numpy()
    elif hasattr(array, "cpu") and hasattr(array.cpu(), "numpy"):
        result = array.cpu().numpy()
    else:
        result = np.array(array)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _ensure_list(value) -> List:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ensure_mapping(value) -> Dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "_asdict"):
        return value._asdict()
    raise TypeError("目标检测评测需要字典格式的预测输出和标签")
