"""Detection basic metric wrapper with ResultSender integration."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from utils.SecAISender import ResultSender
from .detection import (
    DetectionSample,
    ObjectDetectionEvaluator,
    _derive_thresholds_from_metric_names,
    _ensure_list,
    _ensure_mapping,
    _normalize_metrics_request,
)

ArrayLike = Union[np.ndarray, Sequence[float]]
EvaluationResults = Dict[float, Dict[str, Union[float, Dict[str, float]]]]


def cal_basic(
    estimator,
    test_loader,
    metrics: Dict[str, Union[List[str], Dict]],
) -> None:
    """Run detection evaluation and report via ResultSender."""
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

        ResultSender.send_log("进度", "预测结果收集完成，开始计算检测指标")

        metric_names, config = _normalize_metrics_request(metrics)

        requested_thresholds = set()
        if isinstance(config, dict):
            requested_thresholds.update(
                float(th) for th in config.get("iou_thresholds", [])
            )
        derived_thresholds = _derive_thresholds_from_metric_names(metric_names)
        requested_thresholds.update(derived_thresholds)
        if not requested_thresholds:
            requested_thresholds.add(0.5)

        evaluator = ObjectDetectionEvaluator(requested_thresholds)
        evaluation_results = evaluator.evaluate(predictions, ground_truths)

        ResultSender.send_log("进度", "检测评测计算完成，正在写回结果")
        _dispatch_results(metric_names, config, evaluation_results)

        ResultSender.send_status("成功")
        ResultSender.send_log("进度", "目标检测评测结果已写回数据库")
    except Exception as exc:  # pylint: disable=broad-except
        ResultSender.send_status("失败")
        ResultSender.send_log("错误", str(exc))
        raise


def _dispatch_results(
    metric_names: Iterable[str],
    config: Dict,
    evaluation_results: EvaluationResults,
) -> None:
    if not metric_names:
        metric_names = ["map_50"]

    for name in metric_names:
        canonical = _canonical_metric_name(name)
        display_name = name if isinstance(name, str) else str(name)

        if canonical in ("map", "map_50"):
            _send_map_for_threshold(
                evaluation_results,
                0.5,
                display_name if display_name else canonical,
            )
        elif (
            isinstance(canonical, str)
            and canonical.startswith("map_")
            and canonical not in ("map_50", "map_5095")
        ):
            try:
                threshold = float(canonical.split("_")[1]) / 100.0
            except (IndexError, ValueError):
                continue
            key = display_name if display_name else f"map_{int(threshold * 100)}"
            _send_map_for_threshold(evaluation_results, threshold, key)
        elif canonical == "map_5095":
            values = [
                details["map"]
                for thr, details in evaluation_results.items()
                if 0.5 <= thr <= 0.95
            ]
            values = [value for value in values if not np.isnan(value)]
            if values:
                result_key = display_name if display_name else "map_5095"
                ResultSender.send_result(result_key, float(np.mean(values)))
        elif canonical == "per_class_ap":
            if 0.5 in evaluation_results:
                result_key = display_name if display_name else "per_class_ap_50"
                ResultSender.send_result(result_key, evaluation_results[0.5]["per_class"])
        elif canonical == "precision":
            threshold = (
                float(config.get("precision_iou_threshold", 0.5))
                if isinstance(config, dict)
                else 0.5
            )
            display_key = display_name or None
            _send_scalar_metric(evaluation_results, threshold, "precision", display_key)
        elif canonical == "recall":
            threshold = (
                float(config.get("recall_iou_threshold", 0.5))
                if isinstance(config, dict)
                else 0.5
            )
            display_key = display_name or None
            _send_scalar_metric(evaluation_results, threshold, "recall", display_key)


def _canonical_metric_name(name: Any) -> Any:
    from .detection import _canonical_metric_name as _canon  # type: ignore

    return _canon(name)


def _send_map_for_threshold(
    evaluation_results: EvaluationResults,
    threshold: float,
    key: str,
) -> None:
    if threshold in evaluation_results:
        value = evaluation_results[threshold]["map"]
        if not np.isnan(value):
            ResultSender.send_result(key, float(value))


def _send_scalar_metric(
    evaluation_results: EvaluationResults,
    threshold: float,
    metric_key: str,
    display_name: Union[str, None] = None,
) -> None:
    if threshold in evaluation_results:
        value = evaluation_results[threshold].get(metric_key)
        if value is not None and not np.isnan(value):
            label = display_name or f"{metric_key}_{int(threshold * 100)}"
            ResultSender.send_result(label, float(value))
