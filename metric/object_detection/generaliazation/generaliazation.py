"""
Cross-dataset generalization evaluation for object detection models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from metric.object_detection.basic.detection import (
    DetectionSample,
    ObjectDetectionEvaluator,
)
from utils.sender import ConsoleResultSender as ResultSender
# from utils.SecAISender import ResultSender

@dataclass
class DatasetEvaluationResult:
    name: str
    metrics: Dict[float, Dict[str, Any]]
    num_samples: int

    def as_serializable(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "num_samples": self.num_samples,
            "metrics": {},
        }
        for threshold, stats in self.metrics.items():
            payload["metrics"][f"{threshold:.2f}"] = {
                "map": float(stats.get("map", 0.0)),
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "per_class": {
                    str(label): float(value)
                    for label, value in stats.get("per_class", {}).items()
                },
            }
        return payload


def _ensure_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ensure_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    if hasattr(sample, "_asdict"):
        return sample._asdict()
    raise TypeError("预测输出与标注需为 dict 结构，无法转换当前条目")


def _split_batch(batch: Any) -> Tuple[Any, Any]:
    """
    兼容 (images, targets, *extra) 或 {"images":..., "targets":...}。
    """
    if isinstance(batch, Mapping):
        images = batch.get("images")
        targets = batch.get("targets")
        if images is None or targets is None:
            raise ValueError("字典 batch 需包含 'images' 与 'targets' 键")
        return images, targets
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"无法解析 batch: {type(batch)}")


def _collect_samples(
    estimator,
    loader: Iterable,
    max_samples: int | None = None,
) -> Tuple[List[DetectionSample], List[DetectionSample]]:
    predictions: List[DetectionSample] = []
    ground_truths: List[DetectionSample] = []
    for batch in loader:
        images, targets = _split_batch(batch)
        outputs = estimator.predict(images)
        output_list = _ensure_list(outputs)
        target_list = _ensure_list(targets)
        if len(output_list) != len(target_list):
            raise ValueError("预测数量与标注数量不匹配")

        for pred_item, target_item in zip(output_list, target_list):
            predictions.append(
                DetectionSample.from_prediction(_ensure_mapping(pred_item))
            )
            ground_truths.append(
                DetectionSample.from_ground_truth(_ensure_mapping(target_item))
            )
            if max_samples and len(predictions) >= max_samples:
                return predictions, ground_truths
    if not predictions:
        raise ValueError("数据集为空，无法评估泛化能力")
    return predictions, ground_truths


def _evaluate_single_dataset(
    estimator,
    loader: Iterable,
    iou_thresholds: Sequence[float],
    dataset_name: str,
    max_samples: int | None = None,
) -> DatasetEvaluationResult:
    preds, gts = _collect_samples(estimator, loader, max_samples=max_samples)
    evaluator = ObjectDetectionEvaluator(iou_thresholds)
    metrics = evaluator.evaluate(preds, gts)
    return DatasetEvaluationResult(
        name=dataset_name,
        metrics=metrics,
        num_samples=len(preds),
    )


def _build_gap_summary(
    dataset_results: Dict[str, DatasetEvaluationResult],
    gap_pairs: List[Dict[str, str]],
    metric_key: str,
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for pair in gap_pairs:
        source_name = pair["source"]
        target_name = pair["target"]
        label = pair.get("label") or f"{source_name}_to_{target_name}"
        if source_name not in dataset_results or target_name not in dataset_results:
            ResultSender.send_log(
                "警告", f"跳过未找到的数据集对: {source_name}->{target_name}"
            )
            continue
        source_metrics = dataset_results[source_name].metrics
        target_metrics = dataset_results[target_name].metrics

        for threshold in source_metrics.keys():
            src_value = float(source_metrics[threshold].get(metric_key, 0.0))
            tgt_value = float(target_metrics.get(threshold, {}).get(metric_key, 0.0))
            retention = 0.0 if src_value == 0 else tgt_value / src_value
            drop_ratio = 1.0 - retention
            src_per_class = {
                str(k): float(v)
                for k, v in source_metrics[threshold].get("per_class", {}).items()
            }
            tgt_per_class = {
                str(k): float(v)
                for k, v in target_metrics.get(threshold, {}).get("per_class", {}).items()
            }
            summary.append(
                {
                    "pair": label,
                    "threshold": threshold,
                    "source_value": src_value,
                    "target_value": tgt_value,
                    "drop_ratio": drop_ratio,
                    "source_per_class": src_per_class,
                    "target_per_class": tgt_per_class,
                }
            )
    return summary


def _normalize_config(config: Any) -> Dict[str, Any]:
    """
    容错配置：非映射类型统一回退默认。
    """
    if isinstance(config, Mapping):
        return dict(config)
    ResultSender.send_log("提示", "未提供有效泛化配置，使用默认参数 (IoU=0.5)")
    return {}


def evaluate_cross_dataset_generalization(
    estimator,
    dataset_loaders: MutableMapping[str, Iterable],
    generalization_config: Any,
):
    """
    执行跨数据集泛化评估。
    """
    try:
        generalization_config = _normalize_config(generalization_config)
        ResultSender.send_log("进度", "开始跨数据集泛化评估")
        # 配置只需要 gap_pairs；其他参数在代码内写死
        iou_thresholds = [0.5]
        max_samples = None
        metric_key = "map"

        dataset_results: Dict[str, DatasetEvaluationResult] = {}
        for dataset_name, loader in dataset_loaders.items():
            ResultSender.send_log("进度", f"评估数据集: {dataset_name}")
            dataset_results[dataset_name] = _evaluate_single_dataset(
                estimator,
                loader,
                iou_thresholds,
                dataset_name,
                max_samples=max_samples,
            )

        if not dataset_results:
            raise ValueError("未提供任何数据集加载器")

        gap_pairs = generalization_config.get("gap_pairs", [])
        if not gap_pairs:
            # 默认使用前两个数据集构造一个 gap 对
            loader_names = list(dataset_loaders.keys())
            if len(loader_names) >= 2:
                gap_pairs = [
                    {
                        "source": loader_names[0],
                        "target": loader_names[1],
                        "label": "S->T",
                    }
                ]
        gap_summary = (
            _build_gap_summary(dataset_results, gap_pairs, metric_key)
            if gap_pairs
            else []
        )
        pair_metrics = [
            {
                "pair": item["pair"],
                "threshold": item["threshold"],
                "source_map": item["source_value"],
                "target_map": item["target_value"],
                "drop_ratio": item["drop_ratio"],
                "source_per_class": item.get("source_per_class", {}),
                "target_per_class": item.get("target_per_class", {}),
            }
            for item in gap_summary
        ]
        # 按 pair/阈值将关键指标拆分逐项返回，便于直接查看
        for item in pair_metrics:
            threshold = item.get("threshold", 0.0)
            th_suffix = f"{int(float(threshold) * 100)}"
            ResultSender.send_result(f"source_map_{th_suffix}", item.get("source_map", 0.0))
            ResultSender.send_result(f"target_map_{th_suffix}", item.get("target_map", 0.0))
            ResultSender.send_result(f"drop_ratio_{th_suffix}", item.get("drop_ratio", 0.0))
            ResultSender.send_result(
                f"source_per_class_{th_suffix}", item.get("source_per_class", {})
            )
            ResultSender.send_result(
                f"target_per_class_{th_suffix}", item.get("target_per_class", {})
            )
        ResultSender.send_log("信息", "跨数据集泛化评估完成")
        ResultSender.send_status("成功")
    except Exception as exc:  # pylint: disable=broad-except
        ResultSender.send_log("错误", str(exc))
        ResultSender.send_status("失败")
