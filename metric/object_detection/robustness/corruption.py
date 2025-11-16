"""自然扰动鲁棒性评测相关方法."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF

from metric.object_detection.basic.detection import DetectionSample, ObjectDetectionEvaluator
from metric.object_detection.robustness.matching import compute_detection_errors

PredictionLike = Mapping[str, Sequence[float]]
GroundTruthLike = Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class CorruptionRobustnessMetrics:
    """自然扰动鲁棒性指标容器."""

    perturbation_magnitude: float
    performance_drop_rate: float
    perturbation_tolerance: float


@dataclass(frozen=True)
class CorruptionEvaluationResult:
    """描述单种扰动的评估结果."""

    corruption_name: str
    severity: int
    metrics: CorruptionRobustnessMetrics


class CorruptionRobustnessEvaluator:
    """负责计算自然扰动后的鲁棒性指标."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)
        self._detector = ObjectDetectionEvaluator([self.iou_threshold])

    def evaluate_corruption(
        self,
        corruption_name: str,
        severity: int,
        baseline_predictions: Sequence[PredictionLike],
        corrupted_predictions: Sequence[PredictionLike],
        ground_truths: Sequence[GroundTruthLike],
        original_images: Sequence[torch.Tensor],
        corrupted_images: Sequence[torch.Tensor],
        metrics_to_report: Optional[Iterable[str]] = None,
    ) -> CorruptionEvaluationResult:
        metric_filter = self._normalize_metric_selection(metrics_to_report)

        base_samples = self._to_samples(baseline_predictions, sample_type="prediction")
        corr_samples = self._to_samples(corrupted_predictions, sample_type="prediction")
        gt_samples = self._to_samples(ground_truths, sample_type="ground_truth")

        if not gt_samples:
            raise ValueError("必须提供真实标注数据用于评测扰动鲁棒性")
        if len(base_samples) != len(gt_samples):
            raise ValueError("基线预测数量与真实标注数量不匹配")
        if len(corr_samples) != len(gt_samples):
            raise ValueError("扰动预测数量与真实标注数量不匹配")

        baseline_map = self._compute_map(base_samples, gt_samples)
        corrupted_map = self._compute_map(corr_samples, gt_samples)

        misses, _, gt_total, _ = compute_detection_errors(
            corr_samples,
            gt_samples,
            self.iou_threshold,
        )
        miss_rate = misses / gt_total if gt_total > 0 else 0.0
        magnitude = self._compute_perturbation_magnitude(original_images, corrupted_images)

        metrics = self._compose_metrics(
            baseline_map,
            corrupted_map,
            miss_rate,
            magnitude,
            metric_filter,
        )

        return CorruptionEvaluationResult(
            corruption_name=corruption_name,
            severity=int(severity),
            metrics=metrics,
        )

    def _compute_map(
        self,
        predictions: Sequence[DetectionSample],
        ground_truths: Sequence[DetectionSample],
    ) -> float:
        evaluation = self._detector.evaluate(list(predictions), list(ground_truths))
        details = evaluation.get(self.iou_threshold, {})
        return float(details.get("map", 0.0))

    def _compute_perturbation_magnitude(
        self,
        original_images: Sequence[torch.Tensor],
        corrupted_images: Sequence[torch.Tensor],
    ) -> float:
        total = 0.0
        count = 0
        for clean, perturbed in zip(original_images, corrupted_images):
            clean_np = self._to_numpy(clean)
            perturbed_np = self._to_numpy(perturbed)
            if clean_np.shape != perturbed_np.shape:
                min_shape = tuple(min(a, b) for a, b in zip(clean_np.shape, perturbed_np.shape))
                slices = tuple(slice(0, dim) for dim in min_shape)
                clean_np = clean_np[slices]
                perturbed_np = perturbed_np[slices]
            denom = float(np.maximum(clean_np.max() - clean_np.min(), 1.0))
            total += float(np.mean(np.abs(perturbed_np - clean_np)) / denom)
            count += 1
        return total / count if count else 0.0

    def _compose_metrics(
        self,
        baseline_map: float,
        corrupted_map: float,
        miss_rate: float,
        perturbation_magnitude: float,
        metric_filter: Optional[Mapping[str, None]],
    ) -> CorruptionRobustnessMetrics:
        drop_rate = self._map_drop(baseline_map, corrupted_map)
        tolerance = float(max(0.0, 1.0 - miss_rate))

        metric_values: Dict[str, float] = {
            "perturbation_magnitude": perturbation_magnitude,
            "performance_drop_rate": drop_rate,
            "perturbation_tolerance": tolerance,
        }

        if metric_filter:
            metric_values = {k: metric_values[k] for k in metric_values if k in metric_filter}

        return CorruptionRobustnessMetrics(
            perturbation_magnitude=metric_values.get("perturbation_magnitude", 0.0),
            performance_drop_rate=metric_values.get("performance_drop_rate", 0.0),
            perturbation_tolerance=metric_values.get("perturbation_tolerance", 0.0),
        )

    @staticmethod
    def _map_drop(baseline_map: float, corrupted_map: float) -> float:
        if baseline_map <= 0:
            return 0.0
        drop = (baseline_map - corrupted_map) / baseline_map
        return float(max(0.0, drop))

    @staticmethod
    def _normalize_metric_selection(
        metrics_to_report: Optional[Iterable[str]]
    ) -> Optional[Mapping[str, None]]:
        if metrics_to_report is None:
            return None
        normalized: Dict[str, None] = {}
        for name in metrics_to_report:
            if not isinstance(name, str):
                continue
            candidate = name.strip().lower()
            if candidate in {
                "perturbation_magnitude",
                "performance_drop_rate",
                "perturbation_tolerance",
            }:
                normalized[candidate] = None
        return normalized or None

    @staticmethod
    def _to_samples(
        entries: Sequence[PredictionLike],
        sample_type: str,
    ) -> List[DetectionSample]:
        samples: List[DetectionSample] = []
        for entry in entries:
            if isinstance(entry, DetectionSample):
                samples.append(entry)
            else:
                payload = dict(entry)
                if sample_type == "prediction":
                    samples.append(DetectionSample.from_prediction(payload))
                elif sample_type == "ground_truth":
                    samples.append(DetectionSample.from_ground_truth(payload))
                else:
                    raise ValueError(f"未知的样本类型: {sample_type}")
        return samples

    @staticmethod
    def _to_numpy(image: torch.Tensor) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().float().numpy()
        return np.asarray(image, dtype=np.float32)


def _ensure_tensor(image: torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().float()
    else:
        tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _match_range(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    ref_min = float(reference.min()) if reference.numel() else 0.0
    ref_max = float(reference.max()) if reference.numel() else 1.0
    if ref_max <= ref_min:
        ref_min, ref_max = 0.0, 1.0
    return value.clamp(ref_min, ref_max)


def gaussian_noise(
    image: torch.Tensor,
    severity: int = 1,
    magnitude: float = 0.05,
) -> torch.Tensor:
    tensor = _ensure_tensor(image)
    noise_scale = magnitude * max(1, severity)
    noisy = tensor + torch.randn_like(tensor) * noise_scale
    return _match_range(tensor, noisy)


def gaussian_blur(
    image: torch.Tensor,
    severity: int = 1,
    kernel_multiplier: int = 2,
) -> torch.Tensor:
    tensor = _ensure_tensor(image)
    kernel = int(max(1, 1 + 2 * severity * kernel_multiplier))
    if kernel % 2 == 0:
        kernel += 1
    sigma = max(0.1, float(severity))
    blurred = TF.gaussian_blur(tensor, kernel_size=kernel, sigma=sigma)
    return _match_range(tensor, blurred)


def brightness_shift(
    image: torch.Tensor,
    severity: int = 1,
    max_shift: float = 0.25,
) -> torch.Tensor:
    tensor = _ensure_tensor(image)
    shift = max_shift * max(1, severity)
    factor = 1.0 + torch.clamp(torch.tensor(shift), 0.0, 1.5).item()
    adjusted = TF.adjust_brightness(tensor, brightness_factor=factor)
    return _match_range(tensor, adjusted)


def contrast_shift(
    image: torch.Tensor,
    severity: int = 1,
    max_shift: float = 0.4,
) -> torch.Tensor:
    tensor = _ensure_tensor(image)
    factor = 1.0 - max_shift * max(1, severity) * 0.5
    factor = float(max(0.1, factor))
    adjusted = TF.adjust_contrast(tensor, contrast_factor=factor)
    return _match_range(tensor, adjusted)


_CORRUPTION_LIBRARY: Dict[str, Callable[..., torch.Tensor]] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness_shift,
    "brightness_shift": brightness_shift,
    "contrast": contrast_shift,
    "contrast_shift": contrast_shift,
}


def list_available_corruptions() -> Tuple[str, ...]:
    return tuple(sorted(_CORRUPTION_LIBRARY.keys()))


def apply_image_corruption(
    image: torch.Tensor,
    method: str,
    severity: int,
    parameters: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    if severity <= 0:
        severity = 1
    fn = _CORRUPTION_LIBRARY.get(method.lower())
    if fn is None:
        available = ", ".join(list_available_corruptions())
        raise ValueError(f"未知的扰动方法 '{method}'，可选项: {available}")
    params = dict(parameters or {})
    return fn(image, severity=severity, **params)

__all__ = [
    "CorruptionEvaluationResult",
    "CorruptionRobustnessEvaluator",
    "CorruptionRobustnessMetrics",
    "apply_image_corruption",
    "list_available_corruptions",
]
