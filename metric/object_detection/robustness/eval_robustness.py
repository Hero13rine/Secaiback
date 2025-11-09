"""Entry-points for adversarial robustness evaluation of detection models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

from attack import AttackFactory

from .adversarial import (
    AdversarialRobustnessEvaluator,
    AttackEvaluationResult,
    PredictionLike,
    RotationRobustnessMetrics,
)


@dataclass(frozen=True)
class AttackConfig:
    """Configuration for a single adversarial attack evaluation."""

    name: str
    enabled: bool = True
    rotations: Optional[Tuple[float, ...]] = None
    metrics: Optional[Tuple[str, ...]] = None
    factory_config: Optional[Dict[str, Any]] = None


def load_robustness_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load robustness configuration from a YAML file."""

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def evaluate_adversarial_robustness(
    estimator,
    test_data: Union[Iterable, Mapping[Any, Iterable]],
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    iou_threshold: float = 0.5,
    batch_size: int = 1,
) -> Dict[str, AttackEvaluationResult]:
    """Evaluate adversarial robustness using AttackFactory driven attacks.

    Args:
        estimator: Estimator capable of predicting detection outputs. When it
            exposes a ``get_core`` method the returned ART estimator will be
            used for attack generation.
        test_data: Either a single dataloader/iterable yielding ``(images,
            targets)`` batches or a mapping from rotation angle to such
            dataloaders. Images are expected to be tensors compatible with the
            ``predict`` method of ``estimator``.
        config: Parsed configuration dictionary allowing attack, rotation and
            metric selection.
        config_path: Optional path to a YAML file containing the configuration.
        iou_threshold: IoU threshold for metric calculations.
        batch_size: Number of samples processed together when invoking the
            estimator for predictions.

    Returns:
        A dictionary mapping attack names to their robustness evaluation
        results.
    """

    if config_path:
        config = load_robustness_config(config_path)
    config = config or {}

    data_by_rotation = _normalize_test_data(test_data)
    dataset = _collect_dataset(data_by_rotation)
    if not dataset:
        raise ValueError("Test data must provide at least one sample")

    ground_truths_by_rotation: Dict[float, List[PredictionLike]] = {}
    baseline_predictions: Dict[float, List[PredictionLike]] = {}

    for rotation, (images, targets) in dataset.items():
        ground_truths_by_rotation[rotation] = [_normalize_target(target) for target in targets]
        baseline_predictions[rotation] = _run_predictions(estimator, images, batch_size=batch_size)

    attack_configs = _parse_attack_configs(config)
    if not attack_configs:
        return {}

    evaluator = AdversarialRobustnessEvaluator(iou_threshold=iou_threshold)
    results: Dict[str, AttackEvaluationResult] = {}

    for attack in attack_configs:
        if not attack.enabled:
            continue
        attack_instance = _instantiate_attack(estimator, attack.factory_config)
        attack_predictions: Dict[float, List[PredictionLike]] = {}

        rotations_to_use = attack.rotations or tuple(dataset.keys())
        for rotation in rotations_to_use:
            if rotation not in dataset:
                raise KeyError(f"Rotation {rotation} not available in provided data")
            images, _ = dataset[rotation]
            adv_images = [_generate_adversarial_image(attack_instance, image) for image in images]
            attack_predictions[rotation] = _run_predictions(
                estimator,
                adv_images,
                batch_size=batch_size,
            )

        selected_baseline = _filter_rotations(baseline_predictions, attack.rotations)
        selected_ground_truths = _filter_rotations(ground_truths_by_rotation, attack.rotations)
        selected_attack = _filter_rotations(attack_predictions, attack.rotations)

        result = evaluator.evaluate_attack(
            attack.name,
            selected_baseline,
            selected_attack,
            selected_ground_truths,
            attack.metrics,
        )
        results[attack.name] = result

    return results


def _parse_attack_configs(config: Mapping[str, Any]) -> List[AttackConfig]:
    """Translate the hierarchical YAML payload into attack selections."""

    if not isinstance(config, Mapping):
        return []

    robustness_section = config.get("robustness")
    if robustness_section is None and "evaluation" in config:
        evaluation_section = config.get("evaluation")
        if isinstance(evaluation_section, Mapping):
            robustness_section = evaluation_section.get("robustness")

    adversarial_section = (
        robustness_section.get("adversarial") if isinstance(robustness_section, Mapping) else None
    )

    if adversarial_section is None:
        return []

    if isinstance(adversarial_section, Sequence) and not isinstance(
        adversarial_section, (str, bytes)
    ):
        # When a bare list of metrics is provided (classification style configs),
        # simply record the defaults without selecting attacks.
        default_metrics = _extract_metric_list(adversarial_section)
        return [
            AttackConfig(
                name="fgsm",
                metrics=default_metrics,
                factory_config={"method": "fgsm", "parameters": {}},
            )
        ]

    if not isinstance(adversarial_section, Mapping):
        return []

    default_metrics = _extract_metric_list(
        adversarial_section.get("default_metrics") or adversarial_section.get("metrics")
    )
    default_rotations = _normalize_rotations(adversarial_section.get("default_rotations"))
    attacks_payload = adversarial_section.get("attacks")

    if attacks_payload is None:
        # Allow configs that only specify default metrics but no explicit attacks.
        if default_metrics is not None:
            return [
                AttackConfig(
                    name="fgsm",
                    metrics=default_metrics,
                    rotations=default_rotations,
                    factory_config={"method": "fgsm", "parameters": {}},
                )
            ]
        return []

    parsed: List[AttackConfig] = []
    if isinstance(attacks_payload, Mapping):
        for name, payload in attacks_payload.items():
            attack = _build_attack_config(name, payload, default_metrics, default_rotations)
            if attack:
                parsed.append(attack)
    elif isinstance(attacks_payload, Sequence) and not isinstance(attacks_payload, (str, bytes)):
        for payload in attacks_payload:
            if isinstance(payload, str):
                parsed.append(
                    AttackConfig(
                        name=payload,
                        metrics=default_metrics,
                        rotations=default_rotations,
                        factory_config={"method": payload, "parameters": {}},
                    )
                )
            elif isinstance(payload, Mapping):
                name = payload.get("name") or payload.get("method")
                if not name:
                    continue
                attack = _build_attack_config(name, payload, default_metrics, default_rotations)
                if attack:
                    parsed.append(attack)
    else:
        raise ValueError("Unsupported attacks configuration format")

    return [attack for attack in parsed if attack.enabled]


def _build_attack_config(
    name: str,
    payload: Any,
    default_metrics: Optional[Tuple[str, ...]],
    default_rotations: Optional[Tuple[float, ...]],
) -> Optional[AttackConfig]:
    if isinstance(payload, bool):
        return AttackConfig(
            name=name,
            enabled=payload,
            metrics=default_metrics,
            rotations=default_rotations,
            factory_config={"method": name, "parameters": {}},
        )

    if payload is None:
        return AttackConfig(
            name=name,
            metrics=default_metrics,
            rotations=default_rotations,
            factory_config={"method": name, "parameters": {}},
        )

    if isinstance(payload, str):
        method_name = payload.strip() or name
        return AttackConfig(
            name=name,
            metrics=default_metrics,
            rotations=default_rotations,
            factory_config={"method": method_name, "parameters": {}},
        )

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        metrics = _extract_metric_list(payload, default_metrics)
        return AttackConfig(
            name=name,
            metrics=metrics,
            rotations=default_rotations,
            factory_config={"method": name, "parameters": {}},
        )

    if not isinstance(payload, Mapping):
        return None

    enabled = bool(payload.get("enabled", True))
    rotations = _normalize_rotations(payload.get("rotations")) or default_rotations
    metrics = _extract_metric_list(
        payload.get("metrics") or payload.get("outputs"), default_metrics
    )
    method_name = payload.get("method") or name
    parameters_payload = payload.get("parameters") or {}
    parameters = dict(parameters_payload) if isinstance(parameters_payload, Mapping) else {}

    factory_payload = payload.get("factory_config")
    if isinstance(factory_payload, Mapping):
        factory_config = {
            "method": factory_payload.get("method", method_name),
            "parameters": dict(factory_payload.get("parameters", {})),
        }
    else:
        factory_config = {"method": method_name, "parameters": parameters}

    return AttackConfig(
        name=name,
        enabled=enabled,
        rotations=rotations,
        metrics=metrics,
        factory_config=factory_config,
    )


def _extract_metric_list(
    metrics_payload: Any,
    fallback: Optional[Tuple[str, ...]] = None,
) -> Optional[Tuple[str, ...]]:
    if metrics_payload is None:
        return fallback

    if isinstance(metrics_payload, Mapping):
        include = metrics_payload.get("include")
        return _extract_metric_list(include, fallback)

    if isinstance(metrics_payload, Sequence) and not isinstance(metrics_payload, (str, bytes)):
        normalized: List[str] = []
        for item in metrics_payload:
            if not isinstance(item, str):
                continue
            normalized_name = item.strip().lower()
            if normalized_name in {"map_drop_rate", "miss_rate", "false_detection_rate"}:
                normalized.append(normalized_name)
        return tuple(normalized) if normalized else fallback

    if isinstance(metrics_payload, str):
        normalized_name = metrics_payload.strip().lower()
        if normalized_name in {"map_drop_rate", "miss_rate", "false_detection_rate"}:
            return (normalized_name,)

    return fallback


def _normalize_rotations(rotations: Any) -> Optional[Tuple[float, ...]]:
    if rotations is None:
        return None
    if isinstance(rotations, (int, float)):
        return (float(rotations),)
    if isinstance(rotations, Sequence) and not isinstance(rotations, (str, bytes)):
        normalized: List[float] = []
        for item in rotations:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(set(normalized))) if normalized else None
    return None


def _filter_rotations(
    mapping: Mapping[float, Sequence[PredictionLike]],
    rotations: Optional[Tuple[float, ...]],
) -> Dict[float, Sequence[PredictionLike]]:
    if not rotations:
        return dict(mapping)
    filtered: Dict[float, Sequence[PredictionLike]] = {}
    for rotation in rotations:
        if rotation not in mapping:
            raise KeyError(f"Rotation {rotation} not available in provided data")
        filtered[rotation] = mapping[rotation]
    return filtered


def _normalize_test_data(
    test_data: Union[Iterable, Mapping[Any, Iterable]]
) -> Dict[float, Iterable]:
    if isinstance(test_data, Mapping):
        normalized: Dict[float, Iterable] = {}
        for key, loader in test_data.items():
            try:
                rotation = float(key)
            except (TypeError, ValueError):
                continue
            normalized[rotation] = loader
        return normalized
    return {0.0: test_data}


def _collect_dataset(
    data_by_rotation: Mapping[float, Iterable]
) -> Dict[float, Tuple[List[torch.Tensor], List[Mapping[str, Any]]]]:
    dataset: Dict[float, Tuple[List[torch.Tensor], List[Mapping[str, Any]]]] = {}
    for rotation, loader in data_by_rotation.items():
        images: List[torch.Tensor] = []
        targets: List[Mapping[str, Any]] = []
        for images_batch, targets_batch in loader:
            for image in _to_sequence(images_batch):
                images.append(_clone_image(image))
            for target in _to_sequence(targets_batch):
                targets.append(_normalize_target(target))
        dataset[rotation] = (images, targets)
    return dataset


def _run_predictions(
    estimator,
    images: Sequence[torch.Tensor],
    batch_size: int = 1,
) -> List[PredictionLike]:
    predictions: List[PredictionLike] = []
    if batch_size <= 0:
        batch_size = 1

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch_inputs = _stack_batch(batch)
        outputs = estimator.predict(batch_inputs)
        for prediction in _to_sequence(outputs):
            predictions.append(_normalize_prediction(prediction))
    return predictions


def _instantiate_attack(estimator, factory_config: Optional[Dict[str, Any]]):
    if not factory_config:
        raise ValueError("Attack configuration must define method and parameters")
    core_estimator = estimator.get_core() if hasattr(estimator, "get_core") else estimator
    return AttackFactory.create(estimator=core_estimator, config=dict(factory_config))


def _generate_adversarial_image(attack, image: torch.Tensor) -> torch.Tensor:
    np_input = _to_numpy_image(image)
    adv_batch = attack.generate(np_input[None, ...])
    adv_np = adv_batch[0] if isinstance(adv_batch, np.ndarray) else np.array(adv_batch)[0]
    adv_tensor = torch.from_numpy(adv_np)
    if isinstance(image, torch.Tensor):
        adv_tensor = adv_tensor.to(image.device)
        adv_tensor = adv_tensor.type(image.dtype)
    return adv_tensor


def _stack_batch(batch: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(batch) == 1:
        image = batch[0]
        return image.unsqueeze(0) if isinstance(image, torch.Tensor) else torch.as_tensor(image)[None]
    if all(isinstance(item, torch.Tensor) for item in batch):
        return torch.stack(batch)
    arrays = [torch.as_tensor(item) for item in batch]
    return torch.stack(arrays)


def _normalize_prediction(prediction: Any) -> Mapping[str, Any]:
    if isinstance(prediction, Mapping):
        return {key: value for key, value in prediction.items()}
    if hasattr(prediction, "_asdict"):
        return prediction._asdict()
    if isinstance(prediction, tuple) and len(prediction) == 3:
        boxes, scores, labels = prediction
        return {"boxes": boxes, "scores": scores, "labels": labels}
    return {"boxes": [], "scores": [], "labels": []}


def _normalize_target(target: Any) -> Mapping[str, Any]:
    if isinstance(target, Mapping):
        return {key: value for key, value in target.items()}
    if hasattr(target, "_asdict"):
        return target._asdict()
    try:
        return dict(target)
    except (TypeError, ValueError):
        return {"boxes": [], "labels": []}


def _clone_image(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image.detach().clone()
    array = np.array(image)
    return torch.as_tensor(array, dtype=torch.float32)


def _to_sequence(value: Any) -> List[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [value]
        return [value[i] for i in range(value.shape[0])]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        return list(value)
    return [value]


def _to_numpy_image(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        return image.astype(np.float32, copy=False)
    return np.asarray(image, dtype=np.float32)


__all__ = [
    "AttackConfig",
    "AttackEvaluationResult",
    "RotationRobustnessMetrics",
    "evaluate_adversarial_robustness",
    "load_robustness_config",
]
