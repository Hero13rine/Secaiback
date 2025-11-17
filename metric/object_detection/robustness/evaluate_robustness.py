"""用于评估检测模型对抗鲁棒性的入口点."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

from attack import AttackFactory

DEFAULT_CORRUPTION_PARAMETER_CONFIG = Path("config/attack/corruption.yaml")

from .adversarial import (
    AdversarialRobustnessEvaluator,
    AttackEvaluationResult,
    PredictionLike,
    RobustnessMetrics,
)
from .corruption import (
    CorruptionEvaluationResult,
    CorruptionRobustnessEvaluator,
    apply_image_corruption,
)


@dataclass(frozen=True)
class AttackConfig:
    """单个对抗攻击评估的配置.

    Attributes:
        name (str): 攻击名称
        enabled (bool): 是否启用此攻击评估，默认为True
        metrics (Optional[Tuple[str, ...]]): 要计算的指标列表，默认为None
        factory_config (Optional[Dict[str, Any]]): 攻击工厂配置，默认为None
    """

    name: str
    enabled: bool = True
    factory_config: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CorruptionConfig:
    """单个自然扰动评估的配置."""

    name: str
    method: str
    enabled: bool = True
    metrics: Optional[Tuple[str, ...]] = None
    severities: Tuple[int, ...] = (1, 3, 5)
    parameters: Mapping[str, Any] = field(default_factory=dict)


def load_robustness_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """从YAML文件加载鲁棒性配置.

    Args:
        config_path (Union[str, Path]): 配置文件路径

    Returns:
        Dict[str, Any]: 解析后的配置字典
    """

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
    """使用AttackFactory驱动的攻击评估对抗鲁棒性.

    Args:
        estimator: 能够预测检测输出的估计器。当它暴露了``get_core``方法时，
                   返回的ART估计器将用于攻击生成。
        test_data: 单个数据加载器/可迭代对象，产生 ``(images, targets)`` 批次，
                   或者是从任意键到此类数据加载器的映射。
                   图像应该是与 ``estimator`` 的 ``predict`` 方法兼容的张量。
        config: 允许选择攻击和指标的解析后配置字典。
        config_path: 包含配置的YAML文件的可选路径。
        iou_threshold: 指标计算的IoU阈值。
        batch_size: 调用估计器进行预测时一起处理的样本数量。

    Returns:
        一个字典，将攻击名称映射到其鲁棒性评估结果。
    """

    # 如果提供了配置文件路径，则加载配置
    if config_path:
        config = load_robustness_config(config_path)
    config = config or {}
    print("进度: 开始处理数据集...")
    images, targets = _collect_dataset(test_data)
    if not images:
        raise ValueError("测试数据必须至少提供一个样本")
    print("进度", "数据集已加载")

    ground_truths: List[PredictionLike] = [_normalize_target(target) for target in targets]
    baseline_predictions = _run_predictions(estimator, images, batch_size=batch_size)

    # 解析攻击配置
    print("进度: 解析攻击配置...")
    attack_configs, attack_metrics = _parse_attack_configs(config)
    if not attack_configs:
        return {}
    metrics_to_report = attack_metrics or tuple()
    # 创建对抗鲁棒性评估器实例
    evaluator = AdversarialRobustnessEvaluator(
        iou_threshold=iou_threshold,
    )
    results: Dict[str, AttackEvaluationResult] = {}

    # 对每种攻击进行评估
    total_attacks = len([attack for attack in attack_configs if attack.enabled])
    print(f"进度: 开始执行攻击流程测试，共 {total_attacks} 种攻击...")

    processed_attacks = 0
    for attack in attack_configs:
        if not attack.enabled:
            continue

        processed_attacks += 1
        print(f"进度: 处理攻击 '{attack.name}' ({processed_attacks}/{total_attacks})...")

        # 实例化攻击对象
        attack_instance = _instantiate_attack(estimator, attack.factory_config)
        print("  进度: 生成对抗样本...")
        adv_images = [
            _generate_adversarial_image(attack_instance, image) for image in images
        ]
        attack_predictions = _run_predictions(
            estimator,
            adv_images,
            batch_size=batch_size,
        )

        print(f"  进度: 评估攻击效果...")
        result = evaluator.evaluate_attack(
            attack.name,
            baseline_predictions,
            attack_predictions,
            ground_truths,
            metrics_to_report,
        )
        results[attack.name] = result
        print(f"进度: 攻击 '{attack.name}' 处理完成")

    print("进度: 所有攻击处理完成")
    return results


def evaluate_corruption_robustness(
    estimator,
    test_data: Union[Iterable, Mapping[Any, Iterable]],
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    iou_threshold: float = 0.5,
    batch_size: int = 1,
) -> Dict[str, CorruptionEvaluationResult]:
    """评估模型在自然扰动下的鲁棒性."""

    if config_path:
        config = load_robustness_config(config_path)
    config = config or {}

    print("进度: 开始收集用于扰动评测的数据集...")
    images, targets = _collect_dataset(test_data)
    if not images:
        raise ValueError("测试数据必须至少包含一张图像以执行扰动评测")
    print("进度: 数据集已准备")

    ground_truths: List[PredictionLike] = [_normalize_target(target) for target in targets]
    baseline_predictions = _run_predictions(estimator, images, batch_size=batch_size)

    print("进度: 解析扰动配置...")
    corruption_configs = _parse_corruption_configs(config)
    if not corruption_configs:
        return {}

    evaluator = CorruptionRobustnessEvaluator(iou_threshold=iou_threshold)
    results: Dict[str, CorruptionEvaluationResult] = {}

    enabled_configs = [corruption for corruption in corruption_configs if corruption.enabled]
    print(f"进度: 即将执行 {len(enabled_configs)} 种扰动方案...")
    for corruption in enabled_configs:
        for severity in corruption.severities:
            print(
                f"进度: 执行扰动 '{corruption.name}' (method={corruption.method}, severity={severity})"
            )
            corrupted_images = [
                apply_image_corruption(image, corruption.method, severity, corruption.parameters)
                for image in images
            ]
            predictions = _run_predictions(
                estimator,
                corrupted_images,
                batch_size=batch_size,
            )
            result = evaluator.evaluate_corruption(
                corruption.name,
                severity,
                baseline_predictions,
                predictions,
                ground_truths,
                images,
                corrupted_images,
                corruption.metrics,
            )
            key = f"{corruption.name}_severity_{severity}"
            results[key] = result
            print(f"进度: 扰动 '{corruption.name}' (severity={severity}) 处理完成")

    print("进度: 所有扰动评测完成")
    return results


def _parse_attack_configs(config: Mapping[str, Any]) -> Tuple[List[AttackConfig], Optional[Tuple[str, ...]]]:
    """将分层的YAML负载转换为攻击选择和共享指标.

    Args:
        config (Mapping[str, Any]): 配置字典

    Returns:
       Tuple[List[AttackConfig], Optional[Tuple[str, ...]]]:
            解析后的攻击配置列表以及在顶层声明的指标.

    """

    if not isinstance(config, Mapping):
        return [], None

    # 获取鲁棒性部分配置
    robustness_section = config.get("robustness")
    if robustness_section is None and "evaluation" in config:
        evaluation_section = config.get("evaluation")
        if isinstance(evaluation_section, Mapping):
            robustness_section = evaluation_section.get("robustness")

    # 获取对抗攻击部分配置
    adversarial_section = (
        robustness_section.get("adversarial") if isinstance(robustness_section, Mapping) else None
    )

    if adversarial_section is None:
        return [], None

    # 处理仅提供指标列表的情况（分类风格配置）
    if isinstance(adversarial_section, Sequence) and not isinstance(
        adversarial_section, (str, bytes)
    ):
        # 当提供的是纯指标列表（分类风格配置）时，
        # 只记录默认值而不选择攻击
        default_metrics = _extract_metric_list(adversarial_section)
        return [
            AttackConfig(
                name="fgsm",
                factory_config={"method": "fgsm", "parameters": {}},
            )
        ], default_metrics

    if not isinstance(adversarial_section, Mapping):
        return [], None

    # 提取默认指标
    metrics_payload = adversarial_section.get("metrics") or adversarial_section.get("default_metrics")
    default_metrics = _extract_metric_list(metrics_payload)

    attacks_payload = adversarial_section.get("attacks")

    # 如果没有明确指定攻击，则根据默认指标创建FGSM攻击
    if attacks_payload is None:
        # 允许只指定默认指标但不显式指定攻击的配置
        if default_metrics is not None:
            return [
                AttackConfig(
                    name="fgsm",
                    factory_config={"method": "fgsm", "parameters": {}},
                )
            ], default_metrics
        return [], default_metrics

    # 解析攻击配置
    parsed: List[AttackConfig] = []
    if isinstance(attacks_payload, Mapping):
        for name, payload in attacks_payload.items():
            attack = _build_attack_config(name, payload)
            if attack:
                parsed.append(attack)
    elif isinstance(attacks_payload, Sequence) and not isinstance(attacks_payload, (str, bytes)):
        for payload in attacks_payload:
            if isinstance(payload, str):
                parsed.append(
                    AttackConfig(
                        name=payload,
                        factory_config={"method": payload, "parameters": {}},
                    )
                )
            elif isinstance(payload, Mapping):
                name = payload.get("name") or payload.get("method")
                if not name:
                    continue
                attack = _build_attack_config(name, payload)
                if attack:
                    parsed.append(attack)
    else:
        raise ValueError("不支持的攻击配置格式")

    return [attack for attack in parsed if attack.enabled], default_metrics


def _parse_corruption_configs(config: Mapping[str, Any]) -> List[CorruptionConfig]:
    """解析自然扰动配置."""

    if not isinstance(config, Mapping):
        return []

    robustness_section = config.get("robustness")
    if robustness_section is None and "evaluation" in config:
        evaluation_section = config.get("evaluation")
        if isinstance(evaluation_section, Mapping):
            robustness_section = evaluation_section.get("robustness")

    corruption_section = (
        robustness_section.get("corruption") if isinstance(robustness_section, Mapping) else None
    )
    if corruption_section is None:
        return []

    default_metrics = _extract_corruption_metric_list(
        corruption_section.get("mertics")
        or corruption_section.get("metrics")
        or corruption_section.get("default_metrics")
    )

    parameter_config_path = (
        corruption_section.get("parameter_config")
        or corruption_section.get("parameters_config")
    )
    parameter_overrides = _load_corruption_parameter_overrides(parameter_config_path)

    corruptions_payload = corruption_section.get("corruptions") or corruption_section.get("methods")
    if corruptions_payload is None:
        reserved_keys = {
            "mertics",
            "metrics",
            "default_metrics",
            "default_severities",
            "parameter_config",
            "parameters_config",
            "corruptions",
            "methods",
        }
        inferred_payload: List[str] = []
        for key, value in corruption_section.items():
            if key in reserved_keys:
                continue
            if value is False:
                continue
            inferred_payload.append(key)
        if inferred_payload:
            corruptions_payload = inferred_payload
    if corruptions_payload is None:
        if default_metrics is not None:
            fallback = _build_corruption_config(
                "gaussian_noise",
                default_metrics,
                parameter_overrides,
            )
            return [fallback] if fallback else []
        return []

    parsed: List[CorruptionConfig] = []
    if isinstance(corruptions_payload, Mapping):
        for name, payload in corruptions_payload.items():
            if isinstance(payload, bool) and not payload:
                continue
            method_hint = None
            if isinstance(payload, Mapping):
                raw_method = payload.get("method") or payload.get("name")
                if isinstance(raw_method, str):
                    method_hint = raw_method.strip() or None
            corruption = _build_corruption_config(
                name,
                default_metrics,
                parameter_overrides,
                method_hint=method_hint,
            )
            if corruption:
                parsed.append(corruption)
    elif isinstance(corruptions_payload, Sequence) and not isinstance(
        corruptions_payload, (str, bytes)
    ):
        for payload in corruptions_payload:
            method_hint = None
            if isinstance(payload, str):
                name = payload.strip()
                if not name:
                    continue
            elif isinstance(payload, Mapping):
                raw_name = payload.get("name") or payload.get("method")
                if not isinstance(raw_name, str):
                    continue
                name = raw_name.strip()
                raw_method = payload.get("method")
                if isinstance(raw_method, str):
                    method_hint = raw_method.strip() or None
            else:
                continue

            corruption = _build_corruption_config(
                name,
                default_metrics,
                parameter_overrides,
                method_hint=method_hint,
            )
            if corruption:
                parsed.append(corruption)
    else:
        raise ValueError("不支持的扰动配置格式")

    return parsed


def _load_corruption_parameter_overrides(
    config_path: Optional[Union[str, Path]]
) -> Mapping[str, Any]:
    """加载可调节的扰动参数覆盖."""

    candidate_paths: List[Path] = []
    if config_path:
        try:
            candidate_paths.append(Path(config_path))
        except TypeError:
            pass
    candidate_paths.append(DEFAULT_CORRUPTION_PARAMETER_CONFIG)

    overrides: Mapping[str, Any] = {}
    seen: set = set()
    for candidate in candidate_paths:
        if not candidate:
            continue
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            with open(normalized, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"警告: 无法读取扰动参数文件 {normalized}: {exc}")
            continue

        if isinstance(payload, Mapping):
            corruptions_payload = payload.get("corruptions")
            if isinstance(corruptions_payload, Mapping):
                overrides = corruptions_payload
            else:
                overrides = payload
            break

    return overrides


def _build_corruption_config(
    name: str,
    default_metrics: Optional[Tuple[str, ...]],
    parameter_overrides: Mapping[str, Any],
    *,
    method_hint: Optional[str] = None,
) -> Optional[CorruptionConfig]:
    """构建扰动配置对象."""

    if not name:
        return None

    override_payload = (
        parameter_overrides.get(name)
        if isinstance(parameter_overrides, Mapping)
        else None
    )

    if isinstance(override_payload, Mapping):
        override_parameters = dict(override_payload.get("parameters", {}))
        severities = _extract_severity_list(override_payload.get("severities"))
        override_method = override_payload.get("method")
    else:
        override_parameters = {}
        severities = _extract_severity_list(None)
        override_method = None

    method_name = method_hint or (override_method if isinstance(override_method, str) else None)
    if not method_name:
        method_name = name

    return CorruptionConfig(
        name=name,
        method=method_name,
        enabled=True,
        metrics=default_metrics,
        severities=severities,
        parameters=override_parameters,
    )


def _build_attack_config(
    name: str,
    payload: Any,
) -> Optional[AttackConfig]:
    """构建攻击配置对象

    Args:
        name (str): 攻击名称
        payload (Any): 攻击配置载荷
        default_metrics (Optional[Tuple[str, ...]]): 默认指标

    Returns:
        Optional[AttackConfig]: 构建的攻击配置对象
    """

    # 处理布尔类型的payload
    if isinstance(payload, bool):
        return AttackConfig(
            name=name,
            enabled=payload,
            factory_config={"method": name, "parameters": {}},
        )

    # 处理None类型的payload
    if payload is None:
        return AttackConfig(
            name=name,
            factory_config={"method": name, "parameters": {}},
        )

    # 处理字符串类型的payload
    if isinstance(payload, str):
        method_name = payload.strip() or name
        return AttackConfig(
            name=name,
            factory_config={"method": method_name, "parameters": {}},
        )

    # 处理序列类型的payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return AttackConfig(
            name=name,
            factory_config={"method": name, "parameters": {}},
        )

    # 不支持的类型
    if not isinstance(payload, Mapping):
        return None

    # 处理字典类型的payload
    enabled = bool(payload.get("enabled", True))
    method_name = payload.get("method") or name
    # 参数已经在 config/attack 中统一管理，因此这里始终传递空字典
    parameters: Dict[str, Any] = {}

    # 处理工厂配置
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
        factory_config=factory_config,
    )


def _extract_metric_list(
    metrics_payload: Any,
    fallback: Optional[Tuple[str, ...]] = None,
) -> Optional[Tuple[str, ...]]:
    """提取指标列表

    Args:
        metrics_payload (Any): 指标配置载荷
        fallback (Optional[Tuple[str, ...]]): 备用指标列表

    Returns:
        Optional[Tuple[str, ...]]: 提取的指标元组
    """

    # 处理None值
    if metrics_payload is None:
        return fallback

    # 处理字典类型
    if isinstance(metrics_payload, Mapping):
        include = metrics_payload.get("include")
        return _extract_metric_list(include, fallback)

    # 处理序列类型
    if isinstance(metrics_payload, Sequence) and not isinstance(metrics_payload, (str, bytes)):
        normalized: List[str] = []
        for item in metrics_payload:
            if not isinstance(item, str):
                continue
            normalized_name = item.strip().lower()
            # 只保留支持的指标名称
            if normalized_name in {"map_drop_rate", "miss_rate", "false_detection_rate"}:
                normalized.append(normalized_name)
        return tuple(normalized) if normalized else fallback

    # 处理字符串类型
    if isinstance(metrics_payload, str):
        normalized_name = metrics_payload.strip().lower()
        if normalized_name in {"map_drop_rate", "miss_rate", "false_detection_rate"}:
            return (normalized_name,)

    return fallback


def _extract_corruption_metric_list(
    metrics_payload: Any,
    fallback: Optional[Tuple[str, ...]] = None,
) -> Optional[Tuple[str, ...]]:
    if metrics_payload is None:
        return fallback
    allowed = {
        "perturbation_magnitude",
        "performance_drop_rate",
        "perturbation_tolerance",
    }

    if isinstance(metrics_payload, Mapping):
        include = metrics_payload.get("include")
        return _extract_corruption_metric_list(include, fallback)

    if isinstance(metrics_payload, Sequence) and not isinstance(metrics_payload, (str, bytes)):
        normalized = []
        for item in metrics_payload:
            if isinstance(item, str):
                candidate = item.strip().lower()
                if candidate in allowed:
                    normalized.append(candidate)
        return tuple(normalized) if normalized else fallback

    if isinstance(metrics_payload, str):
        candidate = metrics_payload.strip().lower()
        if candidate in allowed:
            return (candidate,)

    return fallback


def _extract_severity_list(
    severity_payload: Any,
    fallback: Tuple[int, ...] = (1, 3, 5),
) -> Tuple[int, ...]:
    if severity_payload is None:
        return fallback

    severities: List[int] = []
    if isinstance(severity_payload, (int, float)):
        value = int(severity_payload)
        if value > 0:
            severities.append(value)
    elif isinstance(severity_payload, Sequence) and not isinstance(
        severity_payload, (str, bytes)
    ):
        for item in severity_payload:
            if isinstance(item, (int, float)):
                value = int(item)
                if value > 0:
                    severities.append(value)
    else:
        return fallback

    return tuple(sorted(set(severities))) if severities else fallback


def _collect_dataset(
    test_data: Union[Iterable, Mapping[Any, Iterable]]
) -> Tuple[List[torch.Tensor], List[Mapping[str, Any]]]:
    """收集数据集"""

    images: List[torch.Tensor] = []
    targets: List[Mapping[str, Any]] = []

    if isinstance(test_data, Mapping):
        loaders = list(test_data.values())
    else:
        loaders = [test_data]

    for loader in loaders:
        for images_batch, targets_batch in loader:
            image_list = _to_sequence(images_batch)
            target_list = _to_sequence(targets_batch)
            if len(image_list) != len(target_list):
                raise ValueError("图像与标注的数量不匹配")
            for image, target in zip(image_list, target_list):
                images.append(_clone_image(image))
                targets.append(_normalize_target(target))

    return images, targets


def _run_predictions(
    estimator,
    images: Sequence[torch.Tensor],
    batch_size: int = 1,
) -> List[PredictionLike]:
    """运行预测

    Args:
        estimator: 估计器对象
        images (Sequence[torch.Tensor]): 图像序列
        batch_size (int): 批处理大小

    Returns:
        List[PredictionLike]: 预测结果列表
    """

    predictions: List[PredictionLike] = []
    # 确保批处理大小至少为1
    if batch_size <= 0:
        batch_size = 1

    # 分批处理图像并进行预测
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch_inputs = _stack_batch(batch)
        outputs = estimator.predict(batch_inputs)
        for prediction in _to_sequence(outputs):
            predictions.append(_normalize_prediction(prediction))

    return predictions


def _instantiate_attack(estimator, factory_config: Optional[Dict[str, Any]]):
    """实例化攻击对象

    Args:
        estimator: 估计器对象
        factory_config (Optional[Dict[str, Any]]): 工厂配置

    Returns:
        实例化的攻击对象
    """

    # 检查工厂配置是否存在
    if not factory_config:
        raise ValueError("攻击配置必须定义方法和参数")

    # 获取核心估计器（如果存在get_core方法）
    core_estimator = estimator.get_core() if hasattr(estimator, "get_core") else estimator
    return AttackFactory.create(estimator=core_estimator, config=dict(factory_config))


def _generate_adversarial_image(attack, image: torch.Tensor) -> torch.Tensor:
    """生成对抗图像

    Args:
        attack: 攻击对象
        image (torch.Tensor): 原始图像

    Returns:
        torch.Tensor: 对抗图像
    """

    # 将图像转换为NumPy数组
    np_input = _to_numpy_image(image)
    np_input = np_input.astype(np.float32, copy=False)
    # 生成对抗样本批次
    adv_batch = attack.generate(np_input[None, ...].astype(np.float32, copy=False))
    if isinstance(adv_batch, np.ndarray):
        adv_np = adv_batch.astype(np.float32, copy=False)[0]
    else:
        adv_np = np.array(adv_batch, dtype=np.float32)[0]
    adv_tensor = torch.from_numpy(adv_np)
    # 保持与原始图像相同的设备和数据类型
    if isinstance(image, torch.Tensor):
        adv_tensor = adv_tensor.to(image.device)
        adv_tensor = adv_tensor.type(image.dtype)
    return adv_tensor

def _stack_batch(batch: Sequence[torch.Tensor]) -> torch.Tensor:
    """堆叠批次图像

    Args:
        batch (Sequence[torch.Tensor]): 图像批次

    Returns:
        torch.Tensor: 堆叠后的张量
    """

    # 处理单张图像的情况
    if len(batch) == 1:
        image = batch[0]
        return image.unsqueeze(0) if isinstance(image, torch.Tensor) else torch.as_tensor(image)[None]

    # 如果所有元素都是张量，则直接堆叠
    if all(isinstance(item, torch.Tensor) for item in batch):
        return torch.stack(batch)

    # 否则先转换为张量再堆叠
    arrays = [torch.as_tensor(item) for item in batch]
    return torch.stack(arrays)


def _normalize_prediction(prediction: Any) -> Mapping[str, Any]:
    """标准化预测结果

    Args:
        prediction (Any): 原始预测结果

    Returns:
        Mapping[str, Any]: 标准化后的预测结果
    """

    # 处理字典类型
    if isinstance(prediction, Mapping):
        return {key: value for key, value in prediction.items()}

    # 处理具有_asdict方法的对象
    if hasattr(prediction, "_asdict"):
        return prediction._asdict()

    # 处理三元组（boxes, scores, labels）
    if isinstance(prediction, tuple) and len(prediction) == 3:
        boxes, scores, labels = prediction
        return {"boxes": boxes, "scores": scores, "labels": labels}

    # 默认返回空的预测结果
    return {"boxes": [], "scores": [], "labels": []}


def _normalize_target(target: Any) -> Mapping[str, Any]:
    """标准化目标

    Args:
        target (Any): 原始目标

    Returns:
        Mapping[str, Any]: 标准化后的目标
    """

    # 处理字典类型
    if isinstance(target, Mapping):
        return {key: value for key, value in target.items()}

    # 处理具有_asdict方法的对象
    if hasattr(target, "_asdict"):
        return target._asdict()

    # 尝试转换为字典
    try:
        return dict(target)
    except (TypeError, ValueError):
        return {"boxes": [], "labels": []}


def _clone_image(image: Any) -> torch.Tensor:
    """克隆图像

    Args:
        image (Any): 原始图像

    Returns:
        torch.Tensor: 克隆后的图像张量
    """

    # 处理PyTorch张量
    if isinstance(image, torch.Tensor):
        cloned_image = image.detach().clone()
        # 如果图像尺寸较大，则调整图像大小以减少内存使用
        if cloned_image.numel() > 1000000:  # 如果元素数量超过100万
            # 检查是否为3通道图像 (C, H, W)
            if cloned_image.dim() == 3 and cloned_image.shape[0] == 3:
                # 缩放图像到较小尺寸以减少内存使用
                from torchvision import transforms
                resize_transform = transforms.Resize((416, 416))  # 缩小到416x416
                cloned_image = resize_transform(cloned_image)
            elif cloned_image.dim() == 3 and cloned_image.shape[0] == 1:
                # 灰度图像
                from torchvision import transforms
                resize_transform = transforms.Resize((416, 416))
                cloned_image = resize_transform(cloned_image)
        return cloned_image

    # 处理其他类型，转换为NumPy数组再转为张量
    array = np.array(image)
    return torch.as_tensor(array, dtype=torch.float32)


def _to_sequence(value: Any) -> List[Any]:
    """将值转换为序列

    Args:
        value (Any): 原始值

    Returns:
        List[Any]: 转换后的列表
    """

    # 处理PyTorch张量
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [value]
        return [value[i] for i in range(value.shape[0])]

    # 处理序列类型（排除字符串、字节和字典）
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        return list(value)

    # 其他情况返回单元素列表
    return [value]


def _to_numpy_image(image: Any) -> np.ndarray:
    """将图像转换为NumPy数组

    Args:
        image (Any): 原始图像

    Returns:
        np.ndarray: 转换后的NumPy数组
    """

    # 处理PyTorch张量
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()

    # 处理NumPy数组
    if isinstance(image, np.ndarray):
        return image.astype(np.float32, copy=False)

    # 其他类型转换为NumPy数组
    return np.asarray(image, dtype=np.float32)


__all__ = [
    "AttackConfig",
    "AttackEvaluationResult",
    "CorruptionConfig",
    "CorruptionEvaluationResult",
    "RobustnessMetrics",
    "evaluate_adversarial_robustness",
    "evaluate_corruption_robustness",
    "load_robustness_config",
]