"""用于评估检测模型对抗鲁棒性的入口点."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import yaml

from attack import AttackFactory
from utils.sender import ResultSender

DEFAULT_CORRUPTION_PARAMETER_CONFIG = Path("config/attack/corruption.yaml")

from .adversarial import (
    AdversarialRobustnessEvaluator,
    AttackEvaluationResult,
    ALL_METRIC_ALIASES,
    PredictionLike,
    RobustnessMetrics,
)
from .corruption import (
    CorruptionEvaluationResult,
    CorruptionRobustnessEvaluator,
    apply_image_corruption,
)


@dataclass(frozen=True)
class ParameterSchedule:
    """对抗攻击调度参数."""

    start: float
    end: float
    step: float
    as_integer: bool = False


@dataclass(frozen=True)
class AttackConfig:
    """单个对抗攻击评估的配置."""

    name: str
    enabled: bool = True
    factory_config: Optional[Dict[str, Any]] = None
    sweep_parameter: Optional[str] = None
    sweep_schedule: Optional[ParameterSchedule] = None


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


def _attack_result_summary(
        results: Mapping[str, AttackEvaluationResult]
) -> List[Mapping[str, Any]]:
    """将对抗攻击结果转换为可发送的简洁结构."""

    summary: List[Mapping[str, Any]] = []
    for attack_name, result in results.items():
        metrics = result.metrics
        summary.append(
            {
                "attack_name": attack_name,
                "map_drop_rate": metrics.map_drop_rate,
                "miss_rate": metrics.miss_rate,
                "false_detection_rate": metrics.false_detection_rate,
            }
        )
    return summary


def _corruption_result_summary(
        results: Mapping[str, CorruptionEvaluationResult]
) -> List[Mapping[str, Any]]:
    """将自然扰动结果转换为可发送的简洁结构."""

    summary: List[Mapping[str, Any]] = []
    for key, result in results.items():
        metrics = result.metrics
        summary.append(
            {
                "corruption_key": key,
                "corruption_name": result.corruption_name,
                "severity": result.severity,
                "perturbation_magnitude": metrics.perturbation_magnitude,
                "performance_drop_rate": metrics.performance_drop_rate,
                "perturbation_tolerance": metrics.perturbation_tolerance,
            }
        )
    return summary


def _publish_robustness_results(
        adversarial_results: Mapping[str, AttackEvaluationResult],
        corruption_results: Mapping[str, CorruptionEvaluationResult],
) -> None:
    """统一发送鲁棒性评测结果."""

    payload: Dict[str, Any] = {}
    if adversarial_results:
        payload["adversarial"] = _attack_result_summary(adversarial_results)
    if corruption_results:
        payload["corruption"] = _corruption_result_summary(corruption_results)

    if payload:
        ResultSender.send_result("robustness", payload)
        ResultSender.send_log("进度", "鲁棒性评测结果已发送")
    else:
        message = "鲁棒性配置未启用攻击或扰动评估"
        ResultSender.send_log("提示", message)
        ResultSender.send_result("robustness", {"message": message})


def evaluation_robustness(
        estimator,
        test_data: Union[Iterable, Mapping[Any, Iterable]],
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        iou_threshold: float = 0.5,
        batch_size: int = 1,
) -> Dict[str, Mapping[str, Any]]:
    """统一入口，根据配置自动执行鲁棒性评估.

    该函数会解析 ``evaluation.robustness`` 段，分别调用对抗攻击与扰动攻击的
    评测函数，并返回统一的结果结构。
    """

    ResultSender.send_log("进度", "开始鲁棒性评估")
    if config_path:
        config = load_robustness_config(config_path)
    config = _normalize_robustness_config(config)
    robustness_section = config.get("robustness") if isinstance(config, Mapping) else None

    adversarial_results: Mapping[str, Any] = {}
    corruption_results: Mapping[str, Any] = {}

    if not isinstance(robustness_section, Mapping):
        message = "配置中未找到 'evaluation.robustness' 段，跳过鲁棒性评估"
        ResultSender.send_log("提示", message)
        ResultSender.send_result("robustness", {"message": message})
        return {"adversarial": adversarial_results, "corruption": corruption_results}

    if _is_section_enabled(robustness_section, "adversarial"):
        ResultSender.send_log("进度", "执行对抗鲁棒性评估")
        adversarial_results = evaluate_adversarial_robustness(
            estimator=estimator,
            test_data=test_data,
            config=config,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
        )
    else:
        ResultSender.send_log("提示", "未启用对抗攻击评估")

    if _is_section_enabled(robustness_section, "corruption"):
        ResultSender.send_log("进度", "执行扰动鲁棒性评估")
        corruption_results = evaluate_corruption_robustness(
            estimator=estimator,
            test_data=test_data,
            config=config,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
        )
    else:
        ResultSender.send_log("提示", "未启用扰动攻击评估")

    _publish_robustness_results(adversarial_results, corruption_results)

    return {"adversarial": adversarial_results, "corruption": corruption_results}


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
    ResultSender.send_log("进度", "准备对抗鲁棒性评估数据集")
    if config_path:
        config = load_robustness_config(config_path)
    config = config or {}
    images, targets = _collect_dataset(test_data)
    ResultSender.send_log("进度", f"对抗评测样本数量: {len(images)}")
    if not images:
        raise ValueError("测试数据必须至少提供一个样本")
    ResultSender.send_log("进度", "对抗评测数据集准备完成")

    ground_truths: List[PredictionLike] = [_normalize_target(target) for target in targets]
    baseline_predictions = _run_predictions(estimator, images, batch_size=batch_size)

    # 解析攻击配置
    ResultSender.send_log("进度", "解析对抗攻击配置")
    attack_configs, attack_metrics = _parse_attack_configs(config)
    if not attack_configs:
        ResultSender.send_log("提示", "未配置对抗攻击，跳过对抗鲁棒性评估")
        return {}
    metrics_to_report = attack_metrics or tuple()
    attack_iterations = [
        (attack, _expand_schedule_values(attack.sweep_schedule))
        for attack in attack_configs
        if attack.enabled
    ]
    total_attacks = sum(len(iterations) for _, iterations in attack_iterations)
    # 创建对抗鲁棒性评估器实例
    evaluator = AdversarialRobustnessEvaluator(
        iou_threshold=iou_threshold,
    )
    results: Dict[str, AttackEvaluationResult] = {}

    ResultSender.send_log("进度", f"共 {total_attacks} 种对抗攻击需要评测")

    processed_attacks = 0
    for attack, schedule in attack_iterations:
        for parameter_value in schedule:
            processed_attacks += 1
            attack_label = attack.name
            metadata: Dict[str, Any] = {"attack": attack.name}
            if (
                    attack.sweep_parameter is not None
                    and parameter_value is not None
            ):
                formatted_value = _format_parameter_value(parameter_value)
                attack_label = (
                    f"{attack.name}_{attack.sweep_parameter}_{formatted_value}"
                )
                metadata[attack.sweep_parameter] = parameter_value

            ResultSender.send_log(
                "进度",
                f"执行攻击 {attack_label} ({processed_attacks}/{total_attacks})",
            )

            factory_config = _prepare_attack_factory_config(attack)
            if attack.sweep_parameter and parameter_value is not None:
                factory_config = _override_attack_parameter(
                    factory_config, attack.sweep_parameter, parameter_value
                )

            attack_instance = _instantiate_attack(estimator, factory_config)
            adv_images = [
                _generate_adversarial_image(attack_instance, image) for image in images
            ]
            attack_predictions = _run_predictions(
                estimator,
                adv_images,
                batch_size=batch_size,
            )

            result = evaluator.evaluate_attack(
                attack_label,
                baseline_predictions,
                attack_predictions,
                ground_truths,
                metrics_to_report,
                metadata=metadata,
            )
            results[attack_label] = result
            ResultSender.send_log(
                "进度",
                f"攻击 {attack_label} 完成 ({processed_attacks}/{total_attacks})",
            )

    ResultSender.send_log("进度", "对抗鲁棒性评估完成")
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

    ResultSender.send_log("进度", "准备扰动鲁棒性评估数据集")
    if config_path:
        config = load_robustness_config(config_path)
    config = config or {}

    images, targets = _collect_dataset(test_data)
    ResultSender.send_log("进度", f"扰动评测样本数量: {len(images)}")
    if not images:
        raise ValueError("测试数据必须至少包含一张图像以执行扰动评测")
    ResultSender.send_log("进度", "扰动评测数据集准备完成")

    ground_truths: List[PredictionLike] = [_normalize_target(target) for target in targets]
    baseline_predictions = _run_predictions(estimator, images, batch_size=batch_size)

    ResultSender.send_log("进度", "解析扰动配置")
    corruption_configs = _parse_corruption_configs(config)
    if not corruption_configs:
        ResultSender.send_log("提示", "未配置扰动方案，跳过扰动鲁棒性评估")
        return {}

    evaluator = CorruptionRobustnessEvaluator(iou_threshold=iou_threshold)
    results: Dict[str, CorruptionEvaluationResult] = {}

    enabled_configs = [corruption for corruption in corruption_configs if corruption.enabled]
    ResultSender.send_log(
        "进度",
        f"共 {len(enabled_configs)} 种扰动方案需要评测",
    )
    for corruption in enabled_configs:
        for severity in corruption.severities:
            ResultSender.send_log(
                "进度",
                f"执行扰动 {corruption.name} (severity={severity})",
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
            ResultSender.send_log(
                "进度",
                f"扰动 {corruption.name} (severity={severity}) 完成",
            )

    ResultSender.send_log("进度", "扰动鲁棒性评估完成")
    return results


def _normalize_robustness_config(
        config: Optional[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """将配置统一为包含 ``robustness`` 键的结构，便于后续解析."""

    if not isinstance(config, Mapping):
        return {}

    if "robustness" in config:
        robustness_section = config.get("robustness")
        if isinstance(robustness_section, Mapping):
            return {"robustness": robustness_section}
        return config

    evaluation_section = config.get("evaluation")
    if isinstance(evaluation_section, Mapping) and "robustness" in evaluation_section:
        return {"robustness": evaluation_section.get("robustness")}

    robustness_like_keys = {"adversarial", "corruption"}
    if robustness_like_keys & set(config.keys()):
        return {"robustness": config}

    return {}


def _is_section_enabled(robustness_section: Mapping[str, Any], key: str) -> bool:
    """根据子配置判断是否需要执行对应的鲁棒性评估."""

    if not isinstance(robustness_section, Mapping):
        return False

    subsection = robustness_section.get(key)
    if subsection is None or subsection is False:
        return False

    if isinstance(subsection, Mapping):
        if subsection.get("enabled") is False:
            return False
        return True

    if isinstance(subsection, Sequence) and not isinstance(subsection, (str, bytes)):
        return len(subsection) > 0

    return True


def _parse_attack_configs(
        config: Mapping[str, Any]
) -> Tuple[List[AttackConfig], Optional[Tuple[str, ...]]]:
    """将分层的YAML负载转换为攻击选择和共享指标.

    Args:
        config (Mapping[str, Any]): 配置字典

    Returns:
       Tuple[List[AttackConfig], Optional[Tuple[str, ...]]]:
            解析后的攻击配置列表及在顶层声明的指标.

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
    parsed = _normalize_attack_declarations(attacks_payload)

    return (
        [attack for attack in parsed if attack.enabled],
        default_metrics,
    )


def _split_attack_parameters(
        parameters_payload: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Optional[str], Optional[ParameterSchedule]]:
    base_parameters: Dict[str, Any] = {}
    sweep_parameter: Optional[str] = None
    sweep_schedule: Optional[ParameterSchedule] = None

    for key, value in parameters_payload.items():
        schedule = _parse_parameter_schedule(value)
        if schedule:
            if sweep_parameter is not None:
                raise ValueError("暂不支持为单个攻击配置多个调度参数")
            sweep_parameter = key
            sweep_schedule = schedule
        else:
            base_parameters[key] = value

    return base_parameters, sweep_parameter, sweep_schedule


def _parse_parameter_schedule(payload: Any) -> Optional[ParameterSchedule]:
    if payload is None:
        return None

    start: Optional[float] = None
    end: Optional[float] = None
    step: Optional[float] = None

    if isinstance(payload, Mapping):
        start = payload.get("start") or payload.get("begin")
        end = payload.get("end") or payload.get("stop") or payload.get("final")
        step = payload.get("step") or payload.get("increment")
    elif isinstance(payload, Sequence) and len(payload) >= 3 and not isinstance(
            payload, (str, bytes)
    ):
        start, end, step = payload[:3]
    else:
        return None

    try:
        start_f = float(start)
        end_f = float(end)
        step_f = float(step)
    except (TypeError, ValueError):
        raise ValueError("参数调度必须包含可转换为浮点数的起始、结束和步长")

    if step_f == 0:
        raise ValueError("参数调度的步长必须非零")
    if step_f > 0 and start_f > end_f:
        raise ValueError("当步长为正时，起始值必须小于等于结束值")
    if step_f < 0 and start_f < end_f:
        raise ValueError("当步长为负时，起始值必须大于等于结束值")

    as_integer = all(_looks_like_int(value) for value in (start, end, step))
    return ParameterSchedule(start=start_f, end=end_f, step=step_f, as_integer=as_integer)


def _looks_like_int(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _expand_schedule_values(
        schedule: Optional[ParameterSchedule],
) -> List[Optional[Union[int, float]]]:
    if not schedule:
        return [None]

    start, end, step = schedule.start, schedule.end, schedule.step
    values: List[Union[int, float]] = []
    current = start
    if step > 0:
        comparator = lambda a, b: a <= b + 1e-9
    else:
        comparator = lambda a, b: a >= b - 1e-9

    while comparator(current, end):
        values.append(current)
        current += step

    if schedule.as_integer:
        return [int(round(value)) for value in values] or [None]

    return [round(value, 10) for value in values] or [None]


def _format_parameter_value(value: Union[int, float]) -> str:
    if isinstance(value, int):
        return str(value)
    normalized = f"{value:.4f}".rstrip("0").rstrip(".")
    return normalized or "0"


def _prepare_attack_factory_config(attack: AttackConfig) -> Dict[str, Any]:
    if attack.factory_config:
        config = dict(attack.factory_config)
    else:
        config = {"method": attack.name, "parameters": {}}
    parameters = dict(config.get("parameters") or {})
    config["parameters"] = parameters
    return config


def _override_attack_parameter(
        factory_config: Mapping[str, Any], parameter_name: str, value: Any
) -> Dict[str, Any]:
    config = dict(factory_config)
    parameters = dict(config.get("parameters") or {})
    parameters[parameter_name] = value
    config["parameters"] = parameters
    return config


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
            ResultSender.send_log(
                "警告",
                f"无法读取扰动参数文件 {normalized}: {exc}",
            )
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

    # 不支持的类型
    if not isinstance(payload, Mapping):
        return None

    # 处理字典类型的payload
    enabled = bool(payload.get("enabled", True))
    method_name = payload.get("method") or name

    parameters_payload = payload.get("parameters")
    if isinstance(parameters_payload, Mapping):
        base_parameters, sweep_parameter, sweep_schedule = _split_attack_parameters(
            parameters_payload
        )
    else:
        base_parameters, sweep_parameter, sweep_schedule = {}, None, None

    # 处理工厂配置
    factory_payload = payload.get("factory_config")
    if isinstance(factory_payload, Mapping):
        parameters = dict(base_parameters)
        parameters.update(factory_payload.get("parameters", {}))
        factory_config = {
            "method": factory_payload.get("method", method_name),
            "parameters": parameters,
        }
    else:
        factory_config = {"method": method_name, "parameters": dict(base_parameters)}

    return AttackConfig(
        name=name,
        enabled=enabled,
        factory_config=factory_config,
        sweep_parameter=sweep_parameter,
        sweep_schedule=sweep_schedule,
    )


def _normalize_attack_declarations(attacks_payload: Any) -> List[AttackConfig]:
    """统一不同风格的攻击声明格式.

    Args:
        attacks_payload (Any): ``adversarial.attacks`` 键下的原始对象。

    Returns:
        List[AttackConfig]: 清洗后的攻击配置列表。
    """

    parsed: List[AttackConfig] = []

    if isinstance(attacks_payload, str):
        attack = _build_attack_config(attacks_payload, attacks_payload)
        if attack:
            parsed.append(attack)
        return parsed

    if isinstance(attacks_payload, Mapping):
        for name, payload in attacks_payload.items():
            attack = _build_attack_config(name, payload)
            if attack:
                parsed.append(attack)
        return parsed

    if isinstance(attacks_payload, Sequence) and not isinstance(attacks_payload, (str, bytes)):
        for payload in attacks_payload:
            if isinstance(payload, str):
                attack = _build_attack_config(payload, payload)
            elif isinstance(payload, Mapping):
                name = payload.get("name") or payload.get("method")
                if not name:
                    continue
                attack = _build_attack_config(name, payload)
            else:
                continue

            if attack:
                parsed.append(attack)
        return parsed

    raise ValueError("不支持的攻击配置格式")


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
            canonical = ALL_METRIC_ALIASES.get(normalized_name)
            if canonical:
                normalized.append(canonical)
        return tuple(normalized) if normalized else fallback

    # 处理字符串类型
    if isinstance(metrics_payload, str):
        normalized_name = metrics_payload.strip().lower()
        canonical = ALL_METRIC_ALIASES.get(normalized_name)
        if canonical:
            return (canonical,)

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
        batch = images[start: start + batch_size]
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
    """克隆图像，保持与标注相同的空间尺寸."""

    # 处理PyTorch张量
    if isinstance(image, torch.Tensor):
        return image.detach().clone()

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