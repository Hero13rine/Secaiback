"""用于评估检测模型对抗鲁棒性的入口点."""
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
    """单个对抗攻击评估的配置.

    Attributes:
        name (str): 攻击名称
        enabled (bool): 是否启用此攻击评估，默认为True
        rotations (Optional[Tuple[float, ...]]): 旋转角度列表，默认为None
        metrics (Optional[Tuple[str, ...]]): 要计算的指标列表，默认为None
        factory_config (Optional[Dict[str, Any]]): 攻击工厂配置，默认为None
    """

    name: str
    enabled: bool = True
    rotations: Optional[Tuple[float, ...]] = None
    metrics: Optional[Tuple[str, ...]] = None
    factory_config: Optional[Dict[str, Any]] = None


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
    use_rotated_iou: Optional[bool] = None,
    iou_threshold: float = 0.5,
    batch_size: int = 1,
) -> Dict[str, AttackEvaluationResult]:
    """使用AttackFactory驱动的攻击评估对抗鲁棒性.

    Args:
        estimator: 能够预测检测输出的估计器。当它暴露了``get_core``方法时，
                   返回的ART估计器将用于攻击生成。
        test_data: 单个数据加载器/可迭代对象，产生 ``(images, targets)`` 批次，
                   或者是从旋转角度到此类数据加载器的映射。
                   图像应该是与 ``estimator`` 的 ``predict`` 方法兼容的张量。
        config: 允许选择攻击、旋转和指标的解析后配置字典。
        config_path: 包含配置的YAML文件的可选路径。
        use_rotated_iou: 是否使用旋转IoU计算，如果为None则从配置中提取
        iou_threshold: 指标计算的IoU阈值。
        batch_size: 调用估计器进行预测时一起处理的样本数量。

    Returns:
        一个字典，将攻击名称映射到其鲁棒性评估结果。
    """

    # 如果提供了配置文件路径，则加载配置
    if config_path:
        config = load_robustness_config(config_path)
    config = config or {}
    if use_rotated_iou is None:
        use_rotated_iou = _extract_use_rotated_iou(config)
    if use_rotated_iou is None:
        raise ValueError("use_rotated_iou 参数是必需的，但在配置中未找到")

    print(f"进度: 开始处理数据集...")
    # 标准化测试数据格式
    data_by_rotation = _normalize_test_data(test_data)
    # 收集数据集
    dataset = _collect_dataset(data_by_rotation)
    if not dataset:
        raise ValueError("测试数据必须至少提供一个样本")
    print("进度", "数据集已加载")
    
    # 存储每个旋转角度的真实标签和基线预测结果
    ground_truths_by_rotation: Dict[float, List[PredictionLike]] = {}
    baseline_predictions: Dict[float, List[PredictionLike]] = {}

    # 对每个旋转角度的数据进行处理
    total_rotations = len(dataset)
    for idx, (rotation, (images, targets)) in enumerate(dataset.items(), 1):
        print(f"进度: 处理旋转角度 {rotation} ({idx}/{total_rotations})...")
        # 标准化真实标签
        ground_truths_by_rotation[rotation] = [_normalize_target(target) for target in targets]
        # 运行基线预测
        baseline_predictions[rotation] = _run_predictions(estimator, images, batch_size=batch_size)

    # 解析攻击配置
    print("进度: 解析攻击配置...")
    attack_configs = _parse_attack_configs(config)
    if not attack_configs:
        return {}

    # 创建对抗鲁棒性评估器实例
    evaluator = AdversarialRobustnessEvaluator(
        iou_threshold=iou_threshold,
        use_rotated_iou=use_rotated_iou,
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
        attack_predictions: Dict[float, List[PredictionLike]] = {}

        # 获取要使用的旋转角度
        rotations_to_use = attack.rotations or tuple(dataset.keys())
        total_rotations_for_attack = len(rotations_to_use)
        
        for rot_idx, rotation in enumerate(rotations_to_use, 1):
            if rotation not in dataset:
                raise KeyError(f"提供的数据中不包含旋转角度 {rotation}")
            
            print(f"  进度: 生成对抗样本 (旋转角度 {rotation}, {rot_idx}/{total_rotations_for_attack})...")
            
            # 获取原始图像
            images, _ = dataset[rotation]
            # 生成对抗图像
            adv_images = [_generate_adversarial_image(attack_instance, image) for image in images]
            # 对抗图像预测
            attack_predictions[rotation] = _run_predictions(
                estimator,
                adv_images,
                batch_size=batch_size,
            )

        # 过滤指定旋转角度的数据
        selected_baseline = _filter_rotations(baseline_predictions, attack.rotations)
        selected_ground_truths = _filter_rotations(ground_truths_by_rotation, attack.rotations)
        selected_attack = _filter_rotations(attack_predictions, attack.rotations)

        # 评估攻击效果
        print(f"  进度: 评估攻击效果...")
        result = evaluator.evaluate_attack(
            attack.name,
            selected_baseline,
            selected_attack,
            selected_ground_truths,
            attack.metrics,
        )
        results[attack.name] = result
        print(f"进度: 攻击 '{attack.name}' 处理完成")

    print("进度: 所有攻击处理完成")
    return results


def _parse_attack_configs(config: Mapping[str, Any]) -> List[AttackConfig]:
    """将分层的YAML负载转换为攻击选择.

    Args:
        config (Mapping[str, Any]): 配置字典

    Returns:
        List[AttackConfig]: 解析后的攻击配置列表
    """

    if not isinstance(config, Mapping):
        return []

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
        return []

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
                metrics=default_metrics,
                factory_config={"method": "fgsm", "parameters": {}},
            )
        ]

    if not isinstance(adversarial_section, Mapping):
        return []

    # 提取默认指标和旋转角度
    default_metrics = _extract_metric_list(
        adversarial_section.get("default_metrics") or adversarial_section.get("metrics")
    )
    default_rotations = _normalize_rotations(adversarial_section.get("default_rotations"))
    attacks_payload = adversarial_section.get("attacks")

    # 如果没有明确指定攻击，则根据默认指标创建FGSM攻击
    if attacks_payload is None:
        # 允许只指定默认指标但不显式指定攻击的配置
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

    # 解析攻击配置
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
        raise ValueError("不支持的攻击配置格式")

    return [attack for attack in parsed if attack.enabled]


def _extract_use_rotated_iou(config: Optional[Mapping[str, Any]]) -> bool:
    """Read the rotated IoU requirement from the model section."""

    if not isinstance(config, Mapping):
        return False

    model_section = config.get("model")
    if not isinstance(model_section, Mapping):
        return False

    direct_flag = model_section.get("use_rotated_iou")
    if isinstance(direct_flag, bool):
        return direct_flag

    instantiation = model_section.get("instantiation")
    if isinstance(instantiation, Mapping):
        instantiation_flag = instantiation.get("use_rotated_iou")
        if isinstance(instantiation_flag, bool):
            return instantiation_flag

        instantiation_params = instantiation.get("parameters")
        if isinstance(instantiation_params, Mapping):
            param_flag = instantiation_params.get("use_rotated_iou")
            if isinstance(param_flag, bool):
                return param_flag

    for key in ("base", "basic", "base_parameters", "parameters"):
        nested = model_section.get(key)
        if isinstance(nested, Mapping) and "use_rotated_iou" in nested:
            flag = nested.get("use_rotated_iou")
            if isinstance(flag, bool):
                return flag

    return False


def _build_attack_config(
    name: str,
    payload: Any,
    default_metrics: Optional[Tuple[str, ...]],
    default_rotations: Optional[Tuple[float, ...]],
) -> Optional[AttackConfig]:
    """构建攻击配置对象
    
    Args:
        name (str): 攻击名称
        payload (Any): 攻击配置载荷
        default_metrics (Optional[Tuple[str, ...]]): 默认指标
        default_rotations (Optional[Tuple[float, ...]]): 默认旋转角度

    Returns:
        Optional[AttackConfig]: 构建的攻击配置对象
    """
    
    # 处理布尔类型的payload
    if isinstance(payload, bool):
        return AttackConfig(
            name=name,
            enabled=payload,
            metrics=default_metrics,
            rotations=default_rotations,
            factory_config={"method": name, "parameters": {}},
        )

    # 处理None类型的payload
    if payload is None:
        return AttackConfig(
            name=name,
            metrics=default_metrics,
            rotations=default_rotations,
            factory_config={"method": name, "parameters": {}},
        )

    # 处理字符串类型的payload
    if isinstance(payload, str):
        method_name = payload.strip() or name
        return AttackConfig(
            name=name,
            metrics=default_metrics,
            rotations=default_rotations,
            factory_config={"method": method_name, "parameters": {}},
        )

    # 处理序列类型的payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        metrics = _extract_metric_list(payload, default_metrics)
        return AttackConfig(
            name=name,
            metrics=metrics,
            rotations=default_rotations,
            factory_config={"method": name, "parameters": {}},
        )

    # 不支持的类型
    if not isinstance(payload, Mapping):
        return None

    # 处理字典类型的payload
    enabled = bool(payload.get("enabled", True))
    rotations = _normalize_rotations(payload.get("rotations")) or default_rotations
    metrics = _extract_metric_list(
        payload.get("metrics") or payload.get("outputs"), default_metrics
    )
    method_name = payload.get("method") or name
    parameters_payload = payload.get("parameters") or {}
    parameters = dict(parameters_payload) if isinstance(parameters_payload, Mapping) else {}

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
        rotations=rotations,
        metrics=metrics,
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


def _normalize_rotations(rotations: Any) -> Optional[Tuple[float, ...]]:
    """标准化旋转角度
    
    Args:
        rotations (Any): 旋转角度配置

    Returns:
        Optional[Tuple[float, ...]]: 标准化后的旋转角度元组
    """
    
    # 处理None值
    if rotations is None:
        return None
    
    # 处理数值类型
    if isinstance(rotations, (int, float)):
        return (float(rotations),)
    
    # 处理序列类型
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
    """过滤指定旋转角度的数据
    
    Args:
        mapping (Mapping[float, Sequence[PredictionLike]]): 原始数据映射
        rotations (Optional[Tuple[float, ...]]): 要过滤的旋转角度

    Returns:
        Dict[float, Sequence[PredictionLike]]: 过滤后的数据
    """
    
    # 如果未指定旋转角度，则返回所有数据
    if not rotations:
        return dict(mapping)
    
    # 过滤指定旋转角度的数据
    filtered: Dict[float, Sequence[PredictionLike]] = {}
    for rotation in rotations:
        if rotation not in mapping:
            raise KeyError(f"提供的数据中不包含旋转角度 {rotation}")
        filtered[rotation] = mapping[rotation]
    
    return filtered


def _normalize_test_data(
    test_data: Union[Iterable, Mapping[Any, Iterable]]
) -> Dict[float, Iterable]:
    """标准化测试数据格式
    
    Args:
        test_data (Union[Iterable, Mapping[Any, Iterable]]): 原始测试数据

    Returns:
        Dict[float, Iterable]: 标准化后的测试数据
    """
    
    # 如果是字典类型，则尝试转换键为浮点数
    if isinstance(test_data, Mapping):
        normalized: Dict[float, Iterable] = {}
        for key, loader in test_data.items():
            try:
                rotation = float(key)
            except (TypeError, ValueError):
                continue
            normalized[rotation] = loader
        return normalized
    
    # 否则默认旋转角度为0.0
    return {0.0: test_data}


def _collect_dataset(
    data_by_rotation: Mapping[float, Iterable]
) -> Dict[float, Tuple[List[torch.Tensor], List[Mapping[str, Any]]]]:
    """收集数据集
    
    Args:
        data_by_rotation (Mapping[float, Iterable]): 按旋转角度组织的数据

    Returns:
        Dict[float, Tuple[List[torch.Tensor], List[Mapping[str, Any]]]]: 收集的数据集
    """
    
    dataset: Dict[float, Tuple[List[torch.Tensor], List[Mapping[str, Any]]]] = {}
    
    # 遍历每个旋转角度的数据
    for rotation, loader in data_by_rotation.items():
        images: List[torch.Tensor] = []
        targets: List[Mapping[str, Any]] = []
        
        # 从数据加载器中提取图像和目标
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
    "RotationRobustnessMetrics",
    "evaluate_adversarial_robustness",
    "load_robustness_config",
]