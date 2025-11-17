"""展示检测模型鲁棒性评测的总入口脚本。"""
"""Entry script for end-to-end robustness evaluation of detection models."""
from __future__ import annotations

import json
from typing import Any, Mapping

import torch

from estimator import EstimatorFactory
from metric.object_detection.robustness import (
    AttackEvaluationResult,
    CorruptionEvaluationResult,
    evaluation_robustness,
)
from method import load_config
from model import load_model
from fasterrcnn_test.load_dataset import load_data


def _prepare_robustness_payload(config: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation_config = config.get("evaluation") if isinstance(config, Mapping) else None
    if isinstance(evaluation_config, Mapping) and "robustness" in evaluation_config:
        return evaluation_config

    if "robustness" in config:
        return config

    return {
        "robustness": {
            "adversarial": {
                "metrics": [
                    "map_drop_rate",
                    "miss_rate",
                    "false_detection_rate",
                ],
                "attacks": ["fgsm"],
            },
            "corruption": {
                "mertics": [
                    "perturbation_magnitude",
                    "performance_drop_rate",
                    "perturbation_tolerance",
                ],
                "corruptions": ["gaussian_noise"],
            },
        }
    }


def _format_adversarial_results(
    results: Mapping[str, AttackEvaluationResult]
) -> Mapping[str, Any]:
    payload = []
    for attack_name, result in results.items():
        metrics = result.metrics
        per_class_clean = dict(metrics.per_class_clean_map)
        per_class_adv = dict(metrics.per_class_adversarial_map)
        per_class_drop = dict(metrics.map_drop_rate_cls)
        payload.append(
            {
                "attack_name": attack_name,
                "metrics": {
                    "map_drop_rate": metrics.map_drop_rate,
                    "miss_rate": metrics.miss_rate,
                    "false_detection_rate": metrics.false_detection_rate,
                    "per_class_clean_map": per_class_clean,
                    "per_class_adversarial_map": per_class_adv,
                    "map_drop_rate_cls": per_class_drop,
                },
            }
        )

        print(f"\n=== 攻击: {attack_name} ===")
        print(
            "整体指标 - mAP下降率: {0:.4f}, 漏检率: {1:.4f}, 误检率: {2:.4f}".format(
                metrics.map_drop_rate,
                metrics.miss_rate,
                metrics.false_detection_rate,
            )
        )

        if per_class_clean:
            print("  Clean mAP (per class):")
            for label, value in sorted(per_class_clean.items()):
                print(f"    class {label}: {value:.4f}")
        if per_class_adv:
            print("  Adversarial mAP (per class):")
            for label, value in sorted(per_class_adv.items()):
                print(f"    class {label}: {value:.4f}")
        if per_class_drop:
            print("  mAP drop rate by class:")
            for label, value in sorted(per_class_drop.items()):
                print(f"    class {label}: {value:.4f}")

    return {"attacks": payload}


def _format_corruption_results(
    results: Mapping[str, CorruptionEvaluationResult]
) -> Mapping[str, Any]:
    payload = []
    for corruption_name, result in results.items():
        metrics = result.metrics
        payload.append(
            {
                "corruption_name": result.corruption_name,
                "severity": result.severity,
                "metrics": {
                    "perturbation_magnitude": metrics.perturbation_magnitude,
                    "performance_drop_rate": metrics.performance_drop_rate,
                    "perturbation_tolerance": metrics.perturbation_tolerance,
                },
            }
        )

        print(f"\n=== 扰动: {corruption_name} ===")
        print(
            "整体指标 - 扰动幅度: {0:.4f}, 性能下降率: {1:.4f}, 扰动容忍度: {2:.4f}".format(
                metrics.perturbation_magnitude,
                metrics.performance_drop_rate,
                metrics.perturbation_tolerance,
            )
        )

    return {"corruptions": payload}


def _print_results(
    adversarial_results: Mapping[str, AttackEvaluationResult],
    corruption_results: Mapping[str, CorruptionEvaluationResult],
) -> None:
    if not adversarial_results and not corruption_results:
        message = "鲁棒性配置中未启用任何攻击或扰动评估。"
        print(message)
        print(json.dumps({"message": message, "attacks": [], "corruptions": []}, ensure_ascii=False, indent=2))
        return

    json_payload: Mapping[str, Any] = {"attacks": [], "corruptions": []}
    if adversarial_results:
        json_payload = {
            **json_payload,
            **_format_adversarial_results(adversarial_results),
        }

    if corruption_results:
        json_payload = {
            **json_payload,
            **_format_corruption_results(corruption_results),
        }

    print("\n鲁棒性评测JSON结果：")
    print(json.dumps(json_payload, ensure_ascii=False, indent=2))



def main(
    model_config_path: str = "config/user/model_pytorch_det_fasterrcnn_robustness.yaml",
) -> None:
    print("进度: 开始执行完整鲁棒性评估测试...")

    print("进度: 加载配置文件...")
    user_config = load_config(model_config_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    print("进度", "配置文件已加载完毕")

    print("进度: 初始化模型...")
    model = load_model(
        model_instantiation_config["model_path"],
        model_instantiation_config["model_name"],
        model_instantiation_config["weight_path"],
        model_instantiation_config["parameters"],
    )
    print("进度", "模型初始化完成")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss = None

    print("进度: 创建估计器...")
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config,
    )

    print("进度: 加载测试数据...")
    test_loader = load_data("fasterrcnn_test/test")
    print("进度", "数据集已加载")

    print("进度: 准备鲁棒性评估配置...")
    robustness_payload = _prepare_robustness_payload(user_config)

    print("进度: 执行统一鲁棒性评估...")
    results = evaluation_robustness(
        estimator=estimator,
        test_data=test_loader,
        config=robustness_payload,
        batch_size=64,
    )
    _print_results(
        adversarial_results=results.get("adversarial", {}),
        corruption_results=results.get("corruption", {}),
    )

    print("检测流程测试完成。")
    print("进度: 完整鲁棒性评估测试完成。")


if __name__ == "__main__":
    main()