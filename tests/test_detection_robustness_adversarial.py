"""展示检测模型的对抗鲁棒性评估的条目脚本。"""
from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from estimator import EstimatorFactory
from metric.object_detection.robustness import (
    AttackEvaluationResult,
    evaluate_adversarial_robustness,
)
from utils import load_config
from model import load_model
from fasterrcnn_test.load_dataset import load_data


def _collate_detection(batch: Sequence):
    images, targets = zip(*batch)
    return list(images), list(targets)


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
            }
        }
    }


def _print_results(results: Mapping[str, AttackEvaluationResult]) -> None:
    if not results:
        message = "鲁棒性配置中未启用任何对抗攻击。"
        print(message)
        print(json.dumps({"attacks": [], "message": message}, ensure_ascii=False, indent=2))
        return

    payload = []
    for attack_name, result in results.items():
        metrics = result.metrics
        per_class_clean = dict(metrics.per_class_clean_map)
        per_class_adv = dict(metrics.per_class_adversarial_map)
        # per_class_drop = dict(metrics.map_drop_rate_cls)  # This property doesn't exist either
        
        # Calculate the increase values instead of accessing them as properties
        miss_rate_increase = metrics.miss_rate - metrics.clean_miss_rate
        false_detection_rate_increase = metrics.false_detection_rate - metrics.clean_false_detection_rate

        attack_payload: Dict[str, Any] = {"attack": result.metadata.get("attack", attack_name)}
        for key, value in result.metadata.items():
            if key == "attack":
                continue
            attack_payload[key] = value
        attack_payload["metrics"] = {
            "clean_map": metrics.clean_map,
            "adversarial_map": metrics.adversarial_map,
            "map_drop_rate": metrics.map_drop_rate,
            "miss_rate": metrics.miss_rate,
            "false_detection_rate": metrics.false_detection_rate,
            "clean_miss_rate": metrics.clean_miss_rate,
            "clean_false_detection_rate": metrics.clean_false_detection_rate,
            "miss_rate_increase": metrics.miss_rate_increase,
            "false_detection_rate_increase": metrics.false_detection_rate_increase,
            "per_class_clean_map": per_class_clean,
            "per_class_adversarial_map": per_class_adv,
        }
        payload.append(attack_payload)

        print(f"\n=== 攻击: {attack_name} ===")
        print(
            "整体指标 - Clean mAP: {0:.4f}, Adv mAP: {1:.4f}, mAP下降率: {2:.4f}".format(
                metrics.clean_map,
                metrics.adversarial_map,
                metrics.map_drop_rate,
            )
        )
        print(
            "  漏检率 Clean: {0:.4f} → Adv: {1:.4f} (Δ{2:.4f})".format(
                metrics.clean_miss_rate,
                metrics.miss_rate,
                miss_rate_increase,
            )
        )
        print(
            "  误检率 Clean: {0:.4f} → Adv: {1:.4f} (Δ{2:.4f})".format(
                metrics.clean_false_detection_rate,
                metrics.false_detection_rate,
                false_detection_rate_increase,
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

    print("\n对抗攻击评测JSON结果：")
    print(json.dumps({"attacks": payload}, ensure_ascii=False, indent=2))


def main(
    model_config_path: str = "config/user/model_pytorch_det_fasterrcnn_adversarial.yaml",
    num_workers: int = 0,
) -> None:
    print("进度: 开始执行对抗鲁棒性评估-对抗共攻击测试...")

    # 1.加载配置文件
    print("进度: 加载配置文件...")
    user_config = load_config(model_config_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    print ("进度", "配置文件已加载完毕")

    print("进度: 初始化模型...")
    model = load_model(model_instantiation_config["model_path"], model_instantiation_config["model_name"]
                       , model_instantiation_config["weight_path"], model_instantiation_config["parameters"])
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

    # 5.加载数据
    print("进度: 加载测试数据...")
    test_loader = load_data("../fasterrcnn_test/test")
    print("进度", "数据集已加载")

    # 6.根据传入的评测类型进行评测
    print("开始执行检测流程测试...")

    print("进度: 准备鲁棒性评估配置...")
    robustness_payload = _prepare_robustness_payload(user_config)

    print("Running adversarial robustness evaluation...")
    results = evaluate_adversarial_robustness(
        estimator=estimator,
        test_data=test_loader,
        config=robustness_payload,
        batch_size=64,
    )
    _print_results(results)

    print("检测流程测试完成。")
    print("进度: 对抗鲁棒性评估测试完成。")
if __name__ == "__main__":
    main()