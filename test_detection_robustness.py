"""Entry script showcasing adversarial robustness evaluation for detection models."""
from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader

from data.dummy_detection_dataset import DummyDetectionDataset
from estimator import EstimatorFactory
from metric.object_detection.robustness import (
    AttackEvaluationResult,
    evaluate_adversarial_robustness,
)
from method.load_config import load_config


def _build_model(instantiation_config: Mapping[str, object]) -> torch.nn.Module:
    module_name = instantiation_config.get("model_module", "model.net.dummy_detector")
    class_name = instantiation_config.get("model_class", "DummyObjectDetectionModel")
    module = importlib.import_module(str(module_name))
    model_cls = getattr(module, str(class_name))
    parameters = instantiation_config.get("parameters") or {}
    return model_cls(**parameters)


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
        print("No attacks were enabled in the robustness configuration.")
        return
    for attack_name, result in results.items():
        print(f"\n=== {attack_name} ===")
        overall = result.overall
        print(
            "Overall - mAP drop: {0:.4f}, miss rate: {1:.4f}, false detection rate: {2:.4f}".format(
                overall.map_drop_rate,
                overall.miss_rate,
                overall.false_detection_rate,
            )
        )
        for rotation, metrics in result.by_rotation.items():
            print(
                "  Rotation {0}: mAP drop {1:.4f}, miss rate {2:.4f}, false detection rate {3:.4f}".format(
                    rotation,
                    metrics.map_drop_rate,
                    metrics.miss_rate,
                    metrics.false_detection_rate,
                )
            )


def main(
    model_config_path: str = "config/user/model_pytorch_det_fasterrcnn.yaml",
    num_workers: int = 0,
) -> None:
    config = load_config(model_config_path)
    model_config = config["model"]["instantiation"]
    estimator_config = config["model"]["estimator"]

    model = _build_model(model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=estimator_config,
    )

    dataset = DummyDetectionDataset(num_samples=4, boxes_per_image=2)
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_detection,
    )

    robustness_payload = _prepare_robustness_payload(config)

    print("Running adversarial robustness evaluation...")
    results = evaluate_adversarial_robustness(
        estimator=estimator,
        test_data={0.0: data_loader},
        config=robustness_payload,
        batch_size=2,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
