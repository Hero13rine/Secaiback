"""Entry point for running an end-to-end detection evaluation smoke test."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader

from estimator import EstimatorFactory
from method.load_config import load_config
from data.dummy_detection_dataset import DummyDetectionDataset
from metric.object_detection.basic.detection import cal_object_detection


def _build_model(instantiation_config: dict) -> torch.nn.Module:
    """Instantiate a detection model defined in the configuration."""

    module_path = instantiation_config.get("model_module")
    class_name = instantiation_config.get("model_class")
    parameters = instantiation_config.get("parameters") or {}

    if not module_path or not class_name:
        raise ValueError("model_module and model_class must be provided in the configuration")

    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    return model_cls(**parameters)


def _collate_detection(batch: Sequence):
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass
class _DirectDetectionEstimator:
    """Minimal estimator that forwards to the torch model for predictions."""

    model: torch.nn.Module

    def predict(self, images: Iterable[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            image_list = list(images)
            return self.model(image_list)


def _build_estimator(
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    estimator_config: dict,
):
    """Create an estimator and gracefully fall back to the direct wrapper."""

    try:
        return EstimatorFactory.create(
            model=model,
            loss=loss,
            optimizer=optimizer,
            config=estimator_config,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"警告: 无法根据配置构建估计器，改为直接推理 ({exc})")
        return _DirectDetectionEstimator(model=model)


def main(config_path: str = "config/user/model_pytorch_det.yaml") -> None:
    """Run the smoke pipeline using the dummy dataset and detection metrics."""

    config = load_config(config_path)
    model_config = config["model"]["instantiation"]
    estimator_config = config["model"].get("estimator", {})
    evaluation_config = config.get("evaluation", {}) or {}

    model = _build_model(model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

    estimator = _build_estimator(model, loss, optimizer, estimator_config)

    dataset = DummyDetectionDataset()
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_detection)

    print("开始执行检测流程测试...")
    cal_object_detection(estimator, data_loader, evaluation_config)
    print("检测流程测试完成。")


if __name__ == "__main__":
    main()