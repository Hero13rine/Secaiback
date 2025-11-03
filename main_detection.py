"""Entry point for running robustness smoke tests on detection models."""
from __future__ import annotations

import importlib
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from estimator import EstimatorFactory
from method.load_config import load_config
from data.dummy_detection_dataset import DummyDetectionDataset


def _build_model(instantiation_config: dict) -> torch.nn.Module:
    module_path = instantiation_config["model_module"]
    class_name = instantiation_config["model_class"]
    parameters = instantiation_config.get("parameters") or {}

    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"Module {module_path} has no attribute {class_name}")
    model_cls = getattr(module, class_name)
    return model_cls(**parameters)


def _collate_detection(batch: Sequence):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main(config_path: str = "config/user/model_pytorch_det.yaml") -> None:
    config = load_config(config_path)
    model_config = config["model"]["instantiation"]
    estimator_config = config["model"]["estimator"]
    data_config = config.get("data", {})
    evaluation_config = config.get("evaluation", {})

    model = _build_model(model_config)
    optimizer = None
    loss = None

    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=estimator_config,
    )

    dataset_params = data_config.get("parameters", {})
    dataset = DummyDetectionDataset(**dataset_params)
    batch_size = data_config.get("batch_size", 2)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_detection)

    max_steps = evaluation_config.get("steps")
    for step, batch in enumerate(data_loader, 1):
        images, targets = batch
        predictions = estimator.predict(images)
        total_boxes = sum(len(pred.get("boxes", [])) for pred in predictions)
        print(f"Step {step}: processed {len(images)} images, predicted {total_boxes} boxes")
        if max_steps is not None and step >= max_steps:
            break


if __name__ == "__main__":
    main()
