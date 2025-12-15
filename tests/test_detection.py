"""Entry point for running robustness smoke tests on detection models."""
from __future__ import annotations

from typing import Sequence
from torch import optim
import torch
from torch.utils.data import DataLoader

from estimator import EstimatorFactory
from utils.load_config import load_config
from data.dummy_detection_dataset import DummyDetectionDataset


def _build_model(instantiation_config: dict) -> torch.nn.Module:
    # 直接使用dummy_detector中的DummyObjectDetectionModel作为测试模型
    from model.net.dummy_detector import DummyObjectDetectionModel
    parameters = instantiation_config.get("parameters") or {}
    return DummyObjectDetectionModel(**parameters)

def _collate_detection(batch: Sequence):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main(config_path: str = "config/user/model_pytorch_det.yaml") -> None:
    config = load_config(config_path)
    model_config = config["model"]["instantiation"]
    estimator_config = config["model"]["estimator"]
    evaluation_config = config.get("evaluation", {})

    model = _build_model(model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()

    estimator = EstimatorFactory.create(

        model=model,
        loss=loss,
        optimizer=optimizer,
        config=estimator_config,
    )


    dataset = DummyDetectionDataset()
    batch_size = 2
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