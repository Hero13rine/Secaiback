+100
-0

"""Smoke test for the corruption-robustness evaluation pipeline.

The script mirrors ``test_fasterrcnn.py`` by wiring together a lightweight model,
a dummy dataset and the :func:`evaluate_corruption_robustness` entrypoint.
It is intentionally simple and focuses on validating that the evaluation flow
can be executed without relying on the larger benchmarking harness.
"""


import pprint
from typing import Iterable, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from data.dummy_detection_dataset import DummyDetectionDataset
from metric.object_detection.robustness.evaluate_robustness import (
    evaluate_corruption_robustness,
)
from model.net.dummy_detector import DummyObjectDetectionModel


class DummyEstimator:
    """Minimal estimator wrapper that exposes the ``predict`` API."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()

    def predict(self, inputs: torch.Tensor | Sequence[torch.Tensor]):
        """Adapt stacked tensors to the dummy detector input contract."""

        if isinstance(inputs, torch.Tensor):
            batch = [inputs[i] for i in range(inputs.shape[0])]
        else:
            batch = list(inputs)

        with torch.no_grad():
            return self.model(batch)


def _collate_detection_batch(batch: Iterable[Tuple[torch.Tensor, dict]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main() -> None:
    torch.manual_seed(42)

    dataset = DummyDetectionDataset(num_samples=6)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_detection_batch)

    model = DummyObjectDetectionModel(num_classes=3)
    estimator = DummyEstimator(model)

    corruption_config = {
        "evaluation": {
            "robustness": {
                "corruption": {
                    "default_metrics": [
                        "perturbation_magnitude",
                        "performance_drop_rate",
                        "perturbation_tolerance",
                    ],
                    "default_severities": [1, 2],
                    "corruptions": {
                        "gaussian_noise": {
                            "method": "gaussian_noise",
                            "severities": [1, 3],
                            "parameters": {"magnitude": 0.05},
                        },
                        "brightness_shift": {
                            "method": "brightness_shift",
                            "severities": [2],
                            "metrics": ["performance_drop_rate"],
                        },
                    },
                }
            }
        }
    }

    print("进度: 启动自然扰动鲁棒性流程测试...")
    results = evaluate_corruption_robustness(
        estimator=estimator,
        test_data=loader,
        config=corruption_config,
        iou_threshold=0.5,
        batch_size=2,
    )

    print("进度: 扰动鲁棒性评测完成，得到以下结果:")
    pprint.pprint(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - manual execution helper
        print(f"扰动鲁棒性测试失败: {exc}")
        raise