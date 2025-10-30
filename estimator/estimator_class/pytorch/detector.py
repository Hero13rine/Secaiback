"""PyTorch estimator wrapper for object detection models."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from estimator.estimator_class.base_estimator import BaseEstimator
from estimator.estimator_factory import EstimatorFactory


@EstimatorFactory.register(framework="pytorch", task="object_detection")
class PyTorchObjectDetectionWrapper(BaseEstimator):
    """Wraps a PyTorch detection model to provide a unified predict interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: Any,
        device: str = "auto",
        score_threshold: float = 0.0,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.optimizer = optimizer
        self.loss = loss
        self.score_threshold = float(score_threshold)

    def predict(self, batch) -> List[dict]:
        self.model.eval()
        images = self._prepare_inputs(batch)
        with torch.no_grad():
            outputs = self.model(images)

        if isinstance(outputs, dict):
            outputs = [outputs]
        results = []
        for output in outputs:
            boxes = output.get("boxes")
            scores = output.get("scores")
            labels = output.get("labels")

            boxes = self._tensor_to_numpy(boxes)
            scores = self._tensor_to_numpy(scores)
            labels = self._tensor_to_numpy(labels, dtype=np.int64)

            if boxes.size == 0:
                results.append({"boxes": boxes.reshape(-1, 4), "scores": scores.reshape(-1), "labels": labels.reshape(-1)})
                continue

            mask = np.ones_like(scores, dtype=bool)
            if self.score_threshold > 0:
                mask = scores >= self.score_threshold

            filtered_boxes = boxes.reshape(-1, 4)[mask]
            filtered_scores = scores.reshape(-1)[mask]
            filtered_labels = labels.reshape(-1)[mask]

            results.append(
                {
                    "boxes": filtered_boxes,
                    "scores": filtered_scores,
                    "labels": filtered_labels,
                }
            )
        return results

    def get_core(self):
        return self.model

    def _prepare_inputs(self, batch) -> List[torch.Tensor]:
        if isinstance(batch, torch.Tensor):
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            return [img.to(self.device) for img in batch]
        if isinstance(batch, (list, tuple)):
            return [self._ensure_tensor(img) for img in batch]
        return [self._ensure_tensor(batch)]

    def _ensure_tensor(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image.to(self.device)
        return torch.as_tensor(image, device=self.device)

    @staticmethod
    def _tensor_to_numpy(tensor, dtype=None) -> np.ndarray:
        if tensor is None:
            return np.zeros((0,), dtype=dtype or np.float32)
        if isinstance(tensor, np.ndarray):
            array = tensor
        else:
            array = tensor.detach().cpu().numpy()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array
