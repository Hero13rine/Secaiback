"""Lightweight dummy object detection model for tests and smoke runs."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DummyObjectDetectionModel(nn.Module):
    """A minimal detector that returns deterministic predictions.

    The module accepts a list of tensors in the same spirit as torchvision's
    detection models and produces one predicted box per image. The box sizes are
    based on the input image spatial dimensions so downstream code that
    inspects bounding boxes receives reasonable values.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        # Register a parameter so optimizers can attach to the module during tests.
        self.logit_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, images: List[torch.Tensor]):  # type: ignore[override]
        outputs = []
        bias_score = self.logit_bias.sigmoid() + 0.5
        for idx, image in enumerate(images):
            if image.dim() != 3:
                raise ValueError("Expected images with shape [C, H, W]")
            _, height, width = image.shape
            device = image.device
            dtype = image.dtype

            # Provide simple but valid bounding boxes within the image bounds.
            x1 = 0.0
            y1 = 0.0
            x2 = max(float(width) - 1.0, 0.0)
            y2 = max(float(height) - 1.0, 0.0)
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=dtype, device=device)

            # Alternate confidence scores to exercise thresholding logic.
            base_score = 0.9 if idx % 2 else 0.2
            scores = torch.tensor([base_score], dtype=dtype, device=device) * bias_score.to(dtype)

            # Cycle label ids starting at 1 to imitate real detectors.
            label_id = (idx % self.num_classes) + 1
            labels = torch.tensor([label_id], dtype=torch.int64, device=device)

            outputs.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        return outputs