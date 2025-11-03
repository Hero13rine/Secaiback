"""Synthetic dataset for exercising the detection pipeline."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class DummyDetectionDataset(Dataset):
    """Produces random images and simple bounding boxes."""

    def __init__(
        self,
        num_samples: int = 8,
        image_size: Tuple[int, int, int] = (3, 64, 64),
        boxes_per_image: int = 1,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if boxes_per_image < 1:
            raise ValueError("boxes_per_image must be positive")
        if len(image_size) != 3:
            raise ValueError("image_size must be a 3-tuple (C, H, W)")
        self.num_samples = int(num_samples)
        self.image_size = tuple(int(x) for x in image_size)
        self.boxes_per_image = int(boxes_per_image)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def __getitem__(self, index: int):  # pragma: no cover - exercised indirectly
        c, h, w = self.image_size
        image = torch.rand(c, h, w, dtype=torch.float32)
        boxes = []
        labels = []
        for box_idx in range(self.boxes_per_image):
            x1 = float(box_idx)
            y1 = float(box_idx)
            x2 = min(float(w - 1), x1 + float(w // 2))
            y2 = min(float(h - 1), y1 + float(h // 2))
            boxes.append([x1, y1, x2, y2])
            labels.append((box_idx % 3) + 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target
