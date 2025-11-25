from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


def _find_image_files(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def _guess_label_path(img_path: Path) -> Path:
    """
    Try common YOLO layouts:
      - <root>/images/.../<name>.jpg  → <root>/labels/.../<name>.txt
      - <root>/.../<name>.jpg         → same folder with .txt
    """
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts[:]
        label_parts[idx] = "labels"
        return Path(*label_parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def _yolo_to_xyxy(box: List[float], w: int, h: int) -> Tuple[float, float, float, float]:
    """
    Convert YOLO normalized [cx, cy, bw, bh] to pixel xyxy.
    """
    cx, cy, bw, bh = box
    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(w), cx + bw / 2.0)
    y2 = min(float(h), cy + bh / 2.0)
    return x1, y1, x2, y2


class ObjDataset(Dataset):
    """
    Minimal YOLO-format detection dataset loader.
    Each image has a .txt with lines: <class_id> cx cy bw bh (normalized 0~1).
    Returns labels as 1-based (background=0) to match TorchVision convention.
    """

    def __init__(self, root: str | Path, *, augment: bool = False):
        self.root = Path(root)
        self.augment = augment
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} does not exist")

        self.images = _find_image_files(self.root)
        if not self.images:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def _read_labels(self, label_path: Path, w: int, h: int) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes_xyxy = []
        labels_1based = []

        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1, y1, x2, y2 = _yolo_to_xyxy([cx, cy, bw, bh], w, h)
                    if x2 > x1 and y2 > y1:
                        boxes_xyxy.append([x1, y1, x2, y2])
                        labels_1based.append(cls + 1)

        if boxes_xyxy:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels = torch.tensor(labels_1based, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return boxes, labels

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = _guess_label_path(img_path)
        boxes, labels = self._read_labels(label_path, w, h)

        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - x2  # new x1
                boxes[:, 2] = w - x1  # new x2

        img_t = TF.to_tensor(img)  # [0,1] float32

        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel() > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def load_data(
    train_root: str | Path | None = None,
    val_root: str | Path | None = None,
    test_root: str | Path | None = None,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 2,
    augment: bool = False,
):
    """
    Create DataLoader objects for YOLO-format detection datasets.

    When only ``train_root`` is provided the function preserves the original
    single-loader behavior. If no roots are provided, the DIOR default paths
    are used and a tuple of ``(train_loader, val_loader, test_loader)`` is
    returned to align with the ``eval_start`` example.
    """

    def _build(root_path: str | Path) -> DataLoader:
        dataset = ObjDataset(root_path, augment=augment)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if train_root is None and val_root is None and test_root is None:
        train_root = Path("data/dataset/dior/train")
        val_root = Path("data/dataset/dior/val")
        test_root = Path("data/dataset/dior/test")

    if val_root is None and test_root is None and train_root is not None:
        return _build(train_root)

    train_loader = _build(train_root or Path("data/dataset/dior/train"))
    val_loader = _build(val_root or Path("data/dataset/dior/val"))
    test_loader = _build(test_root or Path("data/dataset/dior/test"))
    return train_loader, val_loader, test_loader
