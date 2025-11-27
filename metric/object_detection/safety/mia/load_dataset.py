from __future__ import annotations
"""
Unified data loading module for MIA detection project.
This is the ONLY source for dataset loading and path management.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


# ============================================================================
# DEFAULT DATA PATHS - Centralized data path configuration
# ============================================================================
DEFAULT_TRAIN_DIR = '/public/yolo/dior/train'  # Target model's training data
DEFAULT_VAL_DIR = '/public/yolo/dior/val'      # Shadow model's non-member samples
DEFAULT_TEST_DIR = '/public/yolo/dior/test'    # Shadow model's training data


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
    root: str | Path = './data/dataset/dior/test',
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 2,
    augment: bool = False
) -> DataLoader:
    """
    Create a DataLoader for evaluation/testing (or quick sanity-check training).
    Args:
        root: dataset root directory; allows <root>/images and <root>/labels, or flat layout (*.jpg + *.txt).
        batch_size: batch size.
        shuffle: shuffle samples (commonly False for test/val).
        num_workers: number of workers for loading.
        augment: enable simple augmentation (currently horizontal flip).
    Returns:
        test_dataloader: torch.utils.data.DataLoader
    """
    dataset = ObjDataset(root, augment=augment)
    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return test_dataloader


def load_data_mia(
    train_root: str | Path = None,
    val_root: str | Path = None,
    test_root: str | Path = None,
    batch_size: int = 2,
    num_workers: int = 2,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load train/val/test datasets for MIA (Membership Inference Attack) experiments.

    The new DIOR dataset structure:
        - ./data/dataset/dior/train  : training images and labels
        - ./data/dataset/dior/val    : validation images and labels
        - ./data/dataset/dior/test   : test images and labels

    Args:
        train_root: path to training set directory (None = use DEFAULT_TRAIN_DIR)
        val_root: path to validation set directory (None = use DEFAULT_VAL_DIR)
        test_root: path to test set directory (None = use DEFAULT_TEST_DIR)
        batch_size: batch size for all dataloaders
        num_workers: number of workers for data loading
        augment_train: whether to apply augmentation to training data

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    # Use default paths if not provided
    train_root = train_root or DEFAULT_TRAIN_DIR
    val_root = val_root or DEFAULT_VAL_DIR
    test_root = test_root or DEFAULT_TEST_DIR

    # Create datasets
    train_dataset = ObjDataset(train_root, augment=augment_train)
    val_dataset = ObjDataset(val_root, augment=False)
    test_dataset = ObjDataset(test_root, augment=False)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


# NOTE: load_data_mia() is the ONLY public interface for data loading
# All other scripts must use the DataLoaders returned by load_data_mia()
