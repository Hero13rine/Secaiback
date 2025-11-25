"""Utility helpers for detection MIA pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class DetectionSample:
    """Container for detection features and membership label."""

    features: np.ndarray
    label: float


def logscore(values: np.ndarray, log_type: int = 2) -> np.ndarray:
    """Apply log scaling used by the original attack implementation.

    Args:
        values: Score array in ``[0, 1]``.
        log_type: 0 to keep raw values, 1 for natural log, 2 for log2.

    Returns:
        Scaled array with the same shape as ``values``.
    """
    clipped = np.clip(values, 0.0, 0.999999999)
    if log_type == 2:
        return -np.log2(1 - clipped + 1e-20)
    if log_type > 0:
        return -np.log(1 - clipped + 1e-20)
    return clipped


def _normalize_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    norm = boxes.copy().astype(np.float32)
    norm[:, 0] /= width
    norm[:, 1] /= height
    norm[:, 2] /= width
    norm[:, 3] /= height
    return norm


def generate_pointsets(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    img_size: int,
    device: torch.device,
    *,
    max_len: int = 50,
    log_score_type: int = 2,
    num_logit_feature: int = 1,
    regard_in_set: bool = True,
    max_samples: int | None = None,
    batch_size: int = 4,
) -> List[DetectionSample]:
    """Run detection model to build padded point sets for MIA training.

    Args:
        model: Target or shadow model.
        image_paths: Iterable of image files.
        img_size: Resize dimension before inference.
        device: Torch device for inference.
        max_len: Maximum number of detections per sample.
        log_score_type: Score scaling strategy.
        num_logit_feature: Additional per-box channels (label slot kept).
        regard_in_set: Whether these samples are members (1.0) or not (0.0).
        max_samples: Optional cap on processed samples.
        batch_size: Batch size for inference.

    Returns:
        List of :class:`DetectionSample` objects.
    """
    model.eval()
    dataset: List[DetectionSample] = []
    paths = list(image_paths)
    if max_samples is not None:
        paths = paths[:max_samples]

    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images: List[torch.Tensor] = []
            sizes: List[Tuple[int, int]] = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                sizes.append(img.size)
                resized = img.resize((img_size, img_size))
                tensor = TF.normalize(
                    TF.to_tensor(resized),
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
                images.append(tensor.to(device))

            outputs = model(images)
            for output, (orig_w, orig_h) in zip(outputs, sizes):
                boxes = output.get("boxes", torch.empty((0, 4))).cpu().numpy()
                scores = output.get("scores", torch.empty((0,))).cpu().numpy()
                labels = output.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()

                keep = scores > 0.05
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                if boxes.size > 0:
                    scores = logscore(scores, log_score_type)
                    norm_boxes = _normalize_boxes(boxes, orig_w, orig_h)
                    padded = np.zeros((max_len, 4 + num_logit_feature * 2), dtype=np.float32)
                    length = min(len(norm_boxes), max_len)
                    padded[:length, 0:4] = norm_boxes[:length]
                    padded[:length, 4] = scores[:length]
                    # background class is 0 in torchvision; subtract 1 to keep alignment
                    padded[:length, 4 + num_logit_feature] = np.maximum(0, labels[:length] - 1)
                else:
                    padded = np.zeros((max_len, 4 + num_logit_feature * 2), dtype=np.float32)

                dataset.append(DetectionSample(padded, 1.0 if regard_in_set else 0.0))
    return dataset


def make_canvas_data(
    samples: Iterable[DetectionSample],
    *,
    canvas_size: int = 300,
    canvas_type: str = "original",
    ball_size: int = 30,
    normalize: bool = True,
    log_score_type: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert padded point sets into canvas tensors for CNN attacks.

    Args:
        samples: Detection samples with padded point features.
        canvas_size: Output canvas spatial dimension.
        canvas_type: ``original`` for box fill, ``uniform`` for circle fill.
        ball_size: Circle diameter used when ``canvas_type`` is ``uniform``.
        normalize: Whether to normalize canvas heatmap by its mean.
        log_score_type: Score scaling used on raw scores.

    Returns:
        Tuple of ``(canvas_array, labels)`` where canvas_array has shape
        ``(N, canvas_size, canvas_size)``.
    """
    canvases: List[np.ndarray] = []
    labels: List[float] = []

    for sample in samples:
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        for feat in sample.features:
            if np.allclose(feat, 0.0):
                continue
            x0, y0, x1, y1, score, *_ = feat
            score = logscore(np.asarray([score]), log_score_type)[0]
            x0_px = int(max(0, min(canvas_size - 1, x0 * canvas_size)))
            y0_px = int(max(0, min(canvas_size - 1, y0 * canvas_size)))
            x1_px = int(max(0, min(canvas_size - 1, x1 * canvas_size)))
            y1_px = int(max(0, min(canvas_size - 1, y1 * canvas_size)))

            if canvas_type == "uniform":
                cx, cy = (x0_px + x1_px) // 2, (y0_px + y1_px) // 2
                radius = ball_size // 2
                y_range = np.arange(max(0, cy - radius), min(canvas_size, cy + radius + 1))
                x_range = np.arange(max(0, cx - radius), min(canvas_size, cx + radius + 1))
                for y in y_range:
                    for x in x_range:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                            canvas[int(y), int(x)] += score
            else:
                canvas[y0_px : y1_px + 1, x0_px : x1_px + 1] += score

        if normalize and np.sum(canvas) > 0:
            canvas = canvas / canvas.mean()

        canvases.append(canvas)
        labels.append(sample.label)

    return np.stack(canvases, axis=0), np.asarray(labels, dtype=np.float32)


def _find_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def load_dataset(root: Path) -> List[Path]:
    """Load image paths from YOLO-style folder.

    Args:
        root: Directory containing images.

    Returns:
        Sorted list of image paths.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root {root} not found")
    images = _find_images(root)
    if not images:
        raise RuntimeError(f"No images found under {root}")
    return images
