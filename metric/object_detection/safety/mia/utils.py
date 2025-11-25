"""Utility helpers for detection MIA pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AttackDataset(torch.utils.data.Dataset):
    """Lightweight dataset for CNN-based attack training.

    The dataset consumes pre-rendered canvases and binary membership labels.
    A dedicated class keeps the training loop code in :mod:`evaluate_mia`
    straightforward while preserving parity with the original standalone
    implementation.
    """

    def __init__(self, canvases: np.ndarray, labels: np.ndarray) -> None:
        self._canvases = canvases.astype(np.float32)
        self._labels = labels.astype(np.int64)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        canvas = self._canvases[idx]
        tensor = torch.from_numpy(np.repeat(canvas[None, ...], 3, axis=0))
        return tensor, torch.tensor(self._labels[idx])


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


def save_canvas_image(
    canvas: np.ndarray,
    save_path: Path,
    *,
    label: float | None = None,
    cmap: str = "hot",
    global_max: float | None = None,
) -> None:
    """Persist a canvas array for debugging or visualization.

    Args:
        canvas: Two-dimensional canvas array.
        save_path: Destination image path; parent directories are created.
        label: Optional membership label to include in the title.
        cmap: Matplotlib colormap name.
        global_max: Optional normalization constant to keep colors consistent
            across images.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    norm = Normalize(vmin=0, vmax=global_max if global_max is not None else max(np.max(canvas), 1.0))
    plt.imshow(canvas, cmap=cmap, norm=norm)
    plt.colorbar(label="Score Intensity")
    if label is not None:
        plt.title(f"Canvas (Label: {int(label)})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


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
    predictor: torch.nn.Module,
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
        predictor: Target or shadow model/estimator. Must expose ``predict``
            returning a list of detection dicts or be directly callable.
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
    if hasattr(predictor, "eval"):
        predictor.eval()
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

            if hasattr(predictor, "predict"):
                batch_tensor = torch.stack(images)
                outputs = predictor.predict(batch_tensor)
            else:
                outputs = predictor(images)
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
    save_samples: int = 0,
    save_dir: Path | None = None,
    global_normalize: bool = False,
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

    samples_list = list(samples)

    # Pre-pass for global normalization if requested
    global_max = None
    if global_normalize:
        maxima = []
        for sample in samples_list:
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
            maxima.append(np.max(canvas) if np.max(canvas) > 0 else 1.0)
        global_max = float(np.max(maxima)) if maxima else None

    # Main pass generating canvases
    for idx, sample in enumerate(samples_list):
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

        if save_samples > 0 and save_dir is not None and idx < save_samples:
            save_canvas_image(canvas, Path(save_dir) / f"canvas_{idx}.png", label=sample.label, global_max=global_max)

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
