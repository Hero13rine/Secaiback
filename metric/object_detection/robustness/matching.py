"""Shared matching helpers for robustness evaluators."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from metric.object_detection.basic.detection import DetectionSample

__all__ = [
    "compute_detection_errors",
    "greedy_iou_match",
    "pairwise_iou_axis_aligned",
]


def compute_detection_errors(
    predictions: Sequence[DetectionSample],
    ground_truths: Sequence[DetectionSample],
    iou_threshold: float,
) -> Tuple[int, int, int, int]:
    """Return miss/false-positive statistics for two aligned sample sequences."""

    misses = 0
    false_positives = 0
    total_gt = 0
    total_predictions = 0

    for pred, gt in zip(predictions, ground_truths):
        gt_boxes = gt.boxes
        pred_boxes = pred.boxes
        total_gt += gt_boxes.shape[0]
        total_predictions += pred_boxes.shape[0]
        matches = greedy_iou_match(pred_boxes, gt_boxes, iou_threshold)
        matches_found = len(matches)
        misses += gt_boxes.shape[0] - matches_found
        false_positives += pred_boxes.shape[0] - matches_found

    return misses, false_positives, total_gt, total_predictions


def greedy_iou_match(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float,
    pairwise_iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> List[Tuple[int, int]]:
    """Greedy IoU matching that favours highest IoU pairs first."""

    if pairwise_iou_fn is None:
        pairwise_iou_fn = pairwise_iou_axis_aligned

    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return []

    iou_matrix = pairwise_iou_fn(pred_boxes, gt_boxes)
    if iou_matrix.size == 0:
        return []

    matches: List[Tuple[int, int]] = []

    while True:
        flat_index = int(np.argmax(iou_matrix))
        best_iou = float(iou_matrix.flat[flat_index])
        if not np.isfinite(best_iou) or best_iou < threshold or best_iou <= 0.0:
            break
        pred_idx, gt_idx = divmod(flat_index, iou_matrix.shape[1])
        matches.append((pred_idx, gt_idx))
        iou_matrix[pred_idx, :] = -1.0
        iou_matrix[:, gt_idx] = -1.0

    return matches


def pairwise_iou_axis_aligned(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """Vectorised pairwise IoU for axis-aligned boxes."""

    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    pred = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    pred_exp = pred[:, None, :]
    gt_exp = gt[None, :, :]

    ix1 = np.maximum(pred_exp[..., 0], gt_exp[..., 0])
    iy1 = np.maximum(pred_exp[..., 1], gt_exp[..., 1])
    ix2 = np.minimum(pred_exp[..., 2], gt_exp[..., 2])
    iy2 = np.minimum(pred_exp[..., 3], gt_exp[..., 3])

    inter_w = np.maximum(ix2 - ix1, 0.0)
    inter_h = np.maximum(iy2 - iy1, 0.0)
    intersection = inter_w * inter_h

    pred_area = np.maximum(pred_exp[..., 2] - pred_exp[..., 0], 0.0) * np.maximum(
        pred_exp[..., 3] - pred_exp[..., 1], 0.0
    )
    gt_area = np.maximum(gt_exp[..., 2] - gt_exp[..., 0], 0.0) * np.maximum(
        gt_exp[..., 3] - gt_exp[..., 1], 0.0
    )
    union = pred_area + gt_area - intersection
    union = np.maximum(union, np.finfo(np.float32).eps)

    return (intersection / union).astype(np.float32)
