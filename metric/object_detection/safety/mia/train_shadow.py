#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shadow Model Training for MIA (Membership Inference Attack)

In the simplified MIA flow:
- Shadow model is trained on TEST set (as shadow's member samples)
- Validation set is used as shadow's non-member samples for attack model training
- The pre-trained target model already exists and is trained on TRAIN set
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


# Config will be passed as parameter when called from pipeline
# For standalone execution, config is imported in __main__
Config = None

# Collate function for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------
# Evaluate: add mAP computation
# ---------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, iou_thresh=0.5, score_thresh=0.5, num_classes=20):
    model.eval()
    TP, FP, FN = 0, 0, 0
    all_detections, all_annotations = [], []

    for images, targets in tqdm(data_loader, desc="🔍 validate", ncols=120):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            pred_boxes = out['boxes'].cpu()
            pred_scores = out['scores'].cpu()
            pred_labels = out['labels'].cpu()

            keep = pred_scores >= score_thresh
            pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

            gt_boxes = tgt['boxes'].cpu()
            gt_labels = tgt['labels'].cpu()

            all_detections.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            })
            all_annotations.append({
                "boxes": gt_boxes,
                "labels": gt_labels
            })

            # F1 calc
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            if len(pred_boxes) == 0:
                FN += len(gt_boxes)
                continue
            if len(gt_boxes) == 0:
                FP += len(pred_boxes)
                continue

            all_classes = torch.unique(torch.cat([pred_labels, gt_labels]))
            for cls in all_classes:
                p_idx = (pred_labels == cls).nonzero(as_tuple=False).squeeze(1)
                g_idx = (gt_labels == cls).nonzero(as_tuple=False).squeeze(1)
                if p_idx.numel() == 0 and g_idx.numel() == 0:
                    continue
                if p_idx.numel() == 0:
                    FN += int(g_idx.numel())
                    continue
                if g_idx.numel() == 0:
                    FP += int(p_idx.numel())
                    continue
                p_boxes, g_boxes = pred_boxes[p_idx], gt_boxes[g_idx]
                ious = box_iou(p_boxes, g_boxes)
                matched_g = set()
                for i in range(ious.shape[0]):
                    j = int(torch.argmax(ious[i]).item())
                    if ious[i, j] >= iou_thresh and j not in matched_g:
                        TP += 1
                        matched_g.add(j)
                    else:
                        FP += 1
                FN += int(g_boxes.shape[0] - len(matched_g))

    # compute F1
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # compute mAP@0.5
    aps = []
    for cls_id in range(1, num_classes + 1):
        cls_dets, cls_gts = [], []
        for det, gt in zip(all_detections, all_annotations):
            det_mask = det["labels"] == cls_id
            gt_mask = gt["labels"] == cls_id
            if det_mask.any():
                cls_dets.append({
                    "boxes": det["boxes"][det_mask],
                    "scores": det["scores"][det_mask]
                })
            else:
                cls_dets.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,))})
            cls_gts.append(gt["boxes"][gt_mask])

        npos = sum(len(g) for g in cls_gts)
        if npos == 0:
            continue

        scores, matches = [], []
        for det_boxes, det_scores, gt_boxes in zip(
            [d["boxes"] for d in cls_dets], [d["scores"] for d in cls_dets], cls_gts):
            if len(det_boxes) == 0:
                continue
            ious = box_iou(det_boxes, gt_boxes) if len(gt_boxes) > 0 else torch.zeros((len(det_boxes), 0))
            assigned = np.zeros(len(gt_boxes))
            for i in range(len(det_boxes)):
                scores.append(det_scores[i].item())
                if len(gt_boxes) == 0:
                    matches.append(0)
                    continue
                j = torch.argmax(ious[i]).item()
                if ious[i, j] >= iou_thresh and assigned[j] == 0:
                    matches.append(1)
                    assigned[j] = 1
                else:
                    matches.append(0)

        if len(scores) == 0:
            continue

        scores, matches = np.array(scores), np.array(matches)
        order = np.argsort(-scores)
        matches = matches[order]
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        recall_curve = tp / npos
        precision_curve = tp / (tp + fp + 1e-12)
        ap = np.trapz(precision_curve, recall_curve)
        aps.append(ap)

    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    print(f"\n📈 Validate -> P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, mAP@0.5: {mAP:.4f}")
    return precision, recall, f1, mAP


# ---------------------------
# Train Shadow Model
# ---------------------------
def train_shadow_finetune(cfg):
    """
    Train shadow model using the new simplified MIA flow:
    - Training data: TEST set (becomes shadow model's member samples)
    - Validation data: VAL set (for monitoring training progress)

    The shadow model is trained on different data than the target model,
    which allows it to learn similar patterns for membership inference.
    """
    device = torch.device(f"cuda:{getattr(cfg, 'gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = getattr(cfg, 'img_size', 640)
    batch_size = getattr(cfg, 'SHADOW_BATCH_SIZE', getattr(cfg, 'batch_size', 4))
    workers = getattr(cfg, 'workers', 0)
    epochs = getattr(cfg, 'SHADOW_EPOCHS', getattr(cfg, 'EPOCHS', 50))
    lr = getattr(cfg, 'SHADOW_LR', 0.001)
    weight_decay = getattr(cfg, 'weight_decay', 1e-4)
    save_model = getattr(cfg, 'SAVE_MODEL', True)
    score_thresh = getattr(cfg, 'VAL_SCORE_THRESH', 0.5)

    print(f"\n{'='*60}")
    print("Shadow Model Training Configuration (Simplified MIA Flow)")
    print(f"{'='*60}")

    # Check if external dataloaders are provided
    use_external_loaders = getattr(cfg, 'use_external_loaders', False)

    if use_external_loaders:
        print("Using externally provided DataLoaders from pipeline")
        train_loader = cfg.train_loader
        val_loader = cfg.val_loader
        # Estimate dataset sizes from dataloaders
        train_size = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'unknown'
        val_size = len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else 'unknown'
        print(f"Train samples: {train_size}, Val samples: {val_size}")
    else:
        print("Loading datasets using load_dataset module")
        print(f"{'='*60}\n")

        # Load data using load_dataset module (the ONLY data source)
        from load_dataset import load_data_mia

        # For shadow training: we need TEST as training, VAL as validation
        _, val_loader, test_loader = load_data_mia(
            batch_size=batch_size,
            num_workers=workers,
            augment_train=True
        )

        train_loader = test_loader  # Shadow trains on TEST set
        # val_loader already correct

        train_size = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'unknown'
        val_size = len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else 'unknown'
        print(f"Train samples (TEST set): {train_size}, Val samples (VAL set): {val_size}")

    print(f"{'='*60}\n")

    # Always use pretrained weights for shadow model (official FasterRCNN pretrained on COCO)
    use_pretrained = getattr(cfg, 'SHADOW_USE_PRETRAINED', True)
    print(f"Loading Faster R-CNN with official pretrained weights (pretrained={use_pretrained})...")
    if use_pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, getattr(cfg, 'num_classes', 20) + 1)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=getattr(cfg, 'num_classes', 20) + 1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_path = getattr(cfg, 'SHADOW_MODEL_DIR', 'runs/shadow_train/exp/best.pt')
    os.makedirs(os.path.dirname(best_path) or '.', exist_ok=True)
    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=120):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += loss_value
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        num_classes = getattr(cfg, 'num_classes', 20)
        precision, recall, f1, mAP = evaluate(model, val_loader, device, iou_thresh=0.5, score_thresh=score_thresh, num_classes=num_classes)
        if save_model and f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"🏆 New best model saved ({best_f1:.4f}, mAP={mAP:.4f})")

    if save_model:
        last_path = os.path.join(os.path.dirname(best_path), 'last.pt')
        torch.save(model.state_dict(), last_path)
        print(f"✅ Training finished. Last model saved to {last_path}")


def train_shadow_with_config(pipeline_config, train_loader=None, val_loader=None):
    """
    Train shadow model with configuration from pipeline.

    Args:
        pipeline_config: PipelineConfig object from pipeline.py
        train_loader: Optional pre-loaded training DataLoader
        val_loader: Optional pre-loaded validation DataLoader
    """
    # Create a config-like object that train_shadow_finetune expects
    class ConfigAdapter:
        pass

    cfg = ConfigAdapter()

    # Map pipeline config to expected attributes
    cfg.gpu_id = pipeline_config.gpu_id
    cfg.img_size = pipeline_config.img_size
    cfg.num_classes = pipeline_config.num_classes
    cfg.workers = pipeline_config.workers
    cfg.SAVE_MODEL = pipeline_config.SAVE_MODEL
    cfg.TRANSFORM = pipeline_config.TRANSFORM

    # Shadow model specific
    cfg.SHADOW_EPOCHS = pipeline_config.SHADOW_EPOCHS
    cfg.SHADOW_BATCH_SIZE = pipeline_config.SHADOW_BATCH_SIZE
    cfg.SHADOW_LR = pipeline_config.SHADOW_LR
    cfg.SHADOW_USE_PRETRAINED = pipeline_config.SHADOW_USE_PRETRAINED
    cfg.SHADOW_MODEL_DIR = pipeline_config.SHADOW_MODEL_DIR
    cfg.weight_decay = pipeline_config.SHADOW_WEIGHT_DECAY

    # Note: Data directories are managed by load_dataset module

    # If dataloaders provided, pass them to training function
    if train_loader is not None and val_loader is not None:
        cfg.use_external_loaders = True
        cfg.train_loader = train_loader
        cfg.val_loader = val_loader
    else:
        cfg.use_external_loaders = False

    train_shadow_finetune(cfg)


if __name__ == "__main__":
    print("=" * 70)
    print("ERROR: This script cannot be run directly!")
    print("=" * 70)
    print("\nThis script must be called from pipeline.py")
    print("\nUsage:")
    print("  python pipeline.py --steps 2")
    print("  python pipeline.py  # Run complete pipeline")
    print("\n" + "=" * 70)
    import sys
    sys.exit(1)