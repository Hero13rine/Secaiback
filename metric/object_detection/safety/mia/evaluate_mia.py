"""Membership inference attack pipeline for object detection models."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from estimator import EstimatorFactory
from model import load_model
from method import load_config
from utils.SecAISender import ResultSender
from tqdm import tqdm

from .miaUtils import (
    AttackDataset,
    DetectionSample,
    generate_pointsets,
    load_dataset,
    make_canvas_data,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class MIADetectionConfig:
    """Configuration for detection MIA pipeline."""

    target_weight: Path
    shadow_weight: Path
    shadow_model_path: Path
    shadow_model_name: str
    shadow_model_parameters: Mapping[str, Any]
    attack_model_out: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    model_path: Path
    model_name: str
    model_parameters: Mapping[str, Any]
    estimator_config: Dict[str, Any]
    img_size: int = 640
    num_classes: int = 20
    canvas_size: int = 300
    max_len: int = 50
    log_score: int = 2
    canvas_type: str = "original"
    normalize_canvas: bool = True
    attack_epochs: int = 80
    attack_batch: int = 32
    attack_lr: float = 1e-5
    attack_weight_decay: float = 1e-3
    attack_model_type: str = "alex"
    attack_val_split: float = 0.2
    attack_patience: int = 5
    canvas_save_samples: int = 0
    canvas_save_dir: Path | None = None
    member_samples: int = 3000
    nonmember_samples: int = 3000
    device: str = "auto"
    shadow_epochs: int = 5
    shadow_batch_size: int = 4
    shadow_lr: float = 1e-3
    shadow_weight_decay: float = 1e-4

    @classmethod
    def load(cls, attack_cfg: Path, user_cfg: Path) -> "MIADetectionConfig":
        """Load configuration from yaml definitions."""
        attack_data = load_config(attack_cfg)
        user_data = load_config(user_cfg)

        params: Dict[str, Any] = attack_data.get("parameters", {}).get("optional", {})
        model_section: Dict[str, Any] = user_data.get("model", {})
        dataset_section: Dict[str, Any] = model_section.get("dataset", {})
        instantiation: Dict[str, Any] = model_section.get("instantiation", {})
        estimator_config: Dict[str, Any] = model_section.get("estimator", {})

        target_path = Path(instantiation.get("weight_path", "fasterrcnn_dior.pt"))
        shadow_path = Path(params.get("shadow_model", "runs/shadow_train/exp/best.pt"))
        attack_out = Path(params.get("attack_model", "runs/attacker_train/exp/best.pth"))

        return cls(
            target_weight=target_path,
            shadow_weight=shadow_path,
            attack_model_out=attack_out,
            train_dir=Path(dataset_section.get("train_dir", "data/dataset/dior/train")),
            val_dir=Path(dataset_section.get("val_dir", "data/dataset/dior/val")),
            test_dir=Path(dataset_section.get("test_dir", "data/dataset/dior/test")),
            model_path=Path(instantiation.get("model_path", "")),
            model_name=instantiation.get("model_name", ""),
            model_parameters=instantiation.get("parameters", {}),
            shadow_model_path=Path(params.get("shadow_model_path", instantiation.get("model_path", ""))),
            shadow_model_name=params.get("shadow_model_name", instantiation.get("model_name", "")),
            shadow_model_parameters=params.get("shadow_model_parameters", instantiation.get("parameters", {})),
            estimator_config=estimator_config,
            img_size=params.get("img_size", 640),
            num_classes=params.get("num_classes", 20),
            canvas_size=params.get("canvas_size", 300),
            max_len=params.get("max_len", 50),
            log_score=params.get("log_score", 2),
            canvas_type=params.get("canvas_type", "original"),
            normalize_canvas=params.get("normalize_canvas", True),
            attack_epochs=params.get("attack_epochs", 80),
            attack_batch=params.get("attack_batch_size", 32),
            attack_lr=params.get("attack_lr", 1e-5),
            attack_weight_decay=params.get("attack_weight_decay", 1e-3),
            attack_model_type=params.get("attack_model_type", "alex"),
            attack_val_split=params.get("attack_val_split", 0.2),
            attack_patience=params.get("attack_patience", 5),
            canvas_save_samples=params.get("canvas_save_samples", 0),
            canvas_save_dir=Path(params["canvas_save_dir"]) if params.get("canvas_save_dir") else None,
            member_samples=params.get("member_samples", 3000),
            nonmember_samples=params.get("nonmember_samples", 3000),
            device=params.get("device", "auto"),
            shadow_epochs=params.get("shadow_epochs", 5),
            shadow_batch_size=params.get("shadow_batch_size", 4),
            shadow_lr=params.get("shadow_lr", 1e-3),
            shadow_weight_decay=params.get("shadow_weight_decay", 1e-4),
        )

    @property
    def torch_device(self) -> torch.device:
        estimator_params = self.estimator_config.get("parameters", {}) if self.estimator_config else {}
        estimator_device = estimator_params.get("device_type") or estimator_params.get("device")
        requested = self.device if self.device != "auto" else estimator_device
        if requested == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if requested in {None, "auto", "gpu", "cuda"}:
            return torch.device("cuda")
        return torch.device(str(requested))


class AttackModel(nn.Module):
    """CNN attack model used by the original implementation."""

    def __init__(self, hidden_dim: int = 128, model_type: str = "shallow"):
        super().__init__()
        if model_type == "shallow":
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_estimator(model: torch.nn.Module, loss: Any, optimizer: Any, cfg: MIADetectionConfig) -> Any:
    """Wrap a user model with ART estimator using provided components."""

    return EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=cfg.estimator_config,
    )


def _load_estimator_from_weights(
    weight_path: Path,
    model_path: Path,
    model_name: str,
    model_parameters: Mapping[str, Any],
    cfg: MIADetectionConfig,
    fallback: Any,
) -> Any:
    """Load a dedicated estimator instance when weight file is available.

    If the configured weight file does not exist, the provided ``fallback``
    estimator is returned to keep backward compatibility with environments
    that only supply a single initialized model.
    """

    if not weight_path or not weight_path.exists():
        LOGGER.warning("Configured weight %s 不存在，回退到已有估计器", weight_path)
        return fallback

    model = load_model(model_path, model_name, str(weight_path), model_parameters)
    model.eval()
    return build_estimator(model, loss=None, optimizer=None, cfg=cfg)


def _train_shadow_model(
    cfg: MIADetectionConfig,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
) -> Path:
    """Train a shadow model when weights are not provided.

    The simplified flow uses ``test_loader`` data as shadow members and ``val_loader``
    data as shadow non-members.
    """
    LOGGER.info("未找到影子模型权重，开始训练影子模型：%s", cfg.shadow_weight)

    # train_loader is kept for signature clarity with the main pipeline
    # even though the simplified flow relies on test/val split for shadow data
    _ = train_loader

    class _ShadowCfg:
        use_external_loaders = True
        shadow_train_loader = test_loader
        shadow_val_loader = val_loader
        img_size = cfg.img_size
        SHADOW_BATCH_SIZE = cfg.shadow_batch_size
        SHADOW_EPOCHS = cfg.shadow_epochs
        SHADOW_LR = cfg.shadow_lr
        weight_decay = cfg.shadow_weight_decay
        SHADOW_MODEL_DIR = str(cfg.shadow_weight)
        num_classes = cfg.num_classes

    cfg.shadow_weight.parent.mkdir(parents=True, exist_ok=True)
    train_shadow_finetune(_ShadowCfg())
    return cfg.shadow_weight


def _extract_image_paths(loader: Any) -> Sequence[Path]:
    """Extract image paths from a dataloader with validation."""

    dataset = getattr(loader, "dataset", None)
    images = getattr(dataset, "images", None) if dataset is not None else None
    if images is None:
        raise ValueError("数据集缺少 images 属性，无法构建成员推理样本")
    return [Path(p) for p in images]


def _prepare_attack_dataset(
    samples: Sequence[DetectionSample],
    config: MIADetectionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert detection samples into canvases ready for model ingestion."""

    canvases, labels = make_canvas_data(
        samples,
        canvas_size=config.canvas_size,
        canvas_type=config.canvas_type,
        normalize=config.normalize_canvas,
        log_score_type=config.log_score,
        save_samples=config.canvas_save_samples,
        save_dir=config.canvas_save_dir,
        global_normalize=True,
    )
    return canvases, labels


# ---------------------------
# Evaluation helpers
# ---------------------------
def _match_detections(pred_boxes, gt_boxes, iou_thresh: float) -> Tuple[int, int, int]:
    """Match predicted boxes to ground truth boxes using IoU."""

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=ious.device)
    matched_pred = torch.zeros(len(pred_boxes), dtype=torch.bool, device=ious.device)

    for pred_idx in range(len(pred_boxes)):
        gt_idx = torch.argmax(ious[pred_idx])
        if ious[pred_idx, gt_idx] >= iou_thresh and not matched_gt[gt_idx]:
            matched_gt[gt_idx] = True
            matched_pred[pred_idx] = True

    tp = int(matched_pred.sum().item())
    fp = int((~matched_pred).sum().item())
    fn = int((~matched_gt).sum().item())
    return tp, fp, fn


def evaluate_shadow(
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
    num_classes: int = 20,
) -> Tuple[float, float, float, float]:
    """Evaluate a detection model on the provided dataloader.

    Computes precision, recall, F1, and a simplified mAP (per-class precision
    averaged at a single IoU threshold).
    """

    model.eval()
    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                pred_boxes = pred.get("boxes", torch.empty(0, 4, device=device)).to(device)
                pred_labels = pred.get("labels", torch.empty(0, dtype=torch.long, device=device)).to(device)
                scores = pred.get("scores", torch.empty(0, device=device)).to(device)

                gt_boxes = target.get("boxes", torch.empty(0, 4, device=device)).to(device)
                gt_labels = target.get("labels", torch.empty(0, dtype=torch.long, device=device)).to(device)

                for cls in range(1, num_classes + 1):
                    cls_pred_mask = (pred_labels == cls) & (scores >= score_thresh)
                    cls_gt_mask = gt_labels == cls
                    cls_pred_boxes = pred_boxes[cls_pred_mask]
                    cls_gt_boxes = gt_boxes[cls_gt_mask]

                    cls_tp, cls_fp, cls_fn = _match_detections(cls_pred_boxes, cls_gt_boxes, iou_thresh)
                    tp[cls - 1] += cls_tp
                    fp[cls - 1] += cls_fp
                    fn[cls - 1] += cls_fn

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()

    precision = (total_tp / (total_tp + total_fp + 1e-6)).item()
    recall = (total_tp / (total_tp + total_fn + 1e-6)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-6)) if (precision + recall) > 0 else 0.0

    per_class_precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    mAP = per_class_precision.mean().item()

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

    # Prefer explicitly provided shadow loaders, then generic external loaders
    shadow_train_loader = getattr(cfg, 'shadow_train_loader', None)
    shadow_val_loader = getattr(cfg, 'shadow_val_loader', None)
    use_external_loaders = getattr(cfg, 'use_external_loaders', False)

    if shadow_train_loader is not None or shadow_val_loader is not None:
        if shadow_train_loader is None or shadow_val_loader is None:
            raise ValueError("Shadow dataloaders must be provided together")
        print("Using shadow DataLoaders provided by caller")
        train_loader = shadow_train_loader
        val_loader = shadow_val_loader
    elif use_external_loaders:
        print("Using externally provided DataLoaders from pipeline")
        train_loader = getattr(cfg, 'train_loader', None)
        val_loader = getattr(cfg, 'val_loader', None)
        if train_loader is None or val_loader is None:
            raise ValueError("External dataloaders enabled but missing train_loader/val_loader")
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
    print(f"Train samples: {train_size}, Val samples: {val_size}")

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
        precision, recall, f1, mAP = evaluate_shadow(
            model,
            val_loader,
            device,
            iou_thresh=0.5,
            score_thresh=score_thresh,
            num_classes=num_classes,
        )
        if save_model and f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"🏆 New best model saved ({best_f1:.4f}, mAP={mAP:.4f})")

    if save_model:
        last_path = os.path.join(os.path.dirname(best_path), 'last.pt')
        torch.save(model.state_dict(), last_path)
        print(f"✅ Training finished. Last model saved to {last_path}")


def _train_attack_model(config: MIADetectionConfig, member: np.ndarray, non_member: np.ndarray) -> Tuple[AttackModel, Dict[str, float]]:
    """Train the CNN attacker using member/non-member canvases."""

    device = config.torch_device
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(non_member))])
    data = np.concatenate([member, non_member], axis=0)

    dataset = AttackDataset(data, labels)
    val_size = max(1, int(len(dataset) * config.attack_val_split))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.attack_batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.attack_batch, shuffle=False)

    attack = AttackModel(model_type=config.attack_model_type)
    attack.to(device)

    optimizer = optim.Adam(attack.parameters(), lr=config.attack_lr, weight_decay=config.attack_weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    best_metrics: Dict[str, float] = {}
    patience = config.attack_patience

    for epoch in range(config.attack_epochs):
        attack.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = attack(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(inputs)

        attack.eval()
        with torch.no_grad():
            all_probs: list[float] = []
            all_labels: list[int] = []
            for inputs, targets in val_loader:
                outputs = attack(inputs.to(device))
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(targets.numpy())
            if all_labels:
                auc = roc_auc_score(all_labels, all_probs)
                acc = accuracy_score(all_labels, (np.asarray(all_probs) > 0.5).astype(int))
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, (np.asarray(all_probs) > 0.5).astype(int), average="binary"
                )
            else:
                auc = acc = precision = recall = f1 = 0.0
        if auc >= best_auc:
            best_auc = auc
            patience = config.attack_patience
            best_metrics = {"auc": auc, "acc": acc, "precision": precision, "recall": recall, "f1": f1}
            os.makedirs(config.attack_model_out.parent, exist_ok=True)
            torch.save(attack.state_dict(), config.attack_model_out)
        else:
            patience -= 1
            if patience <= 0:
                LOGGER.info("Early stopping attack training after epoch %s", epoch + 1)
                break
        LOGGER.debug("Epoch %s loss %.4f val_auc %.4f", epoch + 1, epoch_loss / len(train_set), auc)

    return attack, best_metrics


def evaluation_mia_detection(
    estimator: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    safety_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """按照 ``eval_start`` 示例运行目标检测的成员推理攻击。"""

    attack_cfg_path = (
        safety_config.get("attack_config")
        or safety_config.get("mia_attack_config")
        or safety_config.get("mia_config")
        or "config/attack/miadet.yaml"
    )
    attack_cfg = load_config(attack_cfg_path) if attack_cfg_path else {}
    attack_params = attack_cfg.get("parameters", {}).get("optional", {}) if isinstance(attack_cfg, dict) else {}

    cfg = MIADetectionConfig(
        target_weight=Path(attack_params.get("target_weight", "target.pt")),
        shadow_weight=Path(attack_params.get("shadow_model", "runs/shadow_train/exp/best.pt")),
        attack_model_out=Path(attack_params.get("attack_model", "runs/attacker_train/exp/best.pth")),
        train_dir=Path(getattr(getattr(train_loader, "dataset", None), "root", "data/dataset/dior/train")),
        val_dir=Path(getattr(getattr(val_loader, "dataset", None), "root", "data/dataset/dior/val")),
        test_dir=Path(getattr(getattr(test_loader, "dataset", None), "root", "data/dataset/dior/test")),
        model_path=Path(attack_params.get("model_path", "")),
        model_name=str(attack_params.get("model_name", "")),
        model_parameters=attack_params.get("model_parameters", {}),
        shadow_model_path=Path(attack_params.get("shadow_model_path", attack_params.get("model_path", ""))),
        shadow_model_name=str(attack_params.get("shadow_model_name", attack_params.get("model_name", ""))),
        shadow_model_parameters=attack_params.get("shadow_model_parameters", attack_params.get("model_parameters", {})),
        estimator_config=getattr(estimator, "config", {}),
        img_size=attack_params.get("img_size", 640),
        num_classes=attack_params.get("num_classes", 20),
        canvas_size=attack_params.get("canvas_size", 300),
        max_len=attack_params.get("max_len", 50),
        log_score=attack_params.get("log_score", 2),
        canvas_type=attack_params.get("canvas_type", "original"),
        normalize_canvas=attack_params.get("normalize_canvas", True),
        attack_epochs=attack_params.get("attack_epochs", 80),
        attack_batch=attack_params.get("attack_batch_size", 32),
        attack_lr=attack_params.get("attack_lr", 1e-5),
        attack_weight_decay=attack_params.get("attack_weight_decay", 1e-3),
        attack_model_type=attack_params.get("attack_model_type", "alex"),
        attack_val_split=attack_params.get("attack_val_split", 0.2),
        attack_patience=attack_params.get("attack_patience", 5),
        canvas_save_samples=attack_params.get("canvas_save_samples", 0),
        canvas_save_dir=Path(attack_params["canvas_save_dir"]) if attack_params.get("canvas_save_dir") else None,
        member_samples=attack_params.get("member_samples", 3000),
        nonmember_samples=attack_params.get("nonmember_samples", 3000),
        device=attack_params.get("device", "auto"),
    )

    device = cfg.torch_device

    train_images = _extract_image_paths(train_loader)
    val_images = _extract_image_paths(val_loader)
    test_images = _extract_image_paths(test_loader)

    ResultSender.send_log("进度", "训练并使用影子模型生成特征")
    shadow_weight = cfg.shadow_weight
    if not shadow_weight.exists():
        _train_shadow_model(cfg, train_loader, val_loader, test_loader)

    shadow_model_path = cfg.shadow_model_path or cfg.model_path
    shadow_model_name = cfg.shadow_model_name or cfg.model_name
    shadow_model_parameters = cfg.shadow_model_parameters or cfg.model_parameters
    shadow_estimator = _load_estimator_from_weights(
        shadow_weight, shadow_model_path, shadow_model_name, shadow_model_parameters, cfg, estimator
    )
    member_samples = generate_pointsets(
        shadow_estimator,
        test_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=True,
        max_samples=cfg.member_samples,
    )
    non_member_samples = generate_pointsets(
        shadow_estimator,
        val_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=False,
        max_samples=cfg.nonmember_samples,
    )

    member_canvas, _ = _prepare_attack_dataset(member_samples, cfg)
    non_member_canvas, _ = _prepare_attack_dataset(non_member_samples, cfg)

    attack_model, metrics = _train_attack_model(cfg, member_canvas, non_member_canvas)

    ResultSender.send_log("进度", "在目标模型上提取评估样本")
    target_estimator = _load_estimator_from_weights(
        cfg.target_weight, cfg.model_path, cfg.model_name, cfg.model_parameters, cfg, estimator
    )
    target_member = generate_pointsets(
        target_estimator,
        train_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=True,
        max_samples=cfg.member_samples,
    )
    target_non_member = generate_pointsets(
        target_estimator,
        test_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=False,
        max_samples=cfg.nonmember_samples,
    )

    eval_canvas, eval_labels = _prepare_attack_dataset([*target_member, *target_non_member], cfg)

    attack_model.eval()
    with torch.no_grad():
        eval_input = np.repeat(eval_canvas[:, None, :, :], 3, axis=1)
        preds = attack_model(torch.from_numpy(eval_input).float().to(device))
        probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()
    auc = roc_auc_score(eval_labels, probs)
    acc = accuracy_score(eval_labels, (probs > 0.5).astype(int))
    precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, (probs > 0.5).astype(int), average="binary")

    result = {
        "attack_training": metrics,
        "evaluation": {
            "auc": float(auc),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
    }

    ResultSender.send_result("mia_detection", result)
    ResultSender.send_status("成功")
    return result


__all__ = [
    "evaluation_mia_detection",
    "MIADetectionConfig",
    "AttackModel",
    "build_estimator",
    "load_dataset",
]