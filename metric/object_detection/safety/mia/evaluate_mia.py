"""Membership inference attack pipeline for object detection models."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.SecAISender import ResultSender

from .utils import DetectionSample, generate_pointsets, load_dataset, make_canvas_data

LOGGER = logging.getLogger(__name__)


@dataclass
class MIADetectionConfig:
    """Configuration for detection MIA pipeline."""

    target_model: Path
    shadow_model: Path
    attack_model_out: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
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
    member_samples: int = 3000
    nonmember_samples: int = 3000
    device: str = "auto"

    @classmethod
    def load(cls, attack_cfg: Path, user_cfg: Path) -> "MIADetectionConfig":
        """Load configuration from yaml definitions."""
        with open(attack_cfg, "r", encoding="utf-8") as handle:
            attack_data = yaml.safe_load(handle) or {}
        with open(user_cfg, "r", encoding="utf-8") as handle:
            user_data = yaml.safe_load(handle) or {}

        params: Dict[str, Any] = attack_data.get("parameters", {}).get("optional", {})
        model_section: Dict[str, Any] = user_data.get("model", {})
        eval_section: Dict[str, Any] = user_data.get("evaluation", {})
        dataset_section: Dict[str, Any] = model_section.get("dataset", {})

        target_path = Path(model_section.get("instantiation", {}).get("weight_path", "fasterrcnn_dior.pt"))
        shadow_path = Path(params.get("shadow_model", "runs/shadow_train/exp/best.pt"))
        attack_out = Path(params.get("attack_model", "runs/attacker_train/exp/best.pth"))

        return cls(
            target_model=target_path,
            shadow_model=shadow_path,
            attack_model_out=attack_out,
            train_dir=Path(dataset_section.get("train_dir", "data/dataset/dior/train")),
            val_dir=Path(dataset_section.get("val_dir", "data/dataset/dior/val")),
            test_dir=Path(dataset_section.get("test_dir", "data/dataset/dior/test")),
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
            member_samples=params.get("member_samples", 3000),
            nonmember_samples=params.get("nonmember_samples", 3000),
            device=params.get("device", "auto"),
        )

    @property
    def torch_device(self) -> torch.device:
        if self.device == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if self.device == "auto":
            return torch.device("cuda")
        return torch.device(self.device)


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


def _load_faster_rcnn(weight_path: Path, num_classes: int, device: torch.device) -> FasterRCNN:
    backbone = resnet_fpn_backbone("resnet50", weights=None)
    model = FasterRCNN(backbone, num_classes=num_classes + 1)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _train_attack_model(config: MIADetectionConfig, member: np.ndarray, non_member: np.ndarray) -> Tuple[AttackModel, Dict[str, float]]:
    device = config.torch_device
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(non_member))])
    data = np.concatenate([member, non_member], axis=0)

    attack = AttackModel(model_type=config.attack_model_type)
    attack.to(device)

    optimizer = optim.Adam(attack.parameters(), lr=config.attack_lr, weight_decay=config.attack_weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(config.attack_epochs):
        attack.train()
        perm = np.random.permutation(len(data))
        epoch_loss = 0.0
        for start in range(0, len(data), config.attack_batch):
            idx = perm[start : start + config.attack_batch]
            batch = data[idx]
            batch_labels = labels[idx]
            batch = np.repeat(batch[:, None, :, :], 3, axis=1)
            inputs = torch.from_numpy(batch).float().to(device)
            targets = torch.from_numpy(batch_labels).long().to(device)

            optimizer.zero_grad()
            outputs = attack(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(inputs)

        attack.eval()
        with torch.no_grad():
            canvas = np.repeat(data[:, None, :, :], 3, axis=1)
            preds = attack(torch.from_numpy(canvas).float().to(device))
            probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()
            auc = roc_auc_score(labels, probs)
            acc = accuracy_score(labels, (probs > 0.5).astype(int))
            precision, recall, f1, _ = precision_recall_fscore_support(labels, (probs > 0.5).astype(int), average="binary")
        if auc > best_auc:
            best_auc = auc
            best_metrics = {"auc": auc, "acc": acc, "precision": precision, "recall": recall, "f1": f1}
            os.makedirs(config.attack_model_out.parent, exist_ok=True)
            torch.save(attack.state_dict(), config.attack_model_out)
        LOGGER.debug("Epoch %s loss %.4f auc %.4f", epoch + 1, epoch_loss / len(data), auc)

    return attack, best_metrics


def evaluation_mia_detection(attack_config: str, user_config: str) -> Dict[str, Any]:
    """Unified entry point mirroring the original pipeline."""
    cfg = MIADetectionConfig.load(Path(attack_config), Path(user_config))
    device = cfg.torch_device

    ResultSender.send_log("进度", "开始加载数据与模型")

    train_images = load_dataset(cfg.train_dir)
    val_images = load_dataset(cfg.val_dir)
    test_images = load_dataset(cfg.test_dir)

    target_model = _load_faster_rcnn(cfg.target_model, cfg.num_classes, device)
    shadow_model = _load_faster_rcnn(cfg.shadow_model, cfg.num_classes, device)

    ResultSender.send_log("进度", "生成影子模型特征")
    member_samples = generate_pointsets(
        shadow_model,
        test_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=True,
        max_samples=cfg.member_samples,
    )
    non_member_samples = generate_pointsets(
        shadow_model,
        val_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=False,
        max_samples=cfg.nonmember_samples,
    )

    member_canvas, member_labels = make_canvas_data(
        member_samples,
        canvas_size=cfg.canvas_size,
        canvas_type=cfg.canvas_type,
        normalize=cfg.normalize_canvas,
        log_score_type=cfg.log_score,
    )
    non_member_canvas, non_member_labels = make_canvas_data(
        non_member_samples,
        canvas_size=cfg.canvas_size,
        canvas_type=cfg.canvas_type,
        normalize=cfg.normalize_canvas,
        log_score_type=cfg.log_score,
    )

    attack_model, metrics = _train_attack_model(cfg, member_canvas, non_member_canvas)

    ResultSender.send_log("进度", "在目标模型上提取评估样本")
    target_member = generate_pointsets(
        target_model,
        train_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=True,
        max_samples=cfg.member_samples,
    )
    target_non_member = generate_pointsets(
        target_model,
        test_images,
        cfg.img_size,
        device,
        max_len=cfg.max_len,
        log_score_type=cfg.log_score,
        regard_in_set=False,
        max_samples=cfg.nonmember_samples,
    )

    eval_canvas, eval_labels = make_canvas_data(
        [*target_member, *target_non_member],
        canvas_size=cfg.canvas_size,
        canvas_type=cfg.canvas_type,
        normalize=cfg.normalize_canvas,
        log_score_type=cfg.log_score,
    )

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


__all__ = ["evaluation_mia_detection", "MIADetectionConfig", "AttackModel"]
