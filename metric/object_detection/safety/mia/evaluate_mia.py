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

from estimator import EstimatorFactory
from model import load_model
from method import load_config
from utils.SecAISender import ResultSender

from .utils import (
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
    shadow_model_path: Path | None = None
    shadow_model_name: str | None = None
    shadow_model_parameters: Mapping[str, Any] | None = None
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

    The training follows the simplified flow from ``add/train_shadow.py``:
    - shadow members come from the provided ``test_loader``
    - shadow non-members come from ``val_loader``
    """

    from add.train_shadow import train_shadow_finetune

    LOGGER.info("未找到影子模型权重，开始训练影子模型：%s", cfg.shadow_weight)

    # train_loader is kept for signature clarity with the main pipeline
    # even though the simplified flow relies on test/val split for shadow data
    _ = train_loader

    class _ShadowCfg:
        use_external_loaders = True
        train_loader = test_loader
        val_loader = val_loader
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
