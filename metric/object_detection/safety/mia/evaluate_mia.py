"""Membership inference attack pipeline (adapted from :mod:`add.pipeline`).

This module reuses the streamlined Faster R-CNN MIA implementation under
``add/`` and adapts its parameters to the YAML configuration style used by the
entry script.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from add import load_dataset as mia_dataset
from add.load_dataset import load_data_mia
from add.pipeline import (
    PipelineConfig,
    step2_train_shadow,
    step3_train_attack,
    step4_evaluate,
)
from method import load_config


@dataclass
class EvaluationContext:
    """Aggregated configuration for running the MIA pipeline."""

    pipeline_config: PipelineConfig
    train_dir: Path
    val_dir: Path
    test_dir: Path


def _merge_attack_config(attack_config_path: str | None) -> Mapping[str, Any]:
    """Load optional attack parameters from YAML if the path is provided."""

    if not attack_config_path:
        return {}

    attack_cfg = load_config(attack_config_path)
    return attack_cfg.get("parameters", {}).get("optional", {})


def _build_pipeline_config(
    model_instantiation: Mapping[str, Any],
    attack_params: Mapping[str, Any],
) -> PipelineConfig:
    """Populate :class:`PipelineConfig` using model and attack settings."""

    config = PipelineConfig()

    # Model paths
    config.TARGET_MODEL_DIR = model_instantiation.get("weight_path", config.TARGET_MODEL_DIR)
    config.SHADOW_MODEL_DIR = attack_params.get("shadow_model", config.SHADOW_MODEL_DIR)
    config.ATTACK_MODEL_DIR = attack_params.get("attack_model", config.ATTACK_MODEL_DIR)

    # General
    config.num_classes = attack_params.get("num_classes", config.num_classes)
    config.img_size = attack_params.get("img_size", config.img_size)

    # Shadow training
    config.SHADOW_EPOCHS = attack_params.get("shadow_epochs", config.SHADOW_EPOCHS)
    config.SHADOW_BATCH_SIZE = attack_params.get("shadow_batch_size", config.SHADOW_BATCH_SIZE)
    config.SHADOW_LR = attack_params.get("shadow_lr", config.SHADOW_LR)
    config.SHADOW_WEIGHT_DECAY = attack_params.get("shadow_weight_decay", config.SHADOW_WEIGHT_DECAY)
    config.SHADOW_USE_PRETRAINED = attack_params.get("shadow_use_pretrained", config.SHADOW_USE_PRETRAINED)

    # Attack training
    config.ATTACK_EPOCHS = attack_params.get("attack_epochs", config.ATTACK_EPOCHS)
    config.ATTACK_BATCH_SIZE = attack_params.get("attack_batch_size", config.ATTACK_BATCH_SIZE)
    config.ATTACK_LR = attack_params.get("attack_lr", config.ATTACK_LR)
    config.ATTACK_WEIGHT_DECAY = attack_params.get("attack_weight_decay", config.ATTACK_WEIGHT_DECAY)
    config.ATTACK_MODEL_TYPE = attack_params.get("attack_model_type", config.ATTACK_MODEL_TYPE)

    # Feature/canvas
    config.CANVAS_SIZE = attack_params.get("canvas_size", config.CANVAS_SIZE)
    config.MAX_LEN = attack_params.get("max_len", config.MAX_LEN)
    config.LOG_SCORE = attack_params.get("log_score", config.LOG_SCORE)
    config.CANVAS_TYPE = attack_params.get("canvas_type", config.CANVAS_TYPE)
    config.NORMALIZE_CANVAS = attack_params.get("normalize_canvas", config.NORMALIZE_CANVAS)

    # Evaluation
    config.MIA_MEMBER_SAMPLES = attack_params.get("member_samples", config.MIA_MEMBER_SAMPLES)
    config.MIA_NONMEMBER_SAMPLES = attack_params.get("nonmember_samples", config.MIA_NONMEMBER_SAMPLES)

    return config


def _prepare_context(
    evaluation_config: Mapping[str, Any],
    model_instantiation: Mapping[str, Any],
) -> EvaluationContext:
    """Create the configuration context from YAML definitions."""

    attack_params = _merge_attack_config(evaluation_config.get("attack_config"))
    pipeline_config = _build_pipeline_config(model_instantiation, attack_params)

    dataset_cfg = evaluation_config.get("dataset", {})
    train_dir = Path(dataset_cfg.get("train_dir", model_instantiation.get("train_dir", "data/dataset/dior/train")))
    val_dir = Path(dataset_cfg.get("val_dir", model_instantiation.get("val_dir", "data/dataset/dior/val")))
    test_dir = Path(dataset_cfg.get("test_dir", model_instantiation.get("test_dir", "data/dataset/dior/test")))

    return EvaluationContext(pipeline_config=pipeline_config, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)


def evaluation_mia_detection(
    train_loader,
    val_loader,
    test_loader,
    evaluation_config: Mapping[str, Any],
    target_model=None,
    model_instantiation: Mapping[str, Any] | None = None,
):
    """Run the membership inference attack pipeline.

    The provided loaders are currently not reused because the pipeline relies on
    ``add.load_dataset`` utilities; they are accepted to preserve the original
    call signature.
    """

    model_instantiation = copy.deepcopy(model_instantiation or {})
    context = _prepare_context(evaluation_config, model_instantiation)
    cfg = context.pipeline_config

    # Override default dataset paths used by the add.* helpers
    mia_dataset.DEFAULT_TRAIN_DIR = str(context.train_dir)
    mia_dataset.DEFAULT_VAL_DIR = str(context.val_dir)
    mia_dataset.DEFAULT_TEST_DIR = str(context.test_dir)

    # Load datasets with configured paths
    train_loader, val_loader, test_loader = load_data_mia(
        train_root=context.train_dir,
        val_root=context.val_dir,
        test_root=context.test_dir,
        batch_size=cfg.SHADOW_BATCH_SIZE,
        num_workers=cfg.workers,
        augment_train=cfg.TRANSFORM,
    )

    # 1) train shadow model on TEST -> VAL
    step2_train_shadow(cfg)

    # 2) train attack model using shadow outputs
    step3_train_attack(cfg)

    # 3) evaluate attack against target model
    step4_evaluate(cfg)


__all__ = ["evaluation_mia_detection"]

