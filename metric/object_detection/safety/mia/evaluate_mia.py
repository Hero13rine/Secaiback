"""Membership inference attack entrypoints."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .pipeline import (
    load_pipeline_config,
    step2_train_shadow,
    step3_train_attack,
    step4_evaluate,
)


def _prepare_pipeline_config(
    evaluation_config: Mapping[str, Any],
    model_instantiation: Mapping[str, Any],
):
    """Create the pipeline configuration from YAML plus overrides."""

    attack_config_path = evaluation_config.get("attack_config") or None
    attack_overrides = evaluation_config.get("attack_override", {})

    target_model_dir = model_instantiation.get("weight_path")
    if not target_model_dir:
        raise ValueError("model.instantiation.weight_path 必须提供，用于设置 TARGET_MODEL_DIR")

    overrides = {
        key: value
        for key, value in {
            "TARGET_MODEL_DIR": target_model_dir,
            **attack_overrides,
        }.items()
        if value is not None
    }

    return load_pipeline_config(attack_config_path, overrides)


def evaluation_mia_detection(
    train_loader,
    val_loader,
    test_loader,
    evaluation_config: Mapping[str, Any],
    target_model=None,
    model_instantiation: Mapping[str, Any] | None = None,
):
    """Run the membership inference attack pipeline.

    DataLoaders are prepared by :mod:`eva_start_mia` and passed through to the
    refactored pipeline so datasets are instantiated only once. The target
    model instance should also be supplied by the caller so it can be evaluated
    directly without reloading from disk.
    """

    model_instantiation = copy.deepcopy(model_instantiation or {})
    cfg = _prepare_pipeline_config(evaluation_config, model_instantiation)

    step2_train_shadow(cfg, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    step3_train_attack(cfg, train_loader=train_loader, test_loader=test_loader)
    step4_evaluate(
        cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        target_model=target_model,
    )


__all__ = ["evaluation_mia_detection"]