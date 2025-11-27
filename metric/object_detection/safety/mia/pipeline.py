"""MIA pipeline helpers.

This module orchestrates the shadow/attack/evaluation steps using a configuration
loaded from ``config/attack/miadet.yaml``. All DataLoaders **must** be provided
by callers (for example from :mod:`eva_start_mia`); the pipeline will only raise
errors if they are missing instead of attempting to load datasets internally.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Mapping, List
from pathlib import Path
from method import load_config

from .atk import train_attack_with_config
from .mia import evaluate_attack_with_config
from .train_shadow import train_shadow_with_config


DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[4] / "config/attack/miadet.yaml"
)

def _load_base_config(path: str | None) -> Mapping[str, Any]:
    """Load base parameters from the miadet YAML file."""

    config_path = path or DEFAULT_CONFIG_PATH
    attack_cfg = load_config(config_path)
    params = attack_cfg.get("parameters", {})
    base = {}
    base.update(params.get("required", {}))
    base.update(params.get("optional", {}))
    return base


def _normalize_keys(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    """Map legacy lowercase keys into the uppercase names used in code."""

    aliases = {
        "shadow_model": "SHADOW_MODEL_DIR",
        "attack_model": "ATTACK_MODEL_DIR",
        "attack_epochs": "ATTACK_EPOCHS",
        "attack_batch_size": "ATTACK_BATCH_SIZE",
        "attack_lr": "ATTACK_LR",
        "attack_weight_decay": "ATTACK_WEIGHT_DECAY",
        "attack_model_type": "ATTACK_MODEL_TYPE",
        "member_samples": "MIA_MEMBER_SAMPLES",
        "nonmember_samples": "MIA_NONMEMBER_SAMPLES",
        "canvas_size": "CANVAS_SIZE",
        "max_len": "MAX_LEN",
        "log_score": "LOG_SCORE",
        "canvas_type": "CANVAS_TYPE",
        "normalize_canvas": "NORMALIZE_CANVAS",
        "shadow_epochs": "SHADOW_EPOCHS",
        "shadow_batch_size": "SHADOW_BATCH_SIZE",
        "shadow_lr": "SHADOW_LR",
        "shadow_weight_decay": "SHADOW_WEIGHT_DECAY",
        "shadow_use_pretrained": "SHADOW_USE_PRETRAINED",
        "pretrained_model": "PRETRAINED_MODEL",
        "results_dir": "RESULTS_DIR",
        "results_file": "RESULTS_FILE",
        "save_model": "SAVE_MODEL",
        "transform": "TRANSFORM",
    }

    normalized = {}
    for key, value in settings.items():
        normalized[aliases.get(key, key)] = value
    return normalized


def load_pipeline_config(
    config_path: str | None = None, overrides: Mapping[str, Any] | None = None
) -> SimpleNamespace:
    """Build a configuration namespace from YAML plus overrides."""

    base = _normalize_keys(_load_base_config(config_path))
    final = {**base, **_normalize_keys(overrides or {})}
    return SimpleNamespace(**final)


def _require_loader(name: str, loader):
    if loader is None:
        raise ValueError(f"{name} DataLoader must be provided")


def step1_configure(config: SimpleNamespace):
    """Display configuration summary."""

    print("\n" + "=" * 70)
    print(" Configuration")
    print("=" * 70 + "\n")

    print("Model Paths:")
    print(f"  - Target model: {config.TARGET_MODEL_DIR}")
    print(f"  - Shadow model: {config.SHADOW_MODEL_DIR}")
    print(f"  - Attack model: {config.ATTACK_MODEL_DIR}")
    if getattr(config, "PRETRAINED_MODEL", None):
        print(f"  - Pretrained weights (local): {config.PRETRAINED_MODEL}")

    print("\nShadow Model Training:")
    print(f"  - Epochs: {config.SHADOW_EPOCHS}")
    print(f"  - Batch size: {config.SHADOW_BATCH_SIZE}")
    print(f"  - Learning rate: {config.SHADOW_LR}")
    print(f"  - Use pretrained: {config.SHADOW_USE_PRETRAINED}")

    print("\nAttack Model Training:")
    print(f"  - Model type: {config.ATTACK_MODEL_TYPE}")
    print(f"  - Epochs: {config.ATTACK_EPOCHS}")
    print(f"  - Batch size: {config.ATTACK_BATCH_SIZE}")
    print(f"  - Learning rate: {config.ATTACK_LR}")
    print(f"  - Canvas size: {config.CANVAS_SIZE}x{config.CANVAS_SIZE}")

    print("\nMIA Evaluation:")
    print(f"  - Member samples: {config.MIA_MEMBER_SAMPLES}")
    print(f"  - Non-member samples: {config.MIA_NONMEMBER_SAMPLES}")


def step2_train_shadow(
    config: SimpleNamespace,
    *,
    train_loader,
    val_loader,
    test_loader,
):
    """Train shadow model using provided loaders."""

    _require_loader("train_loader", train_loader)
    _require_loader("val_loader", val_loader)
    _require_loader("test_loader", test_loader)

    shadow_train_loader = test_loader
    shadow_val_loader = val_loader

    print("\n" + "-" * 70)
    print(" Step 2/4: Shadow Model Training")
    print("-" * 70 + "\n")
    print(f"Training data: TEST set ({len(shadow_train_loader.dataset)} images)")
    print(f"Validation data: VAL set ({len(shadow_val_loader.dataset)} images)")
    print(f"Using pretrained: {config.SHADOW_USE_PRETRAINED}")
    print(f"Epochs: {config.SHADOW_EPOCHS}")
    print(f"Batch size: {config.SHADOW_BATCH_SIZE}")
    print(f"Learning rate: {config.SHADOW_LR}")
    print(f"Output: {config.SHADOW_MODEL_DIR}")

    start_time = time.time()
    train_shadow_with_config(
        config, train_loader=shadow_train_loader, val_loader=shadow_val_loader
    )
    elapsed = time.time() - start_time

    print(f"\n✅ Shadow model training completed in {elapsed/60:.2f} minutes")
    print(f"   Model saved to: {config.SHADOW_MODEL_DIR}")


def step3_train_attack(
    config: SimpleNamespace,
    *,
    train_loader,
    test_loader,
):
    """Train attack model using shadow outputs."""

    _require_loader("train_loader", train_loader)
    _require_loader("test_loader", test_loader)

    if not os.path.exists(config.SHADOW_MODEL_DIR):
        raise FileNotFoundError(f"Shadow model not found: {config.SHADOW_MODEL_DIR}")

    import numpy as np

    member_img_paths = [str(test_loader.dataset.images[i]) for i in range(len(test_loader.dataset))]
    all_train_img_paths = [str(train_loader.dataset.images[i]) for i in range(len(train_loader.dataset))]

    np.random.seed(42)
    num_members = len(member_img_paths)
    if len(all_train_img_paths) > num_members:
        sampled_indices = np.random.choice(len(all_train_img_paths), num_members, replace=False)
        nonmember_img_paths = [all_train_img_paths[i] for i in sampled_indices]
    else:
        nonmember_img_paths = all_train_img_paths

    print("\n" + "-" * 70)
    print(" Step 3/4: Attack Model Training")
    print("-" * 70 + "\n")
    print(f"Member samples (TEST set): {len(member_img_paths)}")
    print(f"Non-member samples (TRAIN set downsampled): {len(nonmember_img_paths)}")
    print(f"Shadow model: {config.SHADOW_MODEL_DIR}")
    print(f"Attack model type: {config.ATTACK_MODEL_TYPE}")
    print(f"Epochs: {config.ATTACK_EPOCHS}")
    print(f"Canvas size: {config.CANVAS_SIZE}x{config.CANVAS_SIZE}")
    print(f"Output: {config.ATTACK_MODEL_DIR}")

    start_time = time.time()
    train_attack_with_config(config, member_img_paths, nonmember_img_paths)
    elapsed = time.time() - start_time

    print(f"\n✅ Attack model training completed in {elapsed/60:.2f} minutes")


def step4_evaluate(
    config: SimpleNamespace,
    *,
    train_loader,
    test_loader,
    target_model=None,
):
    """Evaluate attack against the target model."""

    _require_loader("train_loader", train_loader)
    _require_loader("test_loader", test_loader)

    if target_model is None and not os.path.exists(config.TARGET_MODEL_DIR):
        raise FileNotFoundError(f"Target model not found: {config.TARGET_MODEL_DIR}")

    attack_model_path = config.ATTACK_MODEL_DIR
    if not os.path.exists(attack_model_path):
        attack_dir = os.path.dirname(attack_model_path)
        for candidate in ["best.pth", "last.pth"]:
            alt_path = os.path.join(attack_dir, candidate)
            if os.path.exists(alt_path):
                attack_model_path = alt_path
                print(f"Using attack model: {alt_path}")
                break
        else:
            raise FileNotFoundError(f"Attack model not found in: {attack_dir}")

    target_member_imgs = [str(train_loader.dataset.images[i]) for i in range(len(train_loader.dataset))]
    target_nonmember_imgs = [str(test_loader.dataset.images[i]) for i in range(len(test_loader.dataset))]

    target_member_imgs = target_member_imgs[: config.MIA_MEMBER_SAMPLES]
    target_nonmember_imgs = target_nonmember_imgs[: config.MIA_NONMEMBER_SAMPLES]

    print("\n" + "-" * 70)
    print(" Step 4/4: Attack Evaluation")
    print("-" * 70 + "\n")
    print(f"Member samples (TRAIN set): {len(target_member_imgs)}")
    print(f"Non-member samples (TEST set): {len(target_nonmember_imgs)}")
    print(f"Target model: {config.TARGET_MODEL_DIR}")
    print(f"Attack model: {attack_model_path}")
    print(f"Results: {config.RESULTS_FILE}")

    start_time = time.time()
    evaluate_attack_with_config(
        config,
        target_member_imgs,
        target_nonmember_imgs,
        target_model=target_model,
    )
    elapsed = time.time() - start_time

    print(f"\n✅ Attack evaluation completed in {elapsed/60:.2f} minutes")
    print(f"   Results saved to: {config.RESULTS_FILE}")


def run_pipeline(args, config: SimpleNamespace, *, train_loader, val_loader, test_loader, target_model=None):
    """Run the complete MIA pipeline using provided loaders."""

    print("\n" + "=" * 70)
    print(" MIA Pipeline for Faster R-CNN Object Detection")
    print("=" * 70 + "\n")

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Selected steps: {args.steps if args.steps else 'all'}")

    steps = [int(s) for s in args.steps.split(',')] if args.steps else [1, 2, 3, 4]
    results = {}

    if 1 in steps:
        step1_configure(config)
        results[1] = True

    if 2 in steps:
        step2_train_shadow(config, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        results[2] = True

    if 3 in steps:
        step3_train_attack(config, train_loader=train_loader, test_loader=test_loader)
        results[3] = True

    if 4 in steps:
        step4_evaluate(config, train_loader=train_loader, test_loader=test_loader, target_model=target_model)
        results[4] = True

    all_success = all(results.get(s, False) for s in steps)

    total_elapsed = sum(
        0 for _ in []  # placeholder to keep structure consistent; elapsed logged per step
    )

    print("\n" + "=" * 70)
    print(" Pipeline Summary")
    print("=" * 70 + "\n")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {total_elapsed/60:.2f} minutes (per-step timings logged above)")

    step_names = {1: "Configuration", 2: "Shadow Model Training", 3: "Attack Model Training", 4: "Attack Evaluation"}
    for step in steps:
        status = "✅ Success" if results.get(step, False) else "❌ Failed"
        print(f"  Step {step} ({step_names[step]}): {status}")

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="MIA Pipeline for Faster R-CNN Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--steps', type=str, default=None, help='Comma-separated list of steps to run (e.g., "1,2,3,4" or "2,3")')
    parser.add_argument('--config', type=str, default=None, help='Path to miadet config YAML file')

    args = parser.parse_args()

    raise RuntimeError(
        "This pipeline requires external DataLoaders and target model instances. "
        "Use eva_start_mia or integrate run_pipeline() with prepared loaders instead."
    )


if __name__ == '__main__':
    main()