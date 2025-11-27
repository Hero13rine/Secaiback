

import os
import argparse
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# Dataset loading lives in the legacy ``add`` module to keep a single
# source of truth for file-system assumptions.
from add.load_dataset import load_data_mia

from .atk import train_attack_with_config
from .mia import evaluate_attack_with_config
from .train_shadow import train_shadow_with_config


# ============================================================================
# CONFIGURATION - All parameters are defined here
# ============================================================================
@dataclass
class PipelineConfig:
    """
    Unified configuration for MIA pipeline.
    Modify parameters here to control the entire pipeline.
    """

    # NOTE: Dataset paths are now managed by load_dataset module
    # Use load_dataset.get_default_data_dirs() to get default paths
    # Or pass custom paths to load_dataset functions

    # ========== Model Paths ==========
    TARGET_MODEL_DIR: str = 'fasterrcnn_dior.pt'    # Pre-trained target model
    SHADOW_MODEL_DIR: str = 'runs/shadow_train/exp/best.pt'    # Shadow model output
    ATTACK_MODEL_DIR: str = 'runs/attacker_train/exp/best.pth' # Attack model output

    # ========== General Settings ==========
    gpu_id: int = 0
    num_classes: int = 20  # DIOR dataset has 20 classes
    img_size: int = 640    # Input image size for Faster R-CNN

    # ========== Shadow Model Training ==========
    SHADOW_EPOCHS: int = 30
    SHADOW_BATCH_SIZE: int = 16
    SHADOW_LR: float = 0.001
    SHADOW_USE_PRETRAINED: bool = True  # Use official pretrained weights
    SHADOW_WEIGHT_DECAY: float = 1e-4

    # ========== Attack Model Training ==========
    ATTACK_EPOCHS: int = 80
    ATTACK_BATCH_SIZE: int = 32
    ATTACK_LR: float = 1e-5
    ATTACK_WEIGHT_DECAY: float = 1e-3
    ATTACK_MODEL_TYPE: str = 'alex'  # 'alex' or 'shallow'

    # ========== Canvas/Feature Settings ==========
    CANVAS_SIZE: int = 300      # Canvas size for attack model input
    MAX_LEN: int = 50           # Max detected boxes per image
    LOG_SCORE: int = 2          # 0: raw, 1: ln, 2: log2
    CANVAS_TYPE: str = 'original'  # 'original' or 'uniform'
    NORMALIZE_CANVAS: bool = True

    # ========== MIA Evaluation ==========
    MIA_MEMBER_SAMPLES: int = 3000     # Member samples for evaluation
    MIA_NONMEMBER_SAMPLES: int = 3000  # Non-member samples for evaluation
    TRAIN_SPLIT_RATIO: float = 0.8     # Train/val split for attack model

    # ========== Output Settings ==========
    SAVE_MODEL: bool = True
    RESULTS_DIR: str = 'result'
    RESULTS_FILE: str = 'result/attack_results.csv'

    # ========== Other ==========
    workers: int = 0
    TRANSFORM: bool = True  # Enable augmentation


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_step(step_num, total, description):
    """Print step information"""
    print("\n" + "-" * 70)
    print(f" Step {step_num}/{total}: {description}")
    print("-" * 70 + "\n")


def step1_configure(config: PipelineConfig):
    """
    Step 1: Configuration
    Display and verify configuration settings
    """
    print_step(1, 4, "Configuration")

    print("Dataset Paths:")
    print("  - Data loading managed by load_dataset module")
    print("  - Using default paths from load_dataset.DEFAULT_*_DIR")

    print("\nModel Paths:")
    print(f"  - Target model: {config.TARGET_MODEL_DIR}")
    print(f"  - Shadow model: {config.SHADOW_MODEL_DIR}")
    print(f"  - Attack model: {config.ATTACK_MODEL_DIR}")

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

    # Verify target model exists
    if os.path.exists(config.TARGET_MODEL_DIR):
        print(f"\n✅ Target model found: {config.TARGET_MODEL_DIR}")
    else:
        print(f"\n⚠️ Target model not found: {config.TARGET_MODEL_DIR}")
        print("   Make sure you have a pre-trained target model before running evaluation.")

    # Verify data directories exist by loading DataLoaders
    try:
        train_loader, val_loader, test_loader = load_data_mia(
            batch_size=2,
            num_workers=0,
            augment_train=False,
        )
        print(f"\n✅ TRAIN set found: {len(train_loader.dataset)} images")
        print(f"✅ VAL set found: {len(val_loader.dataset)} images")
        print(f"✅ TEST set found: {len(test_loader.dataset)} images")
        return True
    except Exception as e:
        print(f"\n❌ Failed to load datasets: {e}")
        return False


def step2_train_shadow(
    config: PipelineConfig,
    *,
    train_loader=None,
    val_loader=None,
    test_loader=None,
):
    """
    Step 2: Train Shadow Model
    Train shadow model on TEST set using official pretrained weights
    """
    print_step(2, 4, "Shadow Model Training")

    # For shadow model: train on TEST, validate on VAL
    # Note: load_data_mia returns (train_loader, val_loader, test_loader)
    # But we need (test_loader as train, val_loader as val) for shadow training
    if train_loader is None or val_loader is None or test_loader is None:
        print("Loading datasets using load_dataset.load_data_mia()...")
        _, val_loader_original, test_loader = load_data_mia(
            batch_size=config.SHADOW_BATCH_SIZE,
            num_workers=config.workers,
            augment_train=config.TRANSFORM,
        )
        shadow_train_loader = test_loader
        shadow_val_loader = val_loader_original
    else:
        shadow_train_loader = test_loader
        shadow_val_loader = val_loader

    print("Training shadow model with configuration:")
    print(f"  - Training data: TEST set (shadow's members)")
    print(f"  - Validation data: VAL set")
    print(f"  - Using pretrained: {config.SHADOW_USE_PRETRAINED}")
    print(f"  - Epochs: {config.SHADOW_EPOCHS}")
    print(f"  - Batch size: {config.SHADOW_BATCH_SIZE}")
    print(f"  - Learning rate: {config.SHADOW_LR}")
    print(f"  - Output: {config.SHADOW_MODEL_DIR}")

    start_time = time.time()
    train_shadow_with_config(config, train_loader=shadow_train_loader, val_loader=shadow_val_loader)
    elapsed = time.time() - start_time

    print(f"\n✅ Shadow model training completed in {elapsed/60:.2f} minutes")
    print(f"   Model saved to: {config.SHADOW_MODEL_DIR}")

    return True


def step3_train_attack(
    config: PipelineConfig,
    *,
    train_loader=None,
    test_loader=None,
):
    """
    Step 3: Train Attack Model
    Train attack model using shadow model's outputs
    """
    print_step(3, 4, "Attack Model Training")

    # Check shadow model exists
    if not os.path.exists(config.SHADOW_MODEL_DIR):
        print(f"❌ Shadow model not found: {config.SHADOW_MODEL_DIR}")
        print("   Please run step 2 (shadow model training) first.")
        return False

    import numpy as np

    if train_loader is None or test_loader is None:
        # Load datasets once in pipeline
        print("\n=== Loading datasets for attack model training ===")
        train_loader, _, test_loader = load_data_mia(
            batch_size=2,  # Small batch for loading
            num_workers=0,
            augment_train=False,
        )

    # 成员样本：TEST集（影子模型训练所用）
    member_img_paths = [str(test_loader.dataset.images[i]) for i in range(len(test_loader.dataset))]

    # 非成员样本：TRAIN集（影子模型未见过的数据）
    # 进行下采样，使其数量与成员样本相同
    all_train_img_paths = [str(train_loader.dataset.images[i]) for i in range(len(train_loader.dataset))]

    # 随机下采样，使非成员样本数量与成员样本数量相同
    np.random.seed(42)  # 设置随机种子以保证可复现性
    num_members = len(member_img_paths)
    if len(all_train_img_paths) > num_members:
        sampled_indices = np.random.choice(len(all_train_img_paths), num_members, replace=False)
        nonmember_img_paths = [all_train_img_paths[i] for i in sampled_indices]
    else:
        nonmember_img_paths = all_train_img_paths

    print(f"✅ Prepared attack training data:")
    print(f"  - Member samples (TEST set): {len(member_img_paths)}")
    print(f"  - Non-member samples (TRAIN set downsampled): {len(nonmember_img_paths)}")
    print(f"  - Data balance ratio: 1:1")

    print("\nTraining attack model with configuration:")
    print(f"  - Shadow model: {config.SHADOW_MODEL_DIR}")
    print(f"  - Member samples: TEST set (shadow's training data)")
    print(f"  - Non-member samples: TRAIN set downsampled")
    print(f"  - Attack model type: {config.ATTACK_MODEL_TYPE}")
    print(f"  - Epochs: {config.ATTACK_EPOCHS}")
    print(f"  - Canvas size: {config.CANVAS_SIZE}x{config.CANVAS_SIZE}")
    print(f"  - Output: {config.ATTACK_MODEL_DIR}")

    start_time = time.time()
    train_attack_with_config(config, member_img_paths, nonmember_img_paths)
    elapsed = time.time() - start_time

    print(f"\n✅ Attack model training completed in {elapsed/60:.2f} minutes")

    return True


def step4_evaluate(
    config: PipelineConfig,
    *,
    train_loader=None,
    test_loader=None,
):
    """
    Step 4: Evaluate Attack Model
    Evaluate attack on target model
    """
    print_step(4, 4, "Attack Model Evaluation")

    # Check models exist
    if not os.path.exists(config.TARGET_MODEL_DIR):
        print(f"❌ Target model not found: {config.TARGET_MODEL_DIR}")
        return False

    attack_model_path = config.ATTACK_MODEL_DIR
    if not os.path.exists(attack_model_path):
        # Try to find any attack model
        attack_dir = os.path.dirname(attack_model_path)
        if os.path.exists(attack_dir):
            for f in ['best.pth', 'last.pth']:
                alt_path = os.path.join(attack_dir, f)
                if os.path.exists(alt_path):
                    attack_model_path = alt_path
                    print(f"Using attack model: {alt_path}")
                    break
            else:
                print(f"❌ Attack model not found in: {attack_dir}")
                return False
        else:
            print(f"❌ Attack model directory not found: {attack_dir}")
            print("   Please run step 3 (attack model training) first.")
            return False

    if train_loader is None or test_loader is None:
        # Load datasets once in pipeline
        print("\n=== Loading datasets for attack evaluation ===")
        train_loader, _, test_loader = load_data_mia(
            batch_size=2,  # Small batch for loading
            num_workers=0,
            augment_train=False,
        )

    # 目标模型的成员样本（TRAIN集）
    target_member_imgs = [str(train_loader.dataset.images[i]) for i in range(len(train_loader.dataset))]

    # 目标模型的非成员样本（TEST集）
    target_nonmember_imgs = [str(test_loader.dataset.images[i]) for i in range(len(test_loader.dataset))]

    # 获取配置的样本数量限制
    member_samples = config.MIA_MEMBER_SAMPLES
    nonmember_samples = config.MIA_NONMEMBER_SAMPLES

    # 限制样本数量
    target_member_imgs = target_member_imgs[:member_samples]
    target_nonmember_imgs = target_nonmember_imgs[:nonmember_samples]

    print(f"✅ Prepared evaluation data:")
    print(f"  - Member samples (TRAIN set): {len(target_member_imgs)}")
    print(f"  - Non-member samples (TEST set): {len(target_nonmember_imgs)}")

    print("\nEvaluating attack with configuration:")
    print(f"  - Target model: {config.TARGET_MODEL_DIR}")
    print(f"  - Attack model: {attack_model_path}")
    print(f"  - Member samples: {len(target_member_imgs)} from TRAIN set")
    print(f"  - Non-member samples: {len(target_nonmember_imgs)} from TEST set")
    print(f"  - Results: {config.RESULTS_FILE}")

    start_time = time.time()
    evaluate_attack_with_config(config, target_member_imgs, target_nonmember_imgs)
    elapsed = time.time() - start_time

    print(f"\n✅ Attack evaluation completed in {elapsed/60:.2f} minutes")
    print(f"   Results saved to: {config.RESULTS_FILE}")

    return True


def run_pipeline(args, config: PipelineConfig):
    """Run the complete MIA pipeline or specific steps"""
    print_header("MIA Pipeline for Faster R-CNN Object Detection")

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Selected steps: {args.steps if args.steps else 'all'}")

    # Determine which steps to run
    if args.steps:
        steps = [int(s) for s in args.steps.split(',')]
    else:
        steps = [1, 2, 3, 4]

    total_start = time.time()
    results = {}

    # Step 1: Configuration
    if 1 in steps:
        results[1] = step1_configure(config)
        if not results[1] and args.steps is None:
            print("\n❌ Configuration check failed. Stopping pipeline.")
            return False

    # Step 2: Train Shadow Model
    if 2 in steps:
        results[2] = step2_train_shadow(config)
        if not results[2]:
            print("\n❌ Shadow model training failed.")
            if args.steps is None:
                return False

    # Step 3: Train Attack Model
    if 3 in steps:
        results[3] = step3_train_attack(config)
        if not results[3]:
            print("\n❌ Attack model training failed.")
            if args.steps is None:
                return False

    # Step 4: Evaluate
    if 4 in steps:
        results[4] = step4_evaluate(config)
        if not results[4]:
            print("\n❌ Attack evaluation failed.")
            return False

    total_elapsed = time.time() - total_start

    # Summary
    print_header("Pipeline Summary")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {total_elapsed/60:.2f} minutes")

    print("\nStep Results:")
    step_names = {
        1: "Configuration",
        2: "Shadow Model Training",
        3: "Attack Model Training",
        4: "Attack Evaluation"
    }
    for step in steps:
        status = "✅ Success" if results.get(step, False) else "❌ Failed"
        print(f"  Step {step} ({step_names[step]}): {status}")

    all_success = all(results.get(s, False) for s in steps)
    if all_success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n⚠️ Some steps failed. Check the logs above for details.")

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="MIA Pipeline for Faster R-CNN Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py

  # Run only configuration check
  python pipeline.py --steps 1

  # Run shadow and attack model training
  python pipeline.py --steps 2,3

  # Run only evaluation (requires trained models)
  python pipeline.py --steps 4

  # Run with specific GPU
  python pipeline.py --gpu 1

  # Override shadow model epochs
  python pipeline.py --shadow-epochs 50

Steps:
  1: Configuration - Display and verify settings
  2: Shadow Model Training - Train on TEST set
  3: Attack Model Training - Train using shadow model
  4: Attack Evaluation - Evaluate on target model
        """
    )

    # Pipeline control
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated list of steps to run (e.g., "1,2,3,4" or "2,3")')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use')

    # Dataset paths
    parser.add_argument('--train-dir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Validation data directory')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Test data directory')

    # Model paths
    parser.add_argument('--target-model', type=str, default=None,
                        help='Target model path')
    parser.add_argument('--shadow-model', type=str, default=None,
                        help='Shadow model output path')
    parser.add_argument('--attack-model', type=str, default=None,
                        help='Attack model output path')

    # Shadow model training
    parser.add_argument('--shadow-epochs', type=int, default=None,
                        help='Shadow model training epochs')
    parser.add_argument('--shadow-batch-size', type=int, default=None,
                        help='Shadow model batch size')
    parser.add_argument('--shadow-lr', type=float, default=None,
                        help='Shadow model learning rate')

    # Attack model training
    parser.add_argument('--attack-epochs', type=int, default=None,
                        help='Attack model training epochs')
    parser.add_argument('--attack-batch-size', type=int, default=None,
                        help='Attack model batch size')
    parser.add_argument('--attack-type', type=str, choices=['alex', 'shallow'], default=None,
                        help='Attack model architecture')

    # MIA evaluation
    parser.add_argument('--member-samples', type=int, default=None,
                        help='Number of member samples for evaluation')
    parser.add_argument('--nonmember-samples', type=int, default=None,
                        help='Number of non-member samples for evaluation')

    args = parser.parse_args()

    # Create config with defaults
    config = PipelineConfig()

    # Override config with command line arguments
    if args.gpu is not None:
        config.gpu_id = args.gpu
    # Note: Data directory arguments are ignored - paths managed by load_dataset module
    if args.target_model is not None:
        config.TARGET_MODEL_DIR = args.target_model
    if args.shadow_model is not None:
        config.SHADOW_MODEL_DIR = args.shadow_model
    if args.attack_model is not None:
        config.ATTACK_MODEL_DIR = args.attack_model
    if args.shadow_epochs is not None:
        config.SHADOW_EPOCHS = args.shadow_epochs
    if args.shadow_batch_size is not None:
        config.SHADOW_BATCH_SIZE = args.shadow_batch_size
    if args.shadow_lr is not None:
        config.SHADOW_LR = args.shadow_lr
    if args.attack_epochs is not None:
        config.ATTACK_EPOCHS = args.attack_epochs
    if args.attack_batch_size is not None:
        config.ATTACK_BATCH_SIZE = args.attack_batch_size
    if args.attack_type is not None:
        config.ATTACK_MODEL_TYPE = args.attack_type
    if args.member_samples is not None:
        config.MIA_MEMBER_SAMPLES = args.member_samples
    if args.nonmember_samples is not None:
        config.MIA_NONMEMBER_SAMPLES = args.nonmember_samples

    success = run_pipeline(args, config)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
