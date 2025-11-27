import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import cv2
import gc
import yaml
import glob
from PIL import Image
import warnings

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF

# 过滤掉torchvision的弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Use non-interactive backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure matplotlib fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


# NOTE: This module must be called from pipeline.py
# No standalone execution is supported - config must be passed from pipeline


# 攻击模型定义 (与train_attacker.py保持一致)
class AttackModel(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=128, model_type='shallow'):
        super(AttackModel, self).__init__()

        if model_type == 'shallow':
            # 浅层卷积攻击模型，适配3通道二维输入
            self.model = nn.Sequential(
                # 卷积层提取空间特征
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 全局平均池化
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                # 全连接层分类
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, 2)
            )
        elif model_type == 'alex':
            # AlexNet风格的攻击模型，适配3通道二维输入
            self.model = nn.Sequential(
                # 卷积层提取空间特征
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 全局平均池化
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                # 全连接层分类
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
        else:
            raise ValueError(f"未知的攻击模型类型: {model_type}")

    def forward(self, x):
        return self.model(x)


def logscore(a, log_type=2):
    """计算log分数，添加输入值范围检查避免无效值"""
    # 确保输入值在有效范围内，避免除零错误和无效值
    a = np.clip(a, 0, 0.999999999)
    if log_type == 2:
        return -np.log2(1 - a + 1e-20)
    elif log_type > 0:
        return -np.log(1 - a + 1e-20)
    else:
        return a


def save_canvas_image(canvas, save_path, label=None, cmap='hot', global_max=None):
    """
    保存canvas图像到文件

    Args:
        canvas: 二维canvas数组
        save_path: 保存路径
        label: 可选的标签信息
        cmap: 颜色映射
        global_max: 全局最大值，用于统一尺度
    """
    plt.figure(figsize=(6, 6))
    # 使用归一化器确保颜色映射一致
    if global_max is not None:
        # 如果提供了全局最大值，使用它来确保所有图像的尺度一致
        norm = Normalize(vmin=0, vmax=global_max)
    else:
        # 否则使用当前画布的最大值
        norm = Normalize(vmin=0, vmax=np.max(canvas) if np.max(canvas) > 0 else 1)
    plt.imshow(canvas, cmap=cmap, norm=norm)
    plt.colorbar(label='Score Intensity')

    if label is not None:
        plt.title(f'Canvas (Label: {int(label)})')
    else:
        plt.title('Canvas Visualization')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_canvas_data(dataset, canvas_size=300, canvas_type="original", ball_size=30, normalize=True, log_score_type=2,
                     save_samples=0, save_dir=None, device=None, global_normalize=False):
    """
    将特征点集转换为二维canvas表示

    Args:
        dataset: 包含(特征, 标签)元组的列表
        canvas_size: canvas的尺寸（canvas_size × canvas_size）
        canvas_type: 'original'（使用边界框原始区域）或'uniform'（使用圆形区域）
        ball_size: uniform模式下圆的直径
        normalize: 是否对canvas进行归一化
        log_score_type: log分数类型，0表示不使用logscore，1表示自然对数，2表示以2为底的对数
        save_samples: 保存的样本数量
        save_dir: 保存目录
        device: 设备
        global_normalize: 是否使用全局归一化

    Returns:
        转换后的数据集，包含(canvas, 标签)元组的列表
    """
    canvas_dataset = []
    sample_count = 0

    # 如果启用全局归一化，先计算所有画布的最大值
    global_max = 1.0
    if global_normalize:
        all_canvas_max = []
        # 第一次遍历：计算所有画布的最大值
        for idx, item in enumerate(dataset):
            feats, label = item
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

            # 处理每个特征点
            for feat in feats:
                # 跳过零填充的特征
                if np.sum(feat) < 1e-5:
                    continue

                # 提取边界框坐标和分数
                x0, y0, x1, y1 = feat[:4]  # 归一化坐标 [0, 1]
                score = feat[4]

                # 应用logscore特征放大
                if log_score_type > 0:
                    score = logscore(score, log_score_type)

                # 将归一化坐标转换为canvas坐标
                x0_canvas = int(x0 * canvas_size)
                y0_canvas = int(y0 * canvas_size)
                x1_canvas = int(x1 * canvas_size)
                y1_canvas = int(y1 * canvas_size)

                # 确保坐标在有效范围内
                x0_canvas = max(0, x0_canvas)
                y0_canvas = max(0, y0_canvas)
                x1_canvas = min(canvas_size - 1, x1_canvas)
                y1_canvas = min(canvas_size - 1, y1_canvas)

                if canvas_type == 'uniform':
                    # 圆形模式：在中心点周围绘制圆形区域
                    x_c = (x0_canvas + x1_canvas) // 2
                    y_c = (y0_canvas + y1_canvas) // 2
                    radius = int(ball_size // 2)

                    # 创建圆形掩码
                    y_range = np.arange(max(0, y_c - radius), min(canvas_size, y_c + radius + 1))
                    x_range = np.arange(max(0, x_c - radius), min(canvas_size, x_c + radius + 1))

                    for y in y_range:
                        for x in x_range:
                            if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2:
                                canvas[y, x] += score
                else:  # original模式
                    # 原始模式：直接在边界框区域内填充分数
                    canvas[y0_canvas:y1_canvas + 1, x0_canvas:x1_canvas + 1] += score

            # 保存最大值用于全局归一化
            if np.max(canvas) > 0:
                all_canvas_max.append(np.max(canvas))

        # 计算全局最大值
        if all_canvas_max:
            global_max = np.max(all_canvas_max)
        else:
            global_max = 1.0

        print(f"全局最大值: {global_max}")

    # 第二次遍历：生成最终的画布数据
    for idx, item in enumerate(dataset):
        feats, label = item
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

        # 处理每个特征点
        for feat in feats:
            # 跳过零填充的特征
            if np.sum(feat) < 1e-5:
                continue

            # 提取边界框坐标和分数
            x0, y0, x1, y1 = feat[:4]  # 归一化坐标 [0, 1]
            score = feat[4]

            # 应用logscore特征放大
            if log_score_type > 0:
                score = logscore(score, log_score_type)

            # 将归一化坐标转换为canvas坐标
            x0_canvas = int(x0 * canvas_size)
            y0_canvas = int(y0 * canvas_size)
            x1_canvas = int(x1 * canvas_size)
            y1_canvas = int(y1 * canvas_size)

            # 确保坐标在有效范围内
            x0_canvas = max(0, x0_canvas)
            y0_canvas = max(0, y0_canvas)
            x1_canvas = min(canvas_size - 1, x1_canvas)
            y1_canvas = min(canvas_size - 1, y1_canvas)

            if canvas_type == 'uniform':
                # 圆形模式：在中心点周围绘制圆形区域
                x_c = (x0_canvas + x1_canvas) // 2
                y_c = (y0_canvas + y1_canvas) // 2
                radius = int(ball_size // 2)

                # 创建圆形掩码
                y_range = np.arange(max(0, y_c - radius), min(canvas_size, y_c + radius + 1))
                x_range = np.arange(max(0, x_c - radius), min(canvas_size, x_c + radius + 1))

                for y in y_range:
                    for x in x_range:
                        if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2:
                            canvas[y, x] += score
            else:  # original模式
                # 原始模式：直接在边界框区域内填充分数
                canvas[y0_canvas:y1_canvas + 1, x0_canvas:x1_canvas + 1] += score

        # 如果启用全局归一化，使用固定上限值30
        if global_normalize:
            # 使用固定值30作为上限，超过的值都显示为30
            canvas = np.clip(canvas, 0, 50)
        # 归一化处理 - 与origin代码保持一致，使用均值进行归一化
        elif normalize and np.sum(canvas) > 0:
            canvas = canvas / canvas.mean()

        # 输出一些画布统计信息用于调试
        if idx < 5:  # 只输出前5个样本的信息
            print(
                f"画布 {idx} - 最小值: {np.min(canvas):.6f}, 最大值: {np.max(canvas):.6f}, 均值: {np.mean(canvas):.6f}")

        canvas_dataset.append((canvas, label))

        # 保存样本
        if save_samples > 0 and save_dir is not None and sample_count < save_samples:
            # 平衡选择正样本和负样本
            if sample_count < save_samples // 2 and label == 1:
                save_path = os.path.join(save_dir, f'canvas_positive_{sample_count}.png')
                save_canvas_image(canvas, save_path, label=label, cmap='hot',
                                  global_max=50 if global_normalize else None)
                sample_count += 1
            elif sample_count >= save_samples // 2 and label == 0 and sample_count < save_samples:
                save_path = os.path.join(save_dir, f'canvas_negative_{sample_count - save_samples // 2}.png')
                save_canvas_image(canvas, save_path, label=label, cmap='hot',
                                  global_max=50 if global_normalize else None)
                sample_count += 1

    return canvas_dataset


# 加载Faster R-CNN模型
def load_fasterrcnn_model(model_path, num_classes, device):
    """加载Faster R-CNN模型（与目标/影子模型加载逻辑一致）"""
    # 初始化模型，不使用预训练权重
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FasterRCNN
    from torchvision.models import ResNet50_Weights

    # 创建不带预训练权重的ResNet backbone
    backbone = resnet_fpn_backbone('resnet50', weights=None)
    # 初始化Faster R-CNN模型
    model = FasterRCNN(backbone, num_classes=num_classes + 1)

    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # 切换为评估模式
    model.eval()
    model.to(device)
    return model


# NOTE: For inference, we directly process image paths in batches
# No need for a custom Dataset class


def generate_pointsets(model, img_paths, img_size, device, max_len=6000, log_score_type=2, num_logit_feature=1,
                       regard_in_set=True, max_samples=None, batch_size=4):
    """
    使用模型生成特征点集，包含边界框坐标、分数和标签信息

    Args:
        model: 目标/影子模型
        img_paths: 图像路径列表
        img_size: 图像尺寸
        device: 设备
        max_len: 最大特征长度
        log_score_type: log分数类型，0表示不使用logscore，1表示自然对数，2表示以2为底的对数
        num_logit_feature: logit特征数量
        regard_in_set: 是否将数据视为在训练集中
        max_samples: 最大处理样本数量，None表示处理所有样本
        batch_size: 处理图像的批次大小

    Returns:
        list: 包含(特征, 标签)元组的列表
    """
    model.eval()
    dataset = []
    max_feat_len = 0
    min_feat_len = float('inf')
    sample_count = 0

    # 如果img_paths是文件路径字符串，则读取其中的图像路径列表
    if isinstance(img_paths, str):
        with open(img_paths, 'r') as f:
            img_paths = [line.strip() for line in f.readlines()]

    total_images = min(len(img_paths), max_samples) if max_samples else len(img_paths)
    img_paths_subset = img_paths[:total_images]

    with torch.no_grad():
        # 分批处理图像（直接处理，不使用Dataset）
        for batch_start in tqdm(range(0, len(img_paths_subset), batch_size), desc="生成特征点集"):
            batch_end = min(batch_start + batch_size, len(img_paths_subset))
            batch_paths = img_paths_subset[batch_start:batch_end]

            # 加载并预处理批次图像
            images = []
            orig_sizes = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size
                orig_sizes.append((orig_w, orig_h))

                img_resized = img.resize((img_size, img_size))
                img_tensor = TF.to_tensor(img_resized)
                # ImageNet normalization
                img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                images.append(img_tensor.to(device))

            # 模型推理
            outputs = model(images)

            # 处理每个图像的输出
            for i, output in enumerate(outputs):
                orig_w, orig_h = orig_sizes[i]

                # 获取边界框坐标 [x1, y1, x2, y2]
                boxes = output.get('boxes', torch.empty((0, 4))).cpu().numpy()
                # 获取置信度分数
                scores = output.get('scores', torch.empty((0,))).cpu().numpy()
                # 获取类别标签
                labels = output.get('labels', torch.empty((0,), dtype=torch.int64)).cpu().numpy()

                # 应用置信度阈值过滤（与测试脚本保持一致）
                score_threshold = 0.05
                high_confidence_indices = scores > score_threshold
                boxes = boxes[high_confidence_indices]
                scores = scores[high_confidence_indices]
                labels = labels[high_confidence_indices]

                # 创建特征点集
                if len(boxes) > 0:
                    # 确保分数在有效范围内，避免除零错误
                    scores = np.clip(scores, 0, 0.999999999)
                    # 计算log分数
                    if log_score_type > 0:
                        scores = logscore(scores, log_score_type)

                    # 归一化边界框坐标
                    bboxes_normalized = np.zeros((len(boxes), 4 + num_logit_feature * 2), dtype=np.float32)
                    bboxes_normalized[:, 0] = boxes[:, 0] / orig_w  # x1归一化
                    bboxes_normalized[:, 1] = boxes[:, 1] / orig_h  # y1归一化
                    bboxes_normalized[:, 2] = boxes[:, 2] / orig_w  # x2归一化
                    bboxes_normalized[:, 3] = boxes[:, 3] / orig_h  # y2归一化
                    bboxes_normalized[:, 4] = scores  # 分数
                    # 注意：Faster R-CNN中0是背景类，实际类别从1开始
                    # 为了在后续处理中保持一致性，我们将标签减1
                    adjusted_labels = np.maximum(0, labels - 1)  # 确保不会出现负数
                    bboxes_normalized[:, 4 + num_logit_feature] = adjusted_labels  # 标签

                    # 更新特征长度统计
                    current_len = len(bboxes_normalized)
                    max_feat_len = max(max_feat_len, current_len)
                    min_feat_len = min(min_feat_len, current_len)

                    # 截断到最大长度
                    if current_len > max_len:
                        bboxes_normalized = bboxes_normalized[:max_len]

                    # 填充到最大长度
                    if current_len < max_len:
                        padded_bboxes = np.zeros((max_len, bboxes_normalized.shape[1]), dtype=np.float32)
                        padded_bboxes[:current_len] = bboxes_normalized
                    else:
                        padded_bboxes = bboxes_normalized
                else:
                    # 如果没有预测框，使用全零特征
                    padded_bboxes = np.zeros((max_len, 4 + num_logit_feature * 2), dtype=np.float32)

                # 添加到数据集
                label = 1.0 if regard_in_set else 0.0
                dataset.append((padded_bboxes, label))
                sample_count += 1

                if max_samples and sample_count >= max_samples:
                    break

            if max_samples and sample_count >= max_samples:
                break

    print(f"特征点集生成完成 - 最大长度: {max_feat_len}, 最小长度: {min_feat_len}, 样本数: {len(dataset)}")
    return dataset


def train_attack_model(config, member_img_paths, nonmember_img_paths):
    """
    Train attack model using the simplified MIA flow:
    - Member samples: TEST set (shadow model's training data)
    - Non-member samples: TRAIN set downsampled (shadow model's non-member data)

    Args:
        config: Configuration object with all required parameters
        member_img_paths: List of image paths for member samples (from TEST set)
        nonmember_img_paths: List of image paths for non-member samples (from TRAIN set, downsampled)
    """

    # 打印canvas_type配置以确认
    print(f"当前canvas_type配置: {config.CANVAS_TYPE}")

    # 创建设备
    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载影子模型（使用Faster R-CNN）
    shadow_model_path = config.SHADOW_MODEL_DIR
    print(f"加载影子模型: {shadow_model_path}")

    shadow_model = load_fasterrcnn_model(
        model_path=shadow_model_path,
        num_classes=config.num_classes,
        device=device
    )
    print(f"影子模型加载完成: {shadow_model_path}")

    # 数据集路径已由 pipeline 提供
    print(f"\n=== 使用 Pipeline 提供的数据集路径 ===")
    print(f"成员样本数量（TEST集）: {len(member_img_paths)}")
    print(f"非成员样本数量（TRAIN集下采样）: {len(nonmember_img_paths)}")
    print(f"数据平衡比例: {len(member_img_paths)} : {len(nonmember_img_paths)}")

    # 划分攻击模型的训练/验证集
    print("\n=== 生成攻击模型特征 ===")
    split_ratio = config.train_split_ratio

    # 成员样本划分（用于攻击模型训练/验证）
    train_in_size = int(len(member_img_paths) * split_ratio)
    attack_train_in = member_img_paths[:train_in_size]
    attack_val_in = member_img_paths[train_in_size:]

    # 非成员样本划分（用于攻击模型训练/验证）
    train_out_size = int(len(nonmember_img_paths) * split_ratio)
    attack_train_out = nonmember_img_paths[:train_out_size]
    attack_val_out = nonmember_img_paths[train_out_size:]

    print("生成攻击模型训练数据...")
    print(f"攻击模型训练集 - 成员: {len(attack_train_in)}, 非成员: {len(attack_train_out)}")
    print(f"攻击模型验证集 - 成员: {len(attack_val_in)}, 非成员: {len(attack_val_out)}")

    # 1. 成员样本：TEST集（影子模型的训练数据）
    in_dataset = generate_pointsets(
        shadow_model,
        attack_train_in,
        config.img_size,
        device,
        max_len=config.MAX_LEN,
        log_score_type=config.LOG_SCORE,
        regard_in_set=True,
        max_samples=None,  # 使用所有数据
        batch_size=config.batch_size
    )

    # 2. 非成员样本：TRAIN集下采样（影子模型未见过的数据）
    out_dataset = generate_pointsets(
        shadow_model,
        attack_train_out,
        config.img_size,
        device,
        max_len=config.MAX_LEN,
        log_score_type=config.LOG_SCORE,
        regard_in_set=False,
        max_samples=None,  # 使用所有数据
        batch_size=config.batch_size
    )

    # 打印各样本数目
    print(f"成员样本数（TEST集）: {len(in_dataset)}")
    print(f"非成员样本数（TRAIN集下采样）: {len(out_dataset)}")

    # 合并数据集
    all_dataset = in_dataset + out_dataset
    print(f"总样本数: {len(all_dataset)}")

    # 分离特征和标签
    all_features = np.array([item[0] for item in all_dataset])
    all_labels = np.array([item[1] for item in all_dataset])

    # 统计正样本和负样本数量
    num_in_set = np.sum(all_labels)
    num_out_set = len(all_labels) - num_in_set
    print(f"正样本数（在训练集中）: {num_in_set}")
    print(f"负样本数（不在训练集中）: {num_out_set}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42
    )

    # 打印训练集和验证集中的样本数量
    train_in_set = np.sum(y_train)
    train_out_set = len(y_train) - train_in_set
    val_in_set = np.sum(y_val)
    val_out_set = len(y_val) - val_in_set

    print(f"\n攻击模型训练数据: {len(X_train)}")
    print(f"  - 训练集正样本数: {train_in_set}")
    print(f"  - 训练集负样本数: {train_out_set}")
    print(f"攻击模型验证数据: {len(X_val)}")
    print(f"  - 验证集正样本数: {val_in_set}")
    print(f"  - 验证集负样本数: {val_out_set}")

    # 从配置获取batch_size
    batch_size = getattr(config, 'batch_size', 8)
    print(f"设置训练批次大小: {batch_size}")

    # 将特征转换为canvas表示
    canvas_size = config.input_size  # 使用配置中的输入尺寸
    # 使用从config.py中加载的canvas_type配置
    canvas_type = config.CANVAS_TYPE
    print(f"将特征转换为canvas表示 (尺寸: {canvas_size}x{canvas_size}, 类型: {canvas_type})")

    # 准备数据集格式：[(特征, 标签), ...]
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))

    # 创建保存canvas图像的目录
    canvas_images_dir = 'canvas_images'
    os.makedirs(canvas_images_dir, exist_ok=True)

    # 转换为canvas表示并保存示例图像
    print(f"将特征转换为canvas表示并保存示例图像到: {canvas_images_dir}")
    canvas_train_data = make_canvas_data(
        train_data,
        canvas_size=canvas_size,
        canvas_type=canvas_type,
        normalize=config.NORMALIZE_CANVAS,  # 使用配置中的归一化选项
        log_score_type=config.LOG_SCORE,
        save_samples=20,  # 保存20张样本图（10张正样本，10张负样本）
        save_dir=canvas_images_dir,
        device=device,
        global_normalize=True
    )
    canvas_val_data = make_canvas_data(
        val_data,
        canvas_size=canvas_size,
        canvas_type=canvas_type,
        normalize=config.NORMALIZE_CANVAS,  # 使用配置中的归一化选项
        log_score_type=config.LOG_SCORE,
        global_normalize=True
    )

    print(f"已保存canvas示例图像到 {canvas_images_dir}")

    # 分离canvas特征和标签
    X_train_canvas = np.array([item[0] for item in canvas_train_data])
    y_train = np.array([item[1] for item in canvas_train_data])
    X_val_canvas = np.array([item[0] for item in canvas_val_data])
    y_val = np.array([item[1] for item in canvas_val_data])

    # 释放一些内存
    del train_data, val_data, canvas_train_data, canvas_val_data
    import gc
    gc.collect()

    # 初始化攻击模型（不再需要input_dim参数）
    attack_model = AttackModel(
        model_type=config.ATTACK_MODEL  # 使用配置中的攻击模型类型
    ).to(device)

    # 检查模型参数是否正确初始化
    print(f"初始化模型类型: {config.ATTACK_MODEL}")
    total_params = sum(p.numel() for p in attack_model.parameters() if p.requires_grad)
    print(f"可训练参数总数: {total_params:,}")

    # 设置优化器和损失函数 - 使用配置中的学习率和权重衰减
    optimizer = optim.Adam(
        attack_model.parameters(),
        lr=config.lr_attack,
        weight_decay=config.weight_decay
    )

    # 统计训练集类别分布
    num_members = np.sum(y_train)
    num_non_members = len(y_train) - num_members
    print(f"训练集类别分布: 成员={num_members}, 非成员={num_non_members}")
    print(f"类别比例: 成员/非成员 = {num_members / num_non_members:.2f}")

    # 使用标准的CrossEntropyLoss损失函数（不使用类别权重）
    # 由于采用了数据平衡策略（下采样），两个类别数量应该接近
    criterion = nn.CrossEntropyLoss()

    # 添加学习率调度器 - 帮助模型更好地收敛
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控 AUC（越大越好）
        factor=0.5,  # 学习率衰减因子
        patience=5,  # 5个epoch没有改善就降低学习率
        min_lr=1e-6
    )
    print(f"学习率调度器已启用: ReduceLROnPlateau (mode=max, factor=0.5, patience=5)")

    print(f"Canvas特征维度: {X_train_canvas.shape}")
    print(f"学习率: {config.lr_attack}, 权重衰减: {config.weight_decay}")

    # 创建带exp编号的保存目录
    base_save_dir = 'runs/attacker_train'
    exp_index = 1
    save_dir = os.path.join(base_save_dir, f'exp')
    while os.path.exists(save_dir):
        exp_index += 1
        save_dir = os.path.join(base_save_dir, f'exp{exp_index}' if exp_index > 1 else 'exp')
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型将保存至: {save_dir}")

    # 训练攻击模型
    best_train_loss = float('inf')
    best_val_auc = 0.0  # 初始化最佳验证AUC
    num_epochs = config.EPOCHS  # 使用配置中的训练轮数
    early_stopping_patience = 20  # 早停耐心值
    early_stopping_counter = 0  # 早停计数器

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        attack_model.train()
        train_loss = 0.0

        # 随机打乱训练数据索引
        indices = np.random.permutation(len(X_train_canvas))

        # 计算总批次数
        total_batches = len(range(0, len(X_train_canvas), batch_size))

        # 分批处理，添加批次进度条
        with tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=False) as batch_pbar:
            for i in range(0, len(X_train_canvas), batch_size):
                # 获取批次数据
                batch_indices = indices[i:i + batch_size]
                X_batch_canvas = X_train_canvas[batch_indices]
                y_batch = y_train[batch_indices]

                # 将单通道转换为3通道
                batch_size_current = len(X_batch_canvas)
                X_batch_3channel = np.zeros((batch_size_current, 3, canvas_size, canvas_size), dtype=np.float32)
                for c in range(3):
                    X_batch_3channel[:, c, :, :] = X_batch_canvas

                # 转换为PyTorch张量
                X_batch_tensor = torch.from_numpy(X_batch_3channel).float().to(device)
                y_batch_tensor = torch.from_numpy(y_batch).long().to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = attack_model(X_batch_tensor)
                loss = criterion(outputs, y_batch_tensor)

                # 反向传播和优化
                loss.backward()

                optimizer.step()

                train_loss += loss.item() * len(X_batch_tensor)

                # 更新批次进度条
                batch_pbar.update(1)

                # 释放批次内存
                del X_batch_tensor, y_batch_tensor, X_batch_3channel
                torch.cuda.empty_cache()

        # 计算平均训练损失
        train_loss /= len(X_train_canvas)

        # 验证阶段（每轮都进行）
        attack_model.eval()
        val_loss = 0.0
        val_preds_all = []
        val_probs_all = []
        val_labels_all = []

        # 随机打乱验证数据索引，确保每轮评估不同
        val_indices = np.random.permutation(len(X_val_canvas))

        with torch.no_grad():
            # 计算总验证批次数
            total_val_batches = len(range(0, len(X_val_canvas), batch_size))

            # 分批验证，添加验证进度条
            with tqdm(total=total_val_batches, desc="验证进度", unit="batch", leave=False) as val_pbar:
                for i in range(0, len(X_val_canvas), batch_size):
                    # 获取验证批次（使用随机索引）
                    batch_val_indices = val_indices[i:i + batch_size]
                    X_val_batch_canvas = X_val_canvas[batch_val_indices]
                    y_val_batch = y_val[batch_val_indices]

                    # 将单通道转换为3通道
                    val_batch_size = len(X_val_batch_canvas)
                    X_val_batch_3channel = np.zeros((val_batch_size, 3, canvas_size, canvas_size), dtype=np.float32)
                    for c in range(3):
                        X_val_batch_3channel[:, c, :, :] = X_val_batch_canvas

                    # 转换为PyTorch张量
                    X_val_batch_tensor = torch.from_numpy(X_val_batch_3channel).float().to(device)
                    y_val_batch_tensor = torch.from_numpy(y_val_batch).long().to(device)

                    # 前向传播
                    val_outputs = attack_model(X_val_batch_tensor)
                    batch_val_loss = criterion(val_outputs, y_val_batch_tensor)
                    val_loss += batch_val_loss.item() * len(X_val_batch_tensor)

                    # 计算预测和概率
                    _, val_batch_preds = torch.max(val_outputs, 1)
                    val_batch_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

                    # 收集结果
                    val_preds_all.extend(val_batch_preds.cpu().numpy())
                    val_probs_all.extend(val_batch_probs)
                    val_labels_all.extend(y_val_batch)

                    # 更新验证进度条
                    val_pbar.update(1)

                    # 释放验证批次内存
                    del X_val_batch_tensor, y_val_batch_tensor, X_val_batch_3channel
                    torch.cuda.empty_cache()

        # 计算平均验证损失
        val_loss /= len(X_val_canvas)

        # 计算准确率
        val_acc = accuracy_score(val_labels_all, val_preds_all)

        # 计算预测分布统计
        # 统计成员样本和非成员样本的判断情况
        val_labels_all = np.array(val_labels_all)
        val_preds_all = np.array(val_preds_all)

        # 成员样本（标签为1）的统计
        member_indices = val_labels_all == 1
        member_total = np.sum(member_indices)
        member_predicted_member = np.sum(val_preds_all[member_indices] == 1)
        member_predicted_nonmember = np.sum(val_preds_all[member_indices] == 0)

        # 非成员样本（标签为0）的统计
        nonmember_indices = val_labels_all == 0
        nonmember_total = np.sum(nonmember_indices)
        nonmember_predicted_member = np.sum(val_preds_all[nonmember_indices] == 1)
        nonmember_predicted_nonmember = np.sum(val_preds_all[nonmember_indices] == 0)

        # 计算AUC
        val_auc = roc_auc_score(val_labels_all, val_probs_all)

        # 输出每轮结果
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Val AUC: {val_auc:.4f}")

        # 输出成员样本和非成员样本的判断统计
        print(f"成员样本统计: 总数={member_total}, "
              f"被预测为成员={member_predicted_member}, "
              f"被预测为非成员={member_predicted_nonmember}")
        print(f"非成员样本统计: 总数={nonmember_total}, "
              f"被预测为成员={nonmember_predicted_member}, "
              f"被预测为非成员={nonmember_predicted_nonmember}")

        # 更新学习率调度器
        scheduler.step(val_auc)

        # 保存最佳模型（基于验证AUC）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model_path = os.path.join(save_dir, 'best.pth')
            torch.save({
                'model_state_dict': attack_model.state_dict(),
                'config': config.__dict__,
                'epoch': epoch + 1,
                'val_auc': val_auc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, model_path)
            print(f"保存最佳攻击模型: {model_path}, AUC: {best_val_auc:.4f}")
            early_stopping_counter = 0  # 重置早停计数器
        else:
            early_stopping_counter += 1  # 增加早停计数器

        # 早停机制（基于训练损失）
        if early_stopping_counter >= early_stopping_patience:
            print(f"早停机制触发，训练损失在{early_stopping_patience}个epoch内没有改善")
            break

        # 每轮迭代后清理内存
        gc.collect()
        torch.cuda.empty_cache()

    # 保存最终模型
    last_model_path = os.path.join(save_dir, 'last.pth')
    torch.save({
        'model_state_dict': attack_model.state_dict(),
        'config': config.__dict__,
        'epoch': num_epochs,
        'val_auc': val_auc,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, last_model_path)
    print(f"保存最终攻击模型: {last_model_path}")

    # 如果最佳模型不是最后一个epoch的模型，则也保存为best.pth
    if best_val_auc > val_auc:
        best_model_path = os.path.join(save_dir, 'best.pth')
        torch.save({
            'model_state_dict': attack_model.state_dict(),
            'config': config.__dict__,
            'epoch': epoch + 1,
            'val_auc': best_val_auc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, best_model_path)
        print(f"保存最佳攻击模型: {best_model_path}")

    # 保存特征数据用于后续分析
    data_path = os.path.join(save_dir, 'attack_model_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }, f)

    print(f"训练完成！攻击模型已保存到 {last_model_path}")
    print(f"最佳验证AUC: {best_val_auc:.4f}")


def train_attack_with_config(pipeline_config, member_img_paths, nonmember_img_paths):
    """
    Train attack model with configuration from pipeline.

    Args:
        pipeline_config: PipelineConfig object from pipeline.py
        member_img_paths: List of image paths for member samples
        nonmember_img_paths: List of image paths for non-member samples
    """

    # Create a config-like object
    class ConfigAdapter:
        pass

    cfg = ConfigAdapter()

    # Map pipeline config to expected attributes
    cfg.gpu_id = pipeline_config.gpu_id
    cfg.img_size = pipeline_config.img_size
    cfg.num_classes = pipeline_config.num_classes
    cfg.SAVE_MODEL = pipeline_config.SAVE_MODEL

    # Attack model specific
    cfg.EPOCHS = pipeline_config.ATTACK_EPOCHS
    cfg.batch_size = pipeline_config.ATTACK_BATCH_SIZE
    cfg.lr_attack = pipeline_config.ATTACK_LR
    cfg.weight_decay = pipeline_config.ATTACK_WEIGHT_DECAY
    cfg.ATTACK_MODEL = pipeline_config.ATTACK_MODEL_TYPE
    cfg.ATTACK_MODEL_DIR = pipeline_config.ATTACK_MODEL_DIR

    # Canvas/Feature settings
    cfg.input_size = pipeline_config.CANVAS_SIZE
    cfg.MAX_LEN = pipeline_config.MAX_LEN
    cfg.LOG_SCORE = pipeline_config.LOG_SCORE
    cfg.CANVAS_TYPE = pipeline_config.CANVAS_TYPE
    cfg.NORMALIZE_CANVAS = pipeline_config.NORMALIZE_CANVAS
    cfg.train_split_ratio = pipeline_config.TRAIN_SPLIT_RATIO

    # Model paths
    cfg.SHADOW_MODEL_DIR = pipeline_config.SHADOW_MODEL_DIR

    # Call train_attack_model with config and image paths
    train_attack_model(cfg, member_img_paths, nonmember_img_paths)


if __name__ == '__main__':
    print("=" * 70)
    print("ERROR: This script cannot be run directly!")
    print("=" * 70)
    print("\nThis script must be called from pipeline.py")
    print("\nUsage:")
    print("  python pipeline.py --steps 3")
    print("  python pipeline.py  # Run complete pipeline")
    print("\n" + "=" * 70)
    sys.exit(1)