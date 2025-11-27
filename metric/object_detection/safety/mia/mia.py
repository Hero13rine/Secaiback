import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import torch.nn as nn
from importlib import import_module
import glob
from tqdm import tqdm

# ===============================================================
# ✅ 兼容 PyTorch 2.6 的 torch.load 安全模式问题
# ===============================================================
import torch.serialization

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.core.multiarray.scalar,  # 添加这一行以解决numpy scalar问题
    slice,
    set,
])
_torch_load_orig = torch.load


def torch_load_compat(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    else:
        kwargs.update({"weights_only": False})
    return _torch_load_orig(*args, **kwargs)


torch.load = torch_load_compat

# ===============================================================
# 1. 导入必要的工具
# ===============================================================
# NOTE: This module must be called from pipeline.py via evaluate_attack_with_config()
# No standalone execution is supported - config must be passed from pipeline

import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image
from utils.sender import ResultSender


# ===============================================================
# 2. 复用训练脚本中的核心工具函数（确保特征处理逻辑一致）
# ===============================================================
def print_(x):
    """简化打印函数"""
    print(x)


def logscore(a, log_type=2):
    """得分对数转换（与训练脚本完全一致）"""
    # 确保输入值在有效范围内，避免除零错误和无效值
    a = np.clip(a, 0, 0.999999999)
    if log_type == 2:
        return -np.log2(1 - a + 1e-20)
    elif log_type > 0:
        # 添加额外的检查以避免无效值
        safe_a = np.clip(a, 0, 0.999999999)  # 确保值在安全范围内
        return -np.log(1 - safe_a + 1e-20)
    else:
        return a


def load_yolov7_model(model_path, cfg_path, num_classes, device):
    """加载YOLOv7模型（与训练脚本中加载影子模型的逻辑一致）"""
    raise NotImplementedError("YOLOv7 support has been removed. Use Faster R-CNN instead.")


def load_fasterrcnn_model(model_path, num_classes, device):
    """加载Faster R-CNN模型（与目标/影子模型加载逻辑一致）"""
    # 初始化模型，不使用预训练权重
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FasterRCNN

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


def generate_pointsets(model, img_paths, img_size, device, max_len=50, log_score_type=2,
                       num_logit_feature=1):
    """生成成员推理特征（仅提取特征，不标记标签，适配目标模型测试）"""
    model.eval()
    pointsets = []
    max_feat_len = 0
    min_feat_len = float('inf')

    # 如果img_paths是文件路径字符串，则读取其中的图像路径列表
    if isinstance(img_paths, str):
        with open(img_paths, 'r') as f:
            img_paths = [line.strip() for line in f.readlines()]

    with torch.no_grad():
        # 分批处理图像（直接处理，不使用Dataset）
        batch_size = 4
        for batch_start in tqdm(range(0, len(img_paths), batch_size), desc="生成特征点集"):
            batch_end = min(batch_start + batch_size, len(img_paths))
            batch_paths = img_paths[batch_start:batch_end]

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

                # 应用置信度阈值过滤（与攻击模型训练保持一致）
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

                # 检查最终结果是否有NaN或inf值，并替换为0
                padded_bboxes = np.nan_to_num(padded_bboxes, nan=0.0, posinf=0.0, neginf=0.0)
                pointsets.append(padded_bboxes)

    print(f"特征点集生成完成 - 最大长度: {max_feat_len}, 最小长度: {min_feat_len}, 样本数: {len(pointsets)}")
    return pointsets


# NOTE: For inference, we directly process image paths in batches
# No need for a custom Dataset class


def make_canvas_data(dataset, canvas_size=300, canvas_type="original", ball_size=30, normalize=True,
                     log_score_type=2, global_normalize=False, save_samples=0, save_dir=None):
    """转为画布特征（借鉴mia/evaluate_attack.py的实现）"""
    canvas_data = []
    sample_count = 0

    # 如果启用全局归一化，先计算所有画布的最大值
    global_max = 1.0
    if global_normalize:
        all_canvas_max = []
        # 第一次遍历：计算所有画布的最大值
        for idx, item in enumerate(dataset):
            if isinstance(item, tuple):
                feats, _ = item  # 如果是(特征,标签)元组，则只取特征
            else:
                feats = item  # 如果只是特征数组

            canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

            # 处理每个特征点
            for feat in feats:
                # 跳过零填充的特征
                if np.sum(feat) < 1e-5:
                    continue

                # 提取边界框坐标和分数
                x0, y0, x1, y1 = feat[:4]  # 归一化坐标 [0, 1]
                score = feat[4]

                # 应用logscore特征放大 - 与mia/evaluate_attack.py保持一致
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
                    # 确保坐标范围有效
                    if y1_canvas >= y0_canvas and x1_canvas >= x0_canvas:
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
        if isinstance(item, tuple):
            feats, label = item  # 如果是(特征,标签)元组，则只取特征
        else:
            feats = item  # 如果只是特征数组
            label = None

        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

        # 处理每个特征点
        for feat in feats:
            # 跳过零填充的特征
            if np.sum(feat) < 1e-5:
                continue

            # 提取边界框坐标和分数
            x0, y0, x1, y1 = feat[:4]  # 归一化坐标 [0, 1]
            score = feat[4]

            # 应用logscore特征放大 - 与mia/evaluate_attack.py保持一致
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
                # 确保坐标范围有效
                if y1_canvas >= y0_canvas and x1_canvas >= x0_canvas:
                    canvas[y0_canvas:y1_canvas + 1, x0_canvas:x1_canvas + 1] += score

        # 如果启用全局归一化，使用更合理的归一化方法
        if global_normalize and global_max > 0:
            # 使用固定值30作为上限，超过的值都显示为30
            canvas = np.clip(canvas, 0, 50)
        # 归一化处理 - 与atk.py保持一致，使用均值进行归一化
        elif normalize and np.sum(canvas) > 0:
            canvas = canvas / canvas.mean()

        # 检查是否有NaN或inf值，并替换为0
        canvas = np.nan_to_num(canvas, nan=0.0, posinf=0.0, neginf=0.0)

        # 输出一些画布统计信息用于调试
        if idx < 5:  # 只输出前5个样本的信息
            print(
                f"画布 {idx} - 最小值: {np.min(canvas):.6f}, 最大值: {np.max(canvas):.6f}, 均值: {np.mean(canvas):.6f}")

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

        canvas_data.append(canvas)

    return canvas_data


# ===============================================================
# 3. 攻击模型定义与加载（与训练脚本一致，确保模型结构兼容）
# ===============================================================
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


# ===============================================================
# 4. 目标模型攻击测试核心逻辑
# ===============================================================
class TargetTestDataset(Dataset):
    """目标模型测试数据集（仅含特征，真实标签外部传入）"""

    def __init__(self, canvas_data):
        self.canvas_data = canvas_data

    def __len__(self):
        return len(self.canvas_data)

    def __getitem__(self, idx):
        canvas = self.canvas_data[idx]
        # 转为3通道+PyTorch格式（与训练脚本一致）
        # 确保canvas没有NaN值
        canvas = np.nan_to_num(canvas, nan=0.0, posinf=0.0, neginf=0.0)
        # 再次检查确保canvas中没有NaN值
        if np.isnan(canvas).any() or np.isinf(canvas).any():
            canvas = np.zeros_like(canvas, dtype=np.float32)
        img = np.tile(canvas[..., np.newaxis], (1, 1, 3))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img


def calculate_attack_metrics(true_labels, pred_labels, pred_scores=None):
    """计算目标模型攻击的关键指标（借鉴mia/evaluate_attack.py，增加AUC等指标）"""
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
    # 攻击准确率（整体判断正确比例）
    accuracy = accuracy_score(true_labels, pred_labels)
    # 成员召回率（正确识别目标模型训练样本的比例，核心指标）
    recall_in = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    # 成员精确率（预测为成员的样本中真实成员的比例）
    precision_in = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    # F1分数（综合召回率与精确率）
    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)

    # 计算TPR和FPR（攻击的主要指标）
    # TPR: 正确识别训练集中样本的比例
    # FPR: 错误地将测试集中样本识别为训练集中样本的比例
    tpr = np.sum((pred_labels == 1) & (true_labels == 1)) / np.sum(true_labels == 1)
    fpr = np.sum((pred_labels == 1) & (true_labels == 0)) / np.sum(true_labels == 0)

    # 计算AUC（如果提供了预测分数）
    auc = None
    if pred_scores is not None:
        try:
            auc = roc_auc_score(true_labels, pred_scores)
        except:
            auc = None

    return {
        'accuracy': accuracy,
        'precision': precision_in,  # 与mia/evaluate_attack.py保持一致的命名
        'recall': recall_in,  # 与mia/evaluate_attack.py保持一致的命名
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'auc': auc
    }


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
    import matplotlib
    # Use non-interactive backend
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

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


def plot_roc_curve(true_labels, pred_scores, save_path):
    """绘制ROC曲线并保存（借鉴mia/evaluate_attack.py）"""
    import matplotlib
    # Use non-interactive backend
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Configure matplotlib fonts
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

    from sklearn.metrics import roc_curve, roc_auc_score

    try:
        fpr_curve, tpr_curve, _ = roc_curve(true_labels, pred_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_curve, tpr_curve, label=f'AUC = {roc_auc_score(true_labels, pred_scores):.4f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve of Attack Model')
        plt.legend()
        plt.grid(True)

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存ROC曲线（提高质量）
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        return True
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")
        return False


def attack_target_model(config, target_member_imgs, target_nonmember_imgs, target_model=None):
    """
    Attack target model using the simplified MIA flow.

    Evaluation setup:
    - Member samples: TRAIN set (target model's training data)
    - Non-member samples: TEST set (target model never saw)

    Args:
        config: Configuration object with all required parameters
        target_member_imgs: List of image paths for member samples (TRAIN set)
        target_nonmember_imgs: List of image paths for non-member samples (TEST set)
    """
    # 1. 初始化设备（与训练脚本一致）
    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print_(f"使用设备：{device}")

    # 2. 加载目标模型（核心：攻击的对象）
    print_("\n=== 1/4：加载目标模型 ===")
    if target_model is None:
        if not os.path.exists(config.TARGET_MODEL_DIR):
            raise FileNotFoundError(f"目标模型权重文件不存在：{config.TARGET_MODEL_DIR}")
        target_model = load_fasterrcnn_model(
            model_path=config.TARGET_MODEL_DIR,
            num_classes=config.num_classes,
            device=device
        )
        print_(f"✅ 目标模型加载完成：{config.TARGET_MODEL_DIR}")
    else:
        target_model = target_model.to(device)
        print_("✅ 使用外部传入的目标模型实例")

    target_model.eval()

    # 3. 使用 Pipeline 提供的测试样本
    print_("\n=== 2/4：使用 Pipeline 提供的测试样本 ===")
    print_(f"✅ 目标测试样本准备完成：")
    print_(f"   - 成员样本数（TRAIN集）：{len(target_member_imgs)}")
    print_(f"   - 非成员样本数（TEST集）：{len(target_nonmember_imgs)}")

    # 合并样本与真实标签（1=成员，0=非成员）
    target_test_imgs = target_member_imgs + target_nonmember_imgs
    target_true_labels = [1] * len(target_member_imgs) + [0] * len(target_nonmember_imgs)

    # 4. 提取目标模型的预测特征并转为画布特征
    print_("\n=== 3/4：提取目标模型特征 ===")
    # 生成特征（用目标模型预测）
    target_points = generate_pointsets(
        target_model,
        target_test_imgs,
        config.img_size,
        device,
        max_len=config.MAX_LEN,
        log_score_type=config.LOG_SCORE
    )

    # 转为画布特征（与训练脚本格式一致）
    target_canvas = make_canvas_data(
        target_points,
        canvas_size=config.input_size,
        canvas_type=config.CANVAS_TYPE,
        normalize=config.NORMALIZE_CANVAS,
        log_score_type=config.LOG_SCORE,
        global_normalize=True
    )
    print_(f"✅ 目标模型特征提取完成：共{len(target_canvas)}个样本特征")

    # 5. 加载训练好的攻击模型并预测
    print_("\n=== 4/4：攻击模型预测与结果评估 ===")
    # 加载攻击模型权重
    attack_model_path = config.ATTACK_MODEL_DIR
    if not os.path.exists(attack_model_path):
        # 如果指定的模型路径不存在，尝试查找last.pth
        last_model_path = os.path.join(os.path.dirname(attack_model_path), 'last.pth')
        if os.path.exists(last_model_path):
            attack_model_path = last_model_path
        else:
            raise FileNotFoundError(f"攻击模型权重文件不存在：{config.ATTACK_MODEL_DIR}")

    attack_model = AttackModel(model_type=config.ATTACK_MODEL)

    # 加载模型权重
    # 使用兼容的torch.load函数
    checkpoint = torch.load(attack_model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        attack_model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        attack_model.load_state_dict(checkpoint['state_dict'])
    else:
        attack_model.load_state_dict(checkpoint)

    attack_model = attack_model.to(device).eval()
    print_(f"✅ 攻击模型加载完成：{attack_model_path}")

    # 批量预测（避免GPU内存不足）
    target_dataset = TargetTestDataset(target_canvas)
    target_loader = DataLoader(
        target_dataset,
        batch_size=config.batch_size,  # 复用训练时的batch_size
        shuffle=False,
        num_workers=config.attack_workers,
        pin_memory=True
    )

    # 攻击模型预测
    target_pred_labels = []
    target_pred_scores = []  # 添加预测分数列表
    with torch.no_grad():
        for batch_idx, imgs in enumerate(target_loader):
            imgs = imgs.to(device)
            # 检查输入数据中是否有NaN或Inf
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)

            outputs = attack_model(imgs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取成员类别的概率

            target_pred_labels.extend(preds.cpu().numpy())
            target_pred_scores.extend(probs.cpu().numpy())

    # 转换为numpy数组便于计算
    target_pred_labels = np.array(target_pred_labels)
    target_pred_scores = np.array(target_pred_scores)
    target_true_labels = np.array(target_true_labels)

    # 计算攻击性能指标 - 增加预测分数参数
    attack_metrics = calculate_attack_metrics(target_true_labels, target_pred_labels, target_pred_scores)
    ResultSender.send_log("进度", "攻击模型预测完成，正在汇总性能指标")
    result_args = [
        "Accuracy",
        f"{attack_metrics['accuracy']:.4f}",
        "Precision",
        f"{attack_metrics['precision']:.4f}",
        "Recall",
        f"{attack_metrics['recall']:.4f}",
        "F1",
        f"{attack_metrics['f1']:.4f}",
        "TPR",
        f"{attack_metrics['tpr']:.4f}",
        "FPR",
        f"{attack_metrics['fpr']:.4f}",
    ]
    if attack_metrics['auc'] is not None:
        result_args.extend(["AUC", f"{attack_metrics['auc']:.4f}"])

    # 绘制ROC曲线
    if config.save_results and attack_metrics['auc'] is not None:
        # 创建保存目录
        results_dir = os.path.dirname(config.results_file) if os.path.dirname(config.results_file) else '.'
        os.makedirs(results_dir, exist_ok=True)

        # 绘制并保存ROC曲线
        roc_path = os.path.join(results_dir, 'attack_roc_curve.png')
        if plot_roc_curve(target_true_labels, target_pred_scores, roc_path):
            result_args.extend(["ROC曲线", roc_path])
            ResultSender.send_log("进度", f"ROC曲线已保存到: {roc_path}")

    ResultSender.send_result(*result_args)

    # 保存预测结果和评估指标
    if config.save_results:
        # 仅保留最终指标的输出，移除文件保存部分
        ResultSender.send_log("进度", "评估完成，结果已输出")


def evaluate_attack_with_config(
    pipeline_config,
    target_member_imgs,
    target_nonmember_imgs,
    *,
    target_model=None,
):
    """
    Evaluate attack model with configuration from pipeline.

    Args:
        pipeline_config: PipelineConfig object from pipeline.py
        target_member_imgs: List of image paths for member samples (TRAIN set)
        target_nonmember_imgs: List of image paths for non-member samples (TEST set)
        target_model: Optional pre-loaded target model instance. When provided,
            the attack uses this model directly instead of reloading from disk.
    """

    # Create a config-like object
    class ConfigAdapter:
        pass

    cfg = ConfigAdapter()

    # Map pipeline config to expected attributes
    cfg.gpu_id = pipeline_config.gpu_id
    cfg.img_size = pipeline_config.img_size
    cfg.num_classes = pipeline_config.num_classes
    cfg.save_results = True

    # Model paths
    cfg.TARGET_MODEL_DIR = pipeline_config.TARGET_MODEL_DIR
    cfg.ATTACK_MODEL_DIR = pipeline_config.ATTACK_MODEL_DIR

    # Canvas/Feature settings
    cfg.input_size = pipeline_config.CANVAS_SIZE
    cfg.MAX_LEN = pipeline_config.MAX_LEN
    cfg.LOG_SCORE = pipeline_config.LOG_SCORE
    cfg.CANVAS_TYPE = pipeline_config.CANVAS_TYPE
    cfg.NORMALIZE_CANVAS = pipeline_config.NORMALIZE_CANVAS
    cfg.ATTACK_MODEL = pipeline_config.ATTACK_MODEL_TYPE

    # Output settings
    cfg.results_file = pipeline_config.RESULTS_FILE
    cfg.batch_size = pipeline_config.ATTACK_BATCH_SIZE
    cfg.attack_workers = 0

    # Call attack_target_model with config and image paths
    attack_target_model(
        cfg,
        target_member_imgs,
        target_nonmember_imgs,
        target_model=target_model,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("ERROR: This script cannot be run directly!")
    print("=" * 70)
    print("\nThis script must be called from pipeline.py")
    print("\nUsage:")
    print("  python pipeline.py --steps 4")
    print("  python pipeline.py  # Run complete pipeline")
    print("\n" + "=" * 70)
    sys.exit(1)