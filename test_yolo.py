"""
测试yolo的评估流程 - 完整修正版本

修正重点：
1. 确保模型进入评估模式 (model.eval())。
2. 修复 YOLO 真实标签 (N x 6) 到评估工具字典格式的转换逻辑，特别处理 batch_id 分割。
3. 确保模型和数据在正确的设备上 (GPU/CPU)。
"""

import numpy as np
import torch
import os
import sys
from typing import List, Dict, Any, Optional

# --- 假设导入的依赖模块（需要确保这些模块在您的环境中存在） ---
# 请根据您的实际文件结构确保以下导入路径正确：
from estimator import EstimatorFactory
from method.load_config import load_config
from metric.object_detection.basic.detection import cal_object_detection
from model import load_yolo_model
from data.load_yolo_dataset import load_dior # 您的 DIOR 数据加载器

# 尝试导入 YOLOv7 的工具函数 (可选，用于高级调试)
try:
    # 假设 load_yolo_model / load_dior 已经将 YOLOv7 根目录添加到 sys.path
    from utils.general import xywhn2xyxy 
    YOLOV7_UTILS_AVAILABLE = True
except ImportError:
    YOLOV7_UTILS_AVAILABLE = False
# -------------------------------------------------------------


def yolo_labels_to_dict(
    labels: Optional[torch.Tensor], 
    img_size: int = 640
) -> Dict[str, np.ndarray]:
    """
    将 YOLO 格式的标签/预测结果转换为评测函数期望的字典格式（像素坐标）。
    
    支持格式：
    1. 真实标签 (GT): [batch_id, class_id, x_center, y_center, width, height] (归一化)
    2. 预测输出 (Pred): [x1, y1, x2, y2, confidence, class_id] (像素坐标)
    3. 简化标签: [class_id, x_center, y_center, width, height] (归一化)
    
    Args:
        labels: YOLO 格式标签 tensor (N, 6), (N, 5) 或 None。
        img_size: 图像尺寸，用于将归一化坐标转换为像素坐标（假设是 letterbox 后的尺寸）。
        
    Returns:
        字典格式: {"boxes": [x1, y1, x2, y2], "labels": [class_id], "scores": [score]}
    """
    empty_result = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "scores": np.ones((0,), dtype=np.float32)
    }
    
    if labels is None or labels.numel() == 0:
        return empty_result
    
    # 转换为 numpy
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)
        
    if labels_np.size == 0 or labels_np.shape[0] == 0:
        return empty_result
    
    # 确保是二维数组
    if len(labels_np.shape) == 1:
        labels_np = labels_np.reshape(1, -1)

    # --- 核心转换逻辑 ---
    
    if labels_np.shape[1] == 6:
        # 6个值的情况，根据数据来源判断格式：
        # 1. 真实标签（来自 YOLOv7 LoadImagesAndLabels）：[batch_id, class_id, x_center, y_center, width, height]（归一化）
        # 2. 预测输出（来自模型）：[x1, y1, x2, y2, confidence, class_id]（像素坐标）
        
        # 判断：如果第一个值是整数且较小（batch_id），第二个值也是整数且较大（class_id），
        # 且第三、四个值在 [0, 1] 范围内，则认为是真实标签格式
        first_val = labels_np[0, 0]
        second_val = labels_np[0, 1]
        third_val = labels_np[0, 2]
        fourth_val = labels_np[0, 3]
        
        # 真实标签格式判断：第一个值是 batch_id（0, 1, 2...），第二个值是 class_id（整数），
        # 第三、四个值是归一化坐标（在 [0, 1] 范围内）
        is_ground_truth_format = (
            abs(first_val) < 10 and first_val == int(first_val) and  # batch_id 是小的整数
            second_val >= 0 and second_val == int(second_val) and  # class_id 是整数
            third_val >= 0 and third_val <= 1.0 and  # x_center 归一化
            fourth_val >= 0 and fourth_val <= 1.0  # y_center 归一化
        )
        
        if is_ground_truth_format:
            # 格式：[batch_id, class_id, x_center, y_center, width, height]（归一化）
            # 这是 YOLOv7 LoadImagesAndLabels 返回的真实标签格式
            # class_id 应该是 0-based 索引（0, 1, 2, ..., nc-1），对于 20 个类别应该是 0-19
            class_ids = labels_np[:, 1].astype(np.int64)
            # 确保类别索引在有效范围内（0-19 对于 20 个类别）
            # 如果类别索引超出范围，可能是数据格式问题
            if class_ids.size > 0 and (class_ids.max() >= 20 or class_ids.min() < 0):
                print(f"警告: 类别索引超出范围 [0, 19]: min={class_ids.min()}, max={class_ids.max()}")
            x_center = labels_np[:, 2] * img_size
            y_center = labels_np[:, 3] * img_size
            width = labels_np[:, 4] * img_size
            height = labels_np[:, 5] * img_size
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            scores = np.ones((labels_np.shape[0],), dtype=np.float32)
        else:
            # 格式：[x1, y1, x2, y2, confidence, class_id]（预测输出，像素坐标）
            # 注意：这里直接使用置信度作为最终分数，不再进行 objectness * class_conf 的计算
            # 因为模型输出时已经处理过 objectness 很低的情况，直接使用类别分数作为置信度
            x1 = labels_np[:, 0]
            y1 = labels_np[:, 1]
            x2 = labels_np[:, 2]
            y2 = labels_np[:, 3]
            # 直接使用置信度作为最终分数（已经是处理过的类别分数或 objectness * class_conf）
            scores = labels_np[:, 4].astype(np.float32)
            class_ids = labels_np[:, 5].astype(np.int64)
            
            # 如果坐标是归一化的（在[0,1]范围内），转换为像素坐标
            if x1.max() <= 1.0 and y1.max() <= 1.0 and x2.max() <= 1.0 and y2.max() <= 1.0:
                x1 = x1 * img_size
                y1 = y1 * img_size
                x2 = x2 * img_size
                y2 = y2 * img_size
    
    elif labels_np.shape[1] == 5:
        # 格式 3: 传统 YOLO [class_id, x_center, y_center, width, height] (归一化)
        class_ids = labels_np[:, 0].astype(np.int64)
        yolo_coords = labels_np[:, 1:5]
        
        x_center = yolo_coords[:, 0] * img_size
        y_center = yolo_coords[:, 1] * img_size
        width = yolo_coords[:, 2] * img_size
        height = yolo_coords[:, 3] * img_size
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        scores = np.ones((labels_np.shape[0],), dtype=np.float32)

    else:
        return empty_result
    
    # 坐标裁剪和整理
    x1 = np.clip(x1, 0, img_size)
    y1 = np.clip(y1, 0, img_size)
    x2 = np.clip(x2, 0, img_size)
    y2 = np.clip(y2, 0, img_size)
    
    x1, x2 = np.minimum(x1, x2), np.maximum(x1, x2)
    y1, y2 = np.minimum(y1, y2), np.maximum(y1, y2)
    
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    
    return {
        "boxes": boxes,
        "labels": class_ids,
        "scores": scores
    }


def convert_yolo_loader_to_dict_format(loader):
    """
    将 YOLO 数据加载器转换为评测函数期望的格式
    
    Args:
        loader: YOLO 数据加载器，返回 (images, labels_list, ...) 或 (images, all_labels)
    
    Yields:
        (images, targets_list): images 是 tensor，targets_list 是字典列表
    """
    # 默认图片尺寸
    img_size = 640
    
    for batch in loader:
        if len(batch) >= 2 and isinstance(batch[0], torch.Tensor):
            images = batch[0]
            labels_data = batch[1]
            
            if len(images.shape) == 4:
                # 假设输入是 [batch, channels, height, width]，且是正方形
                img_size = images.shape[-1]
            
            targets_list: List[Dict[str, np.ndarray]] = []
            
            if isinstance(labels_data, torch.Tensor) and labels_data.dim() == 2 and labels_data.shape[1] == 6:
                # 场景 1: 官方 LoadImagesAndLabels 的 collate_fn 结果 (N x 6)
                all_labels = labels_data
                for i in range(images.shape[0]):
                    # 关键：按 batch_id (第 0 列) 分割标签
                    image_labels = all_labels[all_labels[:, 0] == i]
                    target_dict = yolo_labels_to_dict(image_labels, img_size=img_size)
                    targets_list.append(target_dict)
            
            elif isinstance(labels_data, list):
                # 场景 2: 自定义 collate_fn 的结果 (labels_list)
                for labels in labels_data:
                    # labels 已经是 N x 5 或 N x 6 的张量
                    target_dict = yolo_labels_to_dict(labels, img_size=img_size)
                    targets_list.append(target_dict)
            else:
                # 标签数据格式未知，跳过
                yield batch
                continue
                
            yield images, targets_list
        else:
            yield batch


def main():
    # 0、定义关键参数
    evaluation_path = "config/user/model_pytorch_det_tolo.yaml"
    
    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    print("进度: 配置文件已加载完毕")

    # 2.初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo_model(
        weight_path=model_instantiation_config["weight_path"],
        yolov7_root=model_instantiation_config.get("model_path"),
        device=device
    )
    # --- 关键修正：设置模型为评估模式并移动到设备 ---
    model.eval() 
    model.to(device)
    print("进度: 模型初始化完成")

    # 3.获取优化器和损失函数 (此处保持不变)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss = None

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config
    )
    # 注意：模型已经在 load_yolo_model 时移动到正确的设备上了
    print("进度: 估计器已生成")

    # 5.加载数据
    data_yaml_path = "/wkm/secai-common/yolo/yolov7/data/dior.yaml"
    if "data" in user_config and "data_yaml_path" in user_config["data"]:
        data_yaml_path = user_config["data"]["data_yaml_path"]
    
    yolo_loader = load_dior(
        data_yaml_path=data_yaml_path,
        split="val",
        batch_size=8, # 恢复 batch_size，但如果之前有 DataLoader 线程错误，请改为 1 或 0
        img_size=640,
        augment=False 
    )
    print("进度: 数据集已加载")

    # 6.根据传入的评测类型进行评测
    print("开始执行检测流程测试...")
    test_loader = convert_yolo_loader_to_dict_format(yolo_loader)
    
    # 执行评估
    cal_object_detection(estimator, test_loader, evaluation_config)
    print("检测流程测试完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()