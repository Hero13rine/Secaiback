"""
YOLO 数据集加载函数
专门用于加载 YOLO 格式的数据集（如 DIOR）
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image

# 尝试导入 YOLOv7 的数据集工具
try:
    # YOLOv7 的 datasets 模块需要从 YOLOv7 代码路径导入
    import importlib.util
    
    # 尝试从默认路径加载
    yolov7_root = "/wkm/secai-common/yolo/yolov7"
    if os.path.exists(yolov7_root):
        if yolov7_root not in sys.path:
            sys.path.insert(0, yolov7_root)
        
        # 加载 YOLOv7 的 datasets 模块
        try:
            from utils.datasets import LoadImagesAndLabels, letterbox
            YOLOV7_DATASETS_AVAILABLE = True
        except ImportError:
            YOLOV7_DATASETS_AVAILABLE = False
            print("警告: YOLOv7 datasets 模块不可用，将使用简化版本")
    else:
        YOLOV7_DATASETS_AVAILABLE = False
except Exception:
    YOLOV7_DATASETS_AVAILABLE = False


class DIORDataset(Dataset):
    """
    DIOR 数据集的 Dataset 类
    用于加载 YOLO 格式的 DIOR 数据集
    """
    def __init__(
        self,
        data_yaml_path: str,
        img_size: int = 640,
        augment: bool = False,
        hyp: Optional[dict] = None,
        rect: bool = False,
        cache_images: bool = False,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        image_weights: bool = False,
        prefix: str = ''
    ):
        """
        初始化 DIOR 数据集
        
        Args:
            data_yaml_path: DIOR 数据集 YAML 配置文件路径
            img_size: 图像尺寸，默认 640
            augment: 是否使用数据增强，默认 False
            hyp: 超参数字典，用于数据增强
            rect: 是否使用矩形训练，默认 False
            cache_images: 是否缓存图像到内存，默认 False
            single_cls: 是否单类别模式，默认 False
            stride: 模型步长，默认 32
            pad: 填充值，默认 0.0
            image_weights: 是否使用图像权重，默认 False
            prefix: 前缀字符串，用于日志输出
        """
        self.data_yaml_path = data_yaml_path
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        self.rect = rect
        self.cache_images = cache_images
        self.single_cls = single_cls
        self.stride = stride
        self.pad = pad
        self.image_weights = image_weights
        self.prefix = prefix
        
        # 解析 YAML 配置文件
        import yaml
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_dict = yaml.safe_load(f)
        
        self.path = data_dict.get('path', '')  # 数据集根目录
        self.names = data_dict.get('names', {})  # 类别名称字典
        self.nc = len(self.names)  # 类别数量
        
        # 获取数据集路径
        train_path = data_dict.get('train', '')
        val_path = data_dict.get('val', '')
        test_path = data_dict.get('test', '')
        
        # 如果路径是相对路径，则相对于 YAML 文件所在目录
        if not os.path.isabs(train_path):
            train_path = os.path.join(os.path.dirname(data_yaml_path), train_path)
        if not os.path.isabs(val_path):
            val_path = os.path.join(os.path.dirname(data_yaml_path), val_path)
        if not os.path.isabs(test_path):
            test_path = os.path.join(os.path.dirname(data_yaml_path), test_path)
        
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        # 使用 YOLOv7 的 LoadImagesAndLabels 或简化版本
        if YOLOV7_DATASETS_AVAILABLE:
            # 使用 YOLOv7 的官方数据集加载器
            self.dataset = LoadImagesAndLabels(
                path=train_path,
                img_size=img_size,
                batch_size=1,
                augment=augment,
                hyp=hyp,
                rect=rect,
                cache_images=cache_images,
                single_cls=single_cls,
                stride=stride,
                pad=pad,
                image_weights=image_weights,
                prefix=prefix
            )
        else:
            # 使用简化版本
            self.dataset = self._load_simple_dataset(train_path)
    
    def _load_simple_dataset(self, path: str):
        """加载简化版本的数据集"""
        # 读取图像和标签文件
        image_files = []
        label_files = []
        
        if os.path.isdir(path):
            # 查找所有图像文件
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(Path(path).glob(f'*{ext}'))
                image_files.extend(Path(path).glob(f'*{ext.upper()}'))
            
            # 查找对应的标签文件（在 labels 目录中）
            labels_dir = Path(path).parent / 'labels'
            if labels_dir.exists():
                for img_file in image_files:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        label_files.append(label_file)
                    else:
                        label_files.append(None)
        
        return list(zip(image_files, label_files))
    
    def __len__(self):
        if YOLOV7_DATASETS_AVAILABLE:
            return len(self.dataset)
        else:
            return len(self.dataset)
    
    def __getitem__(self, idx):
        if YOLOV7_DATASETS_AVAILABLE:
            # 使用 YOLOv7 的加载方式
            return self.dataset[idx]
        else:
            # 简化版本
            img_file, label_file = self.dataset[idx]
            
            # 加载图像
            img = cv2.imread(str(img_file))
            if img is None:
                raise ValueError(f"无法加载图像: {img_file}")
            
            # 调整图像大小
            img = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            
            # 转换为 tensor
            img = torch.from_numpy(img).float()
            img /= 255.0  # 归一化到 [0, 1]
            
            # 加载标签
            if label_file is not None:
                labels = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([cls, x_center, y_center, width, height])
                
                labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
            else:
                labels = torch.zeros((0, 5))
            
            return img, labels, img_file


def load_dior(
    data_yaml_path: str = "/wkm/secai-common/yolo/yolov7/data/dior.yaml",
    img_size: int = 640,
    batch_size: int = 16,
    split: str = "val",
    augment: bool = False,
    yolov7_root: Optional[str] = None
) -> DataLoader:
    """
    加载 DIOR 数据集（YOLO 格式）
    
    Args:
        data_yaml_path: DIOR 数据集 YAML 配置文件路径
        img_size: 图像尺寸，默认 640
        batch_size: 批次大小，默认 16
        split: 数据集分割，可选 'train', 'val', 'test'，默认 'val'
        augment: 是否使用数据增强（仅对 train 有效），默认 False
        yolov7_root: YOLOv7 代码根目录（可选，会自动检测）
    
    Returns:
        DataLoader 对象
    
    Examples:
        >>> # 加载验证集
        >>> val_loader = load_dior(split='val', batch_size=16)
        
        >>> # 加载训练集（带数据增强）
        >>> train_loader = load_dior(split='train', batch_size=16, augment=True)
        
        >>> # 加载测试集
        >>> test_loader = load_dior(split='test', batch_size=8)
    """
    # 自动检测 YOLOv7 代码路径
    if yolov7_root is None:
        possible_paths = [
            "/wkm/secai-common/yolo/yolov7",
            os.path.join(os.path.dirname(data_yaml_path), "..", ".."),
        ]
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, "models")):
                yolov7_root = abs_path
                break
    
    # 如果找到 YOLOv7 代码路径，添加到 sys.path
    if yolov7_root and os.path.exists(yolov7_root):
        if yolov7_root not in sys.path:
            sys.path.insert(0, yolov7_root)
        
        # 尝试加载 YOLOv7 的 datasets 模块
        try:
            from utils.datasets import LoadImagesAndLabels, letterbox
            global YOLOV7_DATASETS_AVAILABLE
            YOLOV7_DATASETS_AVAILABLE = True
        except ImportError:
            YOLOV7_DATASETS_AVAILABLE = False
            print("警告: YOLOv7 datasets 模块不可用，将使用简化版本")
    
    # 检查 YAML 文件是否存在
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"DIOR 数据集配置文件不存在: {data_yaml_path}")
    
    # 解析 YAML 配置文件
    import yaml
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    # 获取数据集路径
    if split == "train":
        dataset_path = data_dict.get('train', '')
    elif split == "val":
        dataset_path = data_dict.get('val', '')
    elif split == "test":
        dataset_path = data_dict.get('test', '')
    else:
        raise ValueError(f"不支持的 split: {split}（支持: 'train', 'val', 'test'）")
    
    if not dataset_path:
        raise ValueError(f"YAML 配置文件中未找到 {split} 路径")
    
    # 如果路径是相对路径，则相对于 YAML 文件所在目录
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.path.dirname(data_yaml_path), dataset_path)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    
    # 创建数据集
    if YOLOV7_DATASETS_AVAILABLE:
        # 使用 YOLOv7 的官方数据集加载器
        dataset = LoadImagesAndLabels(
            path=dataset_path,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment and split == "train",
            hyp=None,
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            image_weights=False,
            prefix=f"DIOR {split}"
        )
        
        # YOLOv7 的 LoadImagesAndLabels 返回的是特殊格式，需要处理
        # 这里我们创建一个简单的包装器
        class YOLODatasetWrapper(Dataset):
            def __init__(self, yolo_dataset):
                self.yolo_dataset = yolo_dataset
            
            def __len__(self):
                return len(self.yolo_dataset)
            
            def __getitem__(self, idx):
                # YOLOv7 的 LoadImagesAndLabels 返回格式: (img, labels, img_path, shapes)
                # labels 格式: [class_id, x1, y1, x2, y2] (像素坐标) 或 [class_id, x_center, y_center, width, height] (归一化)
                result = self.yolo_dataset[idx]
                if isinstance(result, tuple) and len(result) >= 2:
                    return result[0], result[1]  # 返回 (img, labels)
                return result
        
        dataset = YOLODatasetWrapper(dataset)
    else:
        # 使用简化版本
        dataset = DIORDataset(
            data_yaml_path=data_yaml_path,
            img_size=img_size,
            augment=augment and split == "train",
            prefix=f"DIOR {split}"
        )
    
    # 自定义 collate_fn 用于处理 YOLOv7 数据格式
    def yolo_collate_fn(batch):
        """自定义 collate 函数，用于处理 YOLOv7 数据格式"""
        # YOLOv7 的数据格式可能是 (img, labels, img_path, shapes) 或 (img, labels)
        # 检查第一个样本的格式
        first_item = batch[0]
        
        if isinstance(first_item, (list, tuple)):
            # 提取图像和标签
            images = []
            labels_list = []
            img_paths = []
            shapes = []
            
            for i, item in enumerate(batch):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    images.append(item[0])
                    labels = item[1]
                    
                    # 处理标签格式：YOLOv7 的 LoadImagesAndLabels 返回的标签格式可能是：
                    # 1. [class_id, x1, y1, x2, y2] (像素坐标) - 非 augment 模式
                    # 2. [class_id, x_center, y_center, width, height] (归一化) - 如果未转换
                    # 在 collate_fn 中，需要添加 batch_id 到第一列
                    if isinstance(labels, torch.Tensor) and labels.numel() > 0:
                        # 如果标签的第一列不是 batch_id（即 >= 0 且是整数），则添加 batch_id
                        if labels.shape[0] > 0:
                            first_col = labels[0, 0].item()
                            # 如果第一列不是 batch_id（通常是 0, 1, 2...），则添加 batch_id
                            if first_col != i or labels.shape[1] == 5:
                                # 创建新标签，第一列是 batch_id
                                if labels.shape[1] == 5:
                                    # 格式：[class_id, x1, y1, x2, y2] 或 [class_id, x_center, y_center, width, height]
                                    # 添加 batch_id 到第一列
                                    batch_id_col = torch.full((labels.shape[0], 1), i, dtype=labels.dtype, device=labels.device)
                                    labels = torch.cat([batch_id_col, labels], dim=1)
                                elif labels.shape[1] == 6:
                                    # 格式：[batch_id, class_id, ...]，确保 batch_id 正确
                                    labels = labels.clone()
                                    labels[:, 0] = i
                    
                    labels_list.append(labels)
                    if len(item) >= 3:
                        img_paths.append(item[2])
                    if len(item) >= 4:
                        shapes.append(item[3])
                else:
                    # 如果格式不对，尝试提取第一个元素作为图像
                    images.append(item[0] if isinstance(item, (list, tuple)) else item)
                    labels_list.append(None)
            
            # 堆叠图像（所有图像大小相同）
            try:
                images = torch.stack(images, 0)
            except Exception as e:
                # 如果堆叠失败，尝试确保所有图像大小相同
                # 获取最大尺寸
                max_h = max(img.shape[-2] for img in images)
                max_w = max(img.shape[-1] for img in images)
                # 调整所有图像到相同尺寸
                from torch.nn.functional import pad
                padded_images = []
                for img in images:
                    if img.shape[-2] < max_h or img.shape[-1] < max_w:
                        pad_h = max_h - img.shape[-2]
                        pad_w = max_w - img.shape[-1]
                        img = pad(img, (0, pad_w, 0, pad_h))
                    padded_images.append(img)
                images = torch.stack(padded_images, 0)
            
            # 标签列表（每个图像的标签数量不同，不能堆叠，保持为列表）
            # 返回元组格式：(images, labels_list, img_paths, shapes)
            if len(first_item) >= 3 and img_paths:
                return images, labels_list, img_paths, shapes if shapes else None
            else:
                return images, labels_list
        else:
            # 如果只是单个 tensor，使用默认的 collate
            try:
                return torch.utils.data.dataloader.default_collate(batch)
            except Exception:
                # 如果默认 collate 失败，尝试手动处理
                if isinstance(batch[0], torch.Tensor):
                    return torch.stack(batch, 0)
                else:
                    return batch
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        collate_fn=yolo_collate_fn  # 使用自定义 collate 函数
    )
    
    return dataloader


def load_dior_train_val(
    data_yaml_path: str = "/wkm/secai-common/yolo/yolov7/data/dior.yaml",
    img_size: int = 640,
    batch_size: int = 16,
    augment: bool = True,
    yolov7_root: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    同时加载 DIOR 数据集的训练集和验证集
    
    Args:
        data_yaml_path: DIOR 数据集 YAML 配置文件路径
        img_size: 图像尺寸，默认 640
        batch_size: 批次大小，默认 16
        augment: 是否对训练集使用数据增强，默认 True
        yolov7_root: YOLOv7 代码根目录（可选，会自动检测）
    
    Returns:
        (train_loader, val_loader) 元组
    """
    train_loader = load_dior(
        data_yaml_path=data_yaml_path,
        img_size=img_size,
        batch_size=batch_size,
        split="train",
        augment=augment,
        yolov7_root=yolov7_root
    )
    
    val_loader = load_dior(
        data_yaml_path=data_yaml_path,
        img_size=img_size,
        batch_size=batch_size,
        split="val",
        augment=False,
        yolov7_root=yolov7_root
    )
    
    return train_loader, val_loader

