"""
YOLO 标签与数据批次转换工具。

这些函数最初定义在单独的测试脚本中，此处抽取为公用模块，方便
不同评测/训练脚本共享相同的转换逻辑并集中维护。
"""

from typing import Dict, List, Optional

import numpy as np
import torch


def yolo_labels_to_dict(
    labels: Optional[torch.Tensor],
    img_size: int = 640,
) -> Dict[str, np.ndarray]:
    """
    将 YOLO 输出或标签转换为评测函数需要的字典格式（像素坐标）。

    支持三种格式：
        1. 真实标签: [batch_id, class_id, x_center, y_center, width, height]（归一化）
        2. 预测输出: [x1, y1, x2, y2, confidence, class_id]（像素或归一化坐标）
        3. 简化标签: [class_id, x_center, y_center, width, height]（归一化）
    """
    empty_result = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "scores": np.ones((0,), dtype=np.float32),
    }

    if labels is None or labels.numel() == 0:
        return empty_result

    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    if labels_np.size == 0 or labels_np.shape[0] == 0:
        return empty_result

    if len(labels_np.shape) == 1:
        labels_np = labels_np.reshape(1, -1)

    if labels_np.shape[1] == 6:
        first_val = labels_np[0, 0]
        second_val = labels_np[0, 1]
        third_val = labels_np[0, 2]
        fourth_val = labels_np[0, 3]

        is_ground_truth_format = (
            abs(first_val) < 10
            and first_val == int(first_val)
            and second_val >= 0
            and second_val == int(second_val)
            and 0 <= third_val <= 1.0
            and 0 <= fourth_val <= 1.0
        )

        if is_ground_truth_format:
            class_ids = labels_np[:, 1].astype(np.int64)
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
            x1 = labels_np[:, 0]
            y1 = labels_np[:, 1]
            x2 = labels_np[:, 2]
            y2 = labels_np[:, 3]
            scores = labels_np[:, 4].astype(np.float32)
            class_ids = labels_np[:, 5].astype(np.int64)

            if (
                x1.max() <= 1.0
                and y1.max() <= 1.0
                and x2.max() <= 1.0
                and y2.max() <= 1.0
            ):
                x1 = x1 * img_size
                y1 = y1 * img_size
                x2 = x2 * img_size
                y2 = y2 * img_size
    elif labels_np.shape[1] == 5:
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
        "scores": scores,
    }


def convert_yolo_loader_to_dict_format(loader):
    """
    将 YOLO 数据加载器的输出转换为评测流程需要的 (images, targets_list) 形式。

    loader: 迭代输出 (images, labels, *extras) 或 (images, labels_list) 的迭代器。
    """
    img_size = 640

    for batch in loader:
        if len(batch) >= 2 and isinstance(batch[0], torch.Tensor):
            images = batch[0]
            labels_data = batch[1]

            if len(images.shape) == 4:
                img_size = images.shape[-1]

            targets_list: List[Dict[str, np.ndarray]] = []

            if (
                isinstance(labels_data, torch.Tensor)
                and labels_data.dim() == 2
                and labels_data.shape[1] == 6
            ):
                all_labels = labels_data
                for i in range(images.shape[0]):
                    image_labels = all_labels[all_labels[:, 0] == i]
                    target_dict = yolo_labels_to_dict(image_labels, img_size=img_size)
                    targets_list.append(target_dict)
            elif isinstance(labels_data, list):
                for labels in labels_data:
                    target_dict = yolo_labels_to_dict(labels, img_size=img_size)
                    targets_list.append(target_dict)
            else:
                yield batch
                continue

            yield images, targets_list
        else:
            yield batch


__all__ = ["yolo_labels_to_dict", "convert_yolo_loader_to_dict_format"]
