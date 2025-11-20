"""
目标检测公平性评估 - 子群体性能对比测试法

评测方法：
1. 根据敏感属性（如地理区域）将测试数据划分为不同子群体
2. 分别计算每个子群体的性能指标（mAP）
3. 计算性能极差（最大mAP - 最小mAP）

性能极差越小，说明模型在不同子群体上的表现越公平

适用模型：Faster RCNN（或其他目标检测模型）
适用数据：DIOR数据集（YOLO格式）
"""

import numpy as np
import torch
import re
from typing import Dict, List, Any, Callable
from pathlib import Path
from collections import defaultdict

from utils.SecAISender import ResultSender
# from utils.sender import ConsoleResultSender as ResultSender # 本地调试时使用



# ============================================================================
# 1. 敏感属性提取
# ============================================================================

def extract_sensitive_attribute_from_path(image_path: str, method: str = "directory") -> str:
    """
    从图像路径提取敏感属性

    Args:
        image_path: 图像文件路径
        method: 提取方法
            - "directory": 从父目录名提取
            - "filename_prefix": 从文件名前缀提取

    Returns:
        敏感属性值（字符串）
    """
    path_obj = Path(image_path)

    if method == "directory":
        return path_obj.parent.name
    elif method == "filename_prefix":
        name_parts = path_obj.stem.split('_')
        if len(name_parts) > 0:
            return name_parts[0]
        return "unknown"
    else:
        raise ValueError(f"不支持的提取方法: {method}")


def create_sensitive_attribute_extractor(method: str = "path_directory") -> Callable[[str], str]:
    """
    创建敏感属性提取函数

    Args:
        method: 提取方法
            - "path_directory": 从路径目录名提取
            - "path_prefix": 从路径文件名前缀提取

    Returns:
        敏感属性提取函数
    """
    if method == "path_directory":
        return lambda path: extract_sensitive_attribute_from_path(path, "directory")
    elif method == "path_prefix":
        return lambda path: extract_sensitive_attribute_from_path(path, "filename_prefix")
    else:
        raise ValueError(f"不支持的提取方法: {method}")


# ============================================================================
# 2. YOLO格式转换
# ============================================================================

def convert_yolo_to_standard_format(
    yolo_labels: np.ndarray,
    image_shape: tuple,
    scores: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    将YOLO格式的标注转换为标准格式

    YOLO格式: [class_id, x_center, y_center, width, height] (归一化)
    标准格式: {"boxes": [[x1, y1, x2, y2], ...], "labels": [...]}

    Args:
        yolo_labels: YOLO格式标注 (N, 5)
        image_shape: 图像形状 (height, width)
        scores: 置信度得分（可选）

    Returns:
        标准格式字典
    """
    if len(yolo_labels) == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "scores": np.ones((0,), dtype=np.float32) if scores is None else scores
        }

    yolo_labels = np.array(yolo_labels)
    h, w = image_shape

    # 解析YOLO格式
    class_ids = yolo_labels[:, 0].astype(np.int64)
    x_center = yolo_labels[:, 1] * w
    y_center = yolo_labels[:, 2] * h
    width = yolo_labels[:, 3] * w
    height = yolo_labels[:, 4] * h

    # 转换为 [x1, y1, x2, y2]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)

    return {
        "boxes": boxes,
        "labels": class_ids,
        "scores": scores
    }


def _ensure_list(data):
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    return [data]


def _to_numpy_array(value, dtype, vec_length: int | None = None):
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    elif value is None:
        return None
    else:
        arr = np.array(value)

    if arr.size == 0:
        if vec_length is None:
            return np.zeros((0,), dtype=dtype)
        return np.zeros((0, vec_length), dtype=dtype)

    arr = arr.astype(dtype, copy=False)
    if vec_length is not None:
        arr = arr.reshape(-1, vec_length)
    else:
        arr = arr.reshape(-1)
    return arr


def _normalize_boxes(value):
    boxes = _to_numpy_array(value, np.float32, 4)
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    return boxes


def _normalize_labels(value, count: int):
    labels = _to_numpy_array(value, np.int64)
    if labels is None or labels.size == 0:
        return np.zeros((count,), dtype=np.int64)
    return labels.astype(np.int64, copy=False)


def _normalize_scores(value, count: int):
    scores = _to_numpy_array(value, np.float32)
    if scores is None or scores.size == 0:
        return np.ones((count,), dtype=np.float32)
    return scores.astype(np.float32, copy=False)


def _convert_prediction(prediction: Any) -> Dict[str, np.ndarray]:
    if not isinstance(prediction, dict):
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "scores": np.ones((0,), dtype=np.float32),
        }
    boxes = _normalize_boxes(prediction.get("boxes"))
    labels = _normalize_labels(prediction.get("labels"), boxes.shape[0])
    scores = _normalize_scores(prediction.get("scores"), boxes.shape[0])
    return {"boxes": boxes, "labels": labels, "scores": scores}


def _convert_target(target: Any) -> Dict[str, np.ndarray]:
    if not isinstance(target, dict):
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "scores": np.ones((0,), dtype=np.float32),
        }
    boxes = _normalize_boxes(target.get("boxes"))
    labels = _normalize_labels(target.get("labels"), boxes.shape[0])
    return {
        "boxes": boxes,
        "labels": labels,
        "scores": np.ones((boxes.shape[0],), dtype=np.float32),
    }


def _extract_image_path_from_target(target: Any, fallback_idx: int) -> str:
    if isinstance(target, dict):
        path = target.get("image_path")
        if path:
            return str(path)
    return f"sample_{fallback_idx:06d}.jpg"


def _extract_attr_from_target(target: Any) -> str | None:
    if not isinstance(target, dict):
        return None
    attr = target.get("sensitive_attr")
    if attr is None:
        return None
    if isinstance(attr, torch.Tensor):
        if attr.numel() == 1:
            return str(attr.item())
        return "_".join(str(x) for x in attr.detach().cpu().numpy().tolist())
    return str(attr)


def _normalize_attr_value(attr: Any) -> Any:
    """
    将类似 group_0 的标签规范为数字 0，其余保持原样。
    """
    if attr is None:
        return "unknown"
    attr_str = str(attr).strip()
    match = re.fullmatch(r"group_(\d+)", attr_str, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if attr_str.isdigit():
        return int(attr_str)
    return attr_str


# ============================================================================
# 3. 性能计算
# ============================================================================

def calculate_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        intersection = 0.0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def calculate_simple_precision(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    iou_threshold: float = 0.5
) -> float:
    """简化的精度计算（F1分数）"""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_dict, gt_dict in zip(predictions, ground_truths):
        pred_boxes = pred_dict.get("boxes", np.array([]))
        gt_boxes = gt_dict.get("boxes", np.array([]))

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        matched_gt = set()
        tp = 0
        fp = 0

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    else:
        return 0.0


def calculate_map_for_subgroup(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    iou_threshold: float = 0.5
) -> float:
    """计算子群体的性能指标"""
    try:
        return calculate_simple_precision(predictions, ground_truths, iou_threshold)
    except Exception as e:
        ResultSender.send_log("警告", f"计算性能指标时出错: {e}")
        return 0.0


# ============================================================================
# 4. 性能极差计算
# ============================================================================

def calculate_performance_gap(
    subgroup_performances: Dict[Any, float],
    metric: str = "map"
) -> Dict[str, Any]:
    """计算性能极差"""
    if len(subgroup_performances) == 0:
        return {
            "gap": 0.0,
            "max_performance": 0.0,
            "min_performance": 0.0,
            "max_subgroup": None,
            "min_subgroup": None
        }

    max_subgroup = max(subgroup_performances, key=subgroup_performances.get)
    min_subgroup = min(subgroup_performances, key=subgroup_performances.get)
    max_performance = subgroup_performances[max_subgroup]
    min_performance = subgroup_performances[min_subgroup]
    gap = max_performance - min_performance

    return {
        "gap": gap,
        "max_performance": max_performance,
        "min_performance": min_performance,
        "max_subgroup": max_subgroup,
        "min_subgroup": min_subgroup,
        "metric": metric
    }


# ============================================================================
# 5. 主评估函数
# ============================================================================

def evaluate_fairness_detection(
    estimator,
    test_loader,
    fairness_config: Dict[str, Any]
):
    """
    目标检测公平性评估（子群体性能对比测试法）

    Args:
        estimator: 模型估计器
            必须有 predict(images) 方法
            返回 List[Dict]: [{"boxes": np.ndarray, "scores": np.ndarray, "labels": np.ndarray}, ...]

        test_loader: 测试数据加载器
            支持两种批次格式：
              1) YOLO 格式：(images, labels, img_paths, shapes)
              2) detection DataLoader 格式：(images, targets, ...)

        fairness_config: 配置字典
            {
                "sensitive_attribute": {"method": "path_directory"},
                "metric": "map",
                "iou_threshold": 0.5
            }

    返回结果：
        - performance_gap: 性能极差
        - subgroup_performances: 各子群体性能
        - num_subgroups: 子群体数量
        - total_samples: 总样本数
    """
    try:
        # 解析配置
        sensitive_attr_method = fairness_config.get("sensitive_attribute", {}).get("method", "path_directory")
        metric = fairness_config.get("metric", "map")
        iou_threshold = fairness_config.get("iou_threshold", 0.5)

        extract_sensitive_attr = create_sensitive_attribute_extractor(sensitive_attr_method)
        ResultSender.send_log("信息", f"使用敏感属性提取方法: {sensitive_attr_method}")

        # 收集预测结果
        ResultSender.send_log("信息", "进度: 收集模型预测结果...")
        all_predictions = []
        all_ground_truths = []
        all_image_paths = []
        sample_attrs: List[Any] = []

        batch_count = 0
        for batch_data in test_loader:
            batch_count += 1

            # Case 1: YOLO 风格 (images, labels, paths, shapes)
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3 and isinstance(batch_data[2], (list, tuple)):
                images = batch_data[0]
                labels = batch_data[1]
                img_paths = batch_data[2]
                shapes = batch_data[3] if len(batch_data) > 3 else None

                if isinstance(images, torch.Tensor):
                    images_input = images.cpu().numpy()
                else:
                    images_input = np.array(images)

                try:
                    predictions = estimator.predict(images_input)
                except Exception as e:
                    ResultSender.send_log("错误", f"预测失败: {e}")
                    continue

                batch_gts = []
                for i, label in enumerate(labels):
                    if shapes is not None and i < len(shapes):
                        img_shape = shapes[i] if isinstance(shapes[i], tuple) else (640, 640)
                    else:
                        img_shape = (640, 640)

                    if isinstance(label, torch.Tensor):
                        label_np = label.cpu().numpy()
                    else:
                        label_np = np.array(label)

                    if label_np.ndim == 2 and label_np.shape[1] == 6:
                        label_np = label_np[:, 1:]

                    gt_dict = convert_yolo_to_standard_format(label_np, img_shape)
                    batch_gts.append(gt_dict)

                all_predictions.extend(predictions)
                all_ground_truths.extend(batch_gts)
                all_image_paths.extend(img_paths)

                for path in img_paths:
                    try:
                        attr = extract_sensitive_attr(path)
                    except Exception as e:
                        ResultSender.send_log("警告", f"提取失败 ({path}): {e}")
                        attr = "unknown"
                    sample_attrs.append(_normalize_attr_value(attr))

            # Case 2: Detection DataLoader (images list, targets list, ...)
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                images = batch_data[0]
                targets = batch_data[1]
                images_seq = _ensure_list(images)
                targets_seq = _ensure_list(targets)

                if len(images_seq) != len(targets_seq):
                    ResultSender.send_log("错误", "图像数量与标注数量不一致，跳过该批次")
                    continue

                try:
                    predictions = estimator.predict(images_seq)
                except Exception as e:
                    ResultSender.send_log("错误", f"预测失败: {e}")
                    continue

                predictions_seq = _ensure_list(predictions)
                if len(predictions_seq) != len(targets_seq):
                    ResultSender.send_log("错误", "预测结果数量与输入不一致，跳过该批次")
                    continue

                for pred_item, target_item in zip(predictions_seq, targets_seq):
                    converted_pred = _convert_prediction(pred_item)
                    converted_target = _convert_target(target_item)
                    all_predictions.append(converted_pred)
                    all_ground_truths.append(converted_target)

                    sample_index = len(all_image_paths)
                    img_path = _extract_image_path_from_target(target_item, sample_index)
                    all_image_paths.append(img_path)

                    attr = _extract_attr_from_target(target_item)
                    if not attr:
                        try:
                            attr = extract_sensitive_attr(img_path)
                        except Exception as e:
                            ResultSender.send_log("警告", f"提取失败 ({img_path}): {e}")
                            attr = "unknown"
                    sample_attrs.append(_normalize_attr_value(attr))
            else:
                ResultSender.send_log("错误", f"不支持的批次格式: {type(batch_data)}")
                continue

        # ResultSender.send_log("信息", f"进度: 收集完成，总样本: {len(all_predictions)}")

        # 划分子群体
        # ResultSender.send_log("信息", "进度: 划分子群体...")
        subgroups = defaultdict(list)

        if not sample_attrs and all_image_paths:
            for path in all_image_paths:
                try:
                    sample_attrs.append(_normalize_attr_value(extract_sensitive_attr(path)))
                except Exception as e:
                    ResultSender.send_log("警告", f"提取失败 ({path}): {e}")
                    sample_attrs.append(_normalize_attr_value("unknown"))

        for i, attr in enumerate(sample_attrs):
            subgroups[attr].append(i)

        # ResultSender.send_log("信息", f"进度: 划分为 {len(subgroups)} 个子群体:")
        # for name, indices in subgroups.items():
        #     # ResultSender.send_log("信息", f"进度:   {name}: {len(indices)} 样本")

        # 计算子群体性能
        #ResultSender.send_log("信息", "进度: 计算各子群体性能...")
        subgroup_performances = {}

        for name, indices in subgroups.items():
            subgroup_preds = [all_predictions[i] for i in indices]
            subgroup_gts = [all_ground_truths[i] for i in indices]

            try:
                perf = calculate_map_for_subgroup(subgroup_preds, subgroup_gts, iou_threshold)
                subgroup_performances[name] = perf
                #  ResultSender.send_log("信息", f"进度:   {name}: {perf:.4f}")
            except Exception as e:
                ResultSender.send_log("错误", f"计算失败 ({name}): {e}")
                subgroup_performances[name] = 0.0

        # 计算性能极差
        # ResultSender.send_log("信息", "进度: 计算性能极差...")
        gap_result = calculate_performance_gap(subgroup_performances, metric)

        performance_gap = float(gap_result["gap"])
        max_subgroup = gap_result["max_subgroup"] or ""
        min_subgroup = gap_result["min_subgroup"] or ""
        max_performance = float(gap_result["max_performance"])
        min_performance = float(gap_result["min_performance"])

        # ResultSender.send_log("信息", f"进度: 性能极差: {performance_gap:.4f}")
        # ResultSender.send_log("信息", f"进度:   最高: {max_performance:.4f} ({max_subgroup})")
        # ResultSender.send_log("信息", f"进度:   最低: {min_performance:.4f} ({min_subgroup})")

        ResultSender.send_result("performance_gap", float(performance_gap))
        ResultSender.send_result("subgroup_performances", subgroup_performances)
        ResultSender.send_result("num_subgroups", len(subgroups))
        ResultSender.send_result("total_samples", len(all_predictions))
        ResultSender.send_result("max_subgroup", max_subgroup)
        ResultSender.send_result("min_subgroup", min_subgroup)
        ResultSender.send_result("max_performance", float(max_performance))
        ResultSender.send_result("min_performance", float(min_performance))

        ResultSender.send_status("成功")

    except Exception as err:
        ResultSender.send_log("错误", f"公平性评估失败: {err}")
        ResultSender.send_status("失败")
        raise
