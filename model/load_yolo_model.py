"""
YOLO 模型加载函数
专门用于加载 YOLO 系列模型（YOLOv5、YOLOv7 等）
支持从权重文件中加载包含模型结构的权重文件
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path

import torch
import torch.nn as nn

import utils as project_utils
from utils.yolo.yolo_utils import datasets as yolo_datasets_module
from utils.yolo.yolo_utils import general as yolo_general_module
from utils.yolo.yolo_utils import plots as yolo_plots_module
from utils.yolo.yolo_utils import torch_utils as yolo_torch_utils_module
import utils.yolo.models as yolo_models_package
from utils.yolo.models import common as yolo_models_common_module
from utils.yolo.models import experimental as yolo_models_experimental_module
from utils.yolo.models import yolo as yolo_models_yolo_module
from utils.yolo.models.yolo import Model

non_max_suppression = yolo_general_module.non_max_suppression
xywh2xyxy = yolo_general_module.xywh2xyxy
YOLOV7_UTILS_AVAILABLE = True

LOCAL_YOLO_ROOT = (Path(__file__).resolve().parents[1] / "utils" / "yolo").resolve()


def _register_local_yolo_modules() -> None:
    """Ensure YOLO checkpoints can resolve their historical module paths."""

    module_aliases = {
        "utils.datasets": yolo_datasets_module,
        "utils.general": yolo_general_module,
        "utils.plots": yolo_plots_module,
        "utils.torch_utils": yolo_torch_utils_module,
        "models": yolo_models_package,
        "models.common": yolo_models_common_module,
        "models.experimental": yolo_models_experimental_module,
        "models.yolo": yolo_models_yolo_module,
    }

    for alias, module in module_aliases.items():
        sys.modules[alias] = module

    # 让项目内的 utils 包也能直接访问这些模块
    setattr(project_utils, "datasets", yolo_datasets_module)
    setattr(project_utils, "general", yolo_general_module)
    setattr(project_utils, "plots", yolo_plots_module)
    setattr(project_utils, "torch_utils", yolo_torch_utils_module)


_register_local_yolo_modules()


def _resolve_local_cfg_path() -> Path:
    """Locate a YOLO configuration file shipped with the repository."""

    candidates = [
        LOCAL_YOLO_ROOT / "cfg" / "training" / "yolov7.yaml",
        LOCAL_YOLO_ROOT / "yolov7.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未在 {LOCAL_YOLO_ROOT} 下找到可用的 YOLO 配置文件: {candidates}")


def _load_custom_model_from_definition(
    model_path: str,
    model_class: str,
    checkpoint: dict,
) -> nn.Module:
    """Load a model defined by user-provided python file + class name."""

    model_def_path = Path(model_path).resolve()
    if not model_def_path.exists():
        raise FileNotFoundError(f"模型定义文件不存在: {model_def_path}")

    spec = importlib.util.spec_from_file_location("custom_model", model_def_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模型定义文件: {model_def_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, model_class):
        raise AttributeError(f"模块 {model_def_path} 中未找到类 {model_class}")

    model_class_obj = getattr(module, model_class)
    try:
        yolo_model = model_class_obj()
    except Exception as exc:
        valid_args = inspect.getfullargspec(model_class_obj.__init__).args[1:]
        raise ValueError(
            f"实例化自定义模型 {model_class} 失败: {exc}. 期望参数: {valid_args}"
        ) from exc

    state_dict = checkpoint
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict and hasattr(state_dict["model"], "state_dict"):
            state_dict = state_dict["model"].state_dict()
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()

    yolo_model.load_state_dict(state_dict)
    print("✓ 使用自定义模型定义加载权重")
    return yolo_model



class YOLOWrapper(nn.Module):
    """
    YOLO 模型包装器
    将 YOLO 的输出格式转换为 ART 要求的格式
    """
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model
        # 检测模型的数据类型（用于匹配输入类型）
        self._dtype = None
        if hasattr(yolo_model, 'parameters'):
            try:
                first_param = next(yolo_model.parameters())
                self._dtype = first_param.dtype
            except StopIteration:
                self._dtype = None
    
    def forward(self, x, *args, **kwargs):
        """
        将 YOLO 输出转换为 ART 要求的格式
        
        YOLO 输出格式可能不同，需要根据实际输出调整
        ART 要求格式: List[Dict[str, torch.Tensor]]
        每个字典包含: boxes [N, 4], labels [N], scores [N]
        
        Args:
            x: 输入图像 tensor 或 List[torch.Tensor]（ART 可能传递 List）
            *args: 额外的位置参数（ART 可能传递）
            **kwargs: 额外的关键字参数（ART 可能传递）
        """
        # 处理 ART 可能传递的 List[torch.Tensor] 格式
        if isinstance(x, list):
            # ART 传递的是 List[torch.Tensor]，需要转换为 batch tensor
            if len(x) == 0:
                return []
            # 堆叠为 batch tensor
            x = torch.stack(x, dim=0)
        
        # 确保输入数据在正确的设备上（与模型权重同一设备）
        if hasattr(self.yolo_model, 'parameters'):
            model_param = next(self.yolo_model.parameters())
            model_device = model_param.device
            model_dtype = model_param.dtype
            
            # 移动到正确的设备
            if x.device != model_device:
                x = x.to(device=model_device)
            
            # 处理数据类型和归一化
            # 1. 如果是 uint8 或 int8，转换为 float 并归一化到 [0, 1]
            if x.dtype == torch.uint8 or x.dtype == torch.int8:
                x = x.float() / 255.0  # 归一化到 [0, 1]
            # 2. 如果是 float32 或 float16，检查是否需要归一化
            elif x.dtype in (torch.float32, torch.float16):
                x_min = x.min().item()
                x_max = x.max().item()
                # 如果数值范围不在 [0, 1] 范围内（允许小的误差），说明需要归一化
                # YOLO 模型期望输入在 [0, 1] 范围内
                if x_max > 1.1 or x_min < -0.1:
                    # 如果数值范围看起来像是像素值（0-255），则归一化
                    if x_max <= 255.0 and x_min >= 0.0:
                        x = x / 255.0  # 归一化到 [0, 1]
            
            # 转换为与模型权重相同的类型
            if x.dtype != model_dtype:
                x = x.to(dtype=model_dtype)
        else:
            # 如果没有参数，使用默认处理
            if self._dtype is not None and x.dtype != self._dtype:
                # 如果是 uint8，先转换为 float 并归一化
                if x.dtype == torch.uint8 or x.dtype == torch.int8:
                    x = x.float() / 255.0
                x = x.to(dtype=self._dtype)
        
        # 调用原始 YOLO 模型（忽略额外的参数）
        outputs = self.yolo_model(x)
        
        batch_size = x.shape[0]
        results = []
        
        # YOLO 模型在 eval 模式下通常返回 tuple: (detections, features)
        # detections 格式: [batch, num_boxes, nc+5]，其中前4个是 [x, y, w, h]（中心点+宽高），第5个是 conf，后面是类别分数
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # 取第一个元素（检测结果）
        
        # 处理每个样本的输出
        for i in range(batch_size):
            if isinstance(outputs, torch.Tensor):
                # 如果输出是单个 tensor
                if len(outputs.shape) == 2:
                    # 格式: [num_boxes, 6] 或 [num_boxes, 5] 或 [num_boxes, nc+5]
                    output_i = outputs
                elif len(outputs.shape) == 3:
                    # 格式: [batch, num_boxes, 6] 或 [batch, num_boxes, nc+5]
                    output_i = outputs[i]
                else:
                    # 默认处理：创建空结果
                    boxes = torch.zeros((0, 4), device=x.device)
                    scores = torch.zeros(0, device=x.device)
                    labels = torch.zeros(0, dtype=torch.long, device=x.device)
                    result = {
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores
                    }
                    results.append(result)
                    continue
                
                # 处理 YOLO 输出格式
                if output_i.shape[0] == 0:
                    # 空检测结果
                    boxes = torch.zeros((0, 4), device=x.device)
                    scores = torch.zeros(0, device=x.device)
                    labels = torch.zeros(0, dtype=torch.long, device=x.device)
                else:
                    # YOLO 输出格式: [num_boxes, nc+5]
                    # 前4个: [x, y, w, h] (中心点+宽高，像素坐标)
                    # 第5个: conf (目标置信度)
                    # 后面: 类别分数 [nc]
                    
                    # 获取图像尺寸
                    img_h, img_w = x.shape[-2], x.shape[-1]
                    
                    # 使用 YOLOv7 官方的 non_max_suppression 进行 NMS 处理
                    # YOLOv7 的 non_max_suppression 需要输入格式：[batch, num_boxes, nc+5]
                    # 格式：[x, y, w, h, conf, class_scores...]
                    # 它会内部处理：conf = obj_conf * cls_conf，转换为 xyxy 格式，然后应用 NMS
                    if YOLOV7_UTILS_AVAILABLE and non_max_suppression is not None:
                        # 使用 YOLOv7 官方的 non_max_suppression
                        # output_i 已经是 [num_boxes, nc+5] 格式，需要添加 batch 维度
                        prediction = output_i.unsqueeze(0)  # [1, num_boxes, nc+5]
                        
                        # 使用 YOLOv7 官方的 non_max_suppression
                        # conf_thres=0.001, iou_thres=0.45 (与 YOLOv7 官方测试一致)
                        nms_output = non_max_suppression(
                            prediction,
                            conf_thres=0.001,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False
                        )
                        
                        # non_max_suppression 返回 List[Tensor]，每个 tensor 是 [num_boxes, 6]
                        # 格式：[x1, y1, x2, y2, conf, cls]
                        if len(nms_output) > 0 and len(nms_output[0]) > 0:
                            pred = nms_output[0]  # [num_boxes, 6]
                            boxes = pred[:, :4]  # [x1, y1, x2, y2]
                            scores = pred[:, 4]  # conf
                            labels = pred[:, 5].long()  # cls
                            
                            # 限制类别索引在有效范围内 [0, 19]（DIOR 数据集有20个类别）
                            max_class_id = 19
                            labels = torch.clamp(labels, 0, max_class_id)
                            
                            # 裁剪坐标到图像范围内
                            boxes[:, 0] = torch.clamp(boxes[:, 0], 0, img_w)  # x1
                            boxes[:, 1] = torch.clamp(boxes[:, 1], 0, img_h)  # y1
                            boxes[:, 2] = torch.clamp(boxes[:, 2], 0, img_w)  # x2
                            boxes[:, 3] = torch.clamp(boxes[:, 3], 0, img_h)  # y2
                            
                            # 确保边界框有效（x1 < x2, y1 < y2）
                            valid_size = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                            min_size = 1.0
                            valid_size = valid_size & ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
                            
                            # 过滤无效的边界框
                            boxes = boxes[valid_size]
                            scores = scores[valid_size]
                            labels = labels[valid_size]
                            
                            # 限制最终检测数量（按置信度排序）
                            max_final_detections = 300
                            if len(boxes) > max_final_detections:
                                _, top_indices = torch.topk(scores, max_final_detections)
                                boxes = boxes[top_indices]
                                scores = scores[top_indices]
                                labels = labels[top_indices]
                        else:
                            # 如果没有检测结果，创建空结果
                            boxes = torch.zeros((0, 4), device=x.device, dtype=x.dtype)
                            scores = torch.zeros(0, device=x.device, dtype=x.dtype)
                            labels = torch.zeros(0, dtype=torch.long, device=x.device)
            elif isinstance(outputs, list):
                # 如果输出是列表
                output_i = outputs[i] if i < len(outputs) else outputs[0]
                if isinstance(output_i, dict):
                    # 已经是 ART 格式
                    boxes = output_i.get('boxes', torch.zeros((0, 4), device=x.device))
                    scores = output_i.get('scores', torch.zeros(0, device=x.device))
                    labels = output_i.get('labels', torch.zeros(0, dtype=torch.long, device=x.device))
                else:
                    # 假设是 tensor
                    if isinstance(output_i, torch.Tensor):
                        if len(output_i.shape) == 2:
                            boxes = output_i[:, :4]
                            scores = output_i[:, 4] if output_i.shape[1] > 4 else torch.ones(output_i.shape[0], device=x.device)
                            labels = output_i[:, 5].long() if output_i.shape[1] > 5 else torch.zeros(output_i.shape[0], dtype=torch.long, device=x.device)
                        else:
                            boxes = torch.zeros((0, 4), device=x.device)
                            scores = torch.zeros(0, device=x.device)
                            labels = torch.zeros(0, dtype=torch.long, device=x.device)
                    else:
                        boxes = torch.zeros((0, 4), device=x.device)
                        scores = torch.zeros(0, device=x.device)
                        labels = torch.zeros(0, dtype=torch.long, device=x.device)
            else:
                # 默认处理
                boxes = torch.zeros((0, 4), device=x.device)
                scores = torch.zeros(0, device=x.device)
                labels = torch.zeros(0, dtype=torch.long, device=x.device)
            
            # 确保 boxes 格式正确 [x1, y1, x2, y2]
            if boxes.shape[0] > 0:
                # 确保 x1 < x2, y1 < y2
                boxes[:, [0, 2]] = torch.sort(boxes[:, [0, 2]], dim=1)[0]
                boxes[:, [1, 3]] = torch.sort(boxes[:, [1, 3]], dim=1)[0]
            
            result = {
                "boxes": boxes,
                "labels": labels,
                "scores": scores
            }
            results.append(result)
        
        return results


def load_yolo_model(
    weight_path: str,
    model_path: str = None,
    model_class: str = None,
    device: str = "cpu"
) -> nn.Module:
    """
    仅依赖本地 utils/yolo 代码加载 YOLO 模型。
    """
    print(f"加载 YOLO 权重文件: {weight_path}")

    weight_path_obj = Path(weight_path)
    if not weight_path_obj.exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    if not LOCAL_YOLO_ROOT.exists():
        raise FileNotFoundError(f"本地 YOLO 代码路径不存在: {LOCAL_YOLO_ROOT}")

    try:
        try:
            checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"加载权重文件失败: {exc}") from exc

    if isinstance(checkpoint, nn.Module):
        yolo_model = checkpoint
        print("✓ 权重文件本身包含模型")
    elif 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
        yolo_model = checkpoint['model']
        print("✓ 从 checkpoint['model'] 获取模型对象")
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        yolo_model = Model(cfg=str(_resolve_local_cfg_path()))
        state_dict = checkpoint['model']
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        yolo_model.load_state_dict(state_dict)
        print("✓ 从 checkpoint['model'] 字典加载权重")
    else:
        if model_path and model_class:
            yolo_model = _load_custom_model_from_definition(model_path, model_class, checkpoint)
        else:
            cfg_path = _resolve_local_cfg_path()
            yolo_model = Model(cfg=str(cfg_path))
            state_dict = checkpoint
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict and hasattr(state_dict['model'], 'state_dict'):
                    state_dict = state_dict['model'].state_dict()
            if isinstance(state_dict, nn.Module):
                state_dict = state_dict.state_dict()
            yolo_model.load_state_dict(state_dict)
            print(f"✓ 使用本地配置 {cfg_path} 实例化模型并加载权重")

    yolo_model.eval()
    wrapped_model = YOLOWrapper(yolo_model)
    for param in wrapped_model.parameters():
        param.requires_grad = True

    print("✓ YOLO 模型加载成功")
    return wrapped_model

def load_yolov7_model(weight_path: str, device: str = "cpu") -> nn.Module:
    """
    从权重文件加载 YOLOv7 模型（便捷函数）
    
    Args:
        weight_path: YOLOv7 权重文件路径（.pt 文件）
        device: 加载设备，默认 "cpu"
    
    Returns:
        YOLOv7 模型对象（已包装，符合 ART 格式）
    """
    return load_yolo_model(weight_path, device=device)
