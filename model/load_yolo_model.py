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

# 尝试导入 YOLOv7 的工具函数
try:
    # 尝试从 YOLOv7 代码路径导入
    yolov7_root = "/wkm/secai-common/yolo/yolov7"
    if os.path.exists(yolov7_root) and yolov7_root not in sys.path:
        sys.path.insert(0, yolov7_root)
    
    from utils.general import non_max_suppression, xywh2xyxy
    YOLOV7_UTILS_AVAILABLE = True
except ImportError:
    YOLOV7_UTILS_AVAILABLE = False
    # 如果导入失败，使用 torchvision 的 NMS 作为备选
    try:
        import torchvision
        non_max_suppression = None  # 将在代码中处理
    except ImportError:
        torchvision = None


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
    yolov7_root: str = None,
    model_path: str = None,
    model_class: str = None,
    device: str = "cpu"
) -> nn.Module:
    """
    从权重文件加载 YOLO 模型
    
    Args:
        weight_path: YOLO 权重文件路径（.pt 文件）
        yolov7_root: YOLOv7 代码根目录（可选，会自动检测）
        model_path: 模型定义文件路径（可选，如果权重文件包含模型结构则不需要）
        model_class: 模型类名（可选，如果权重文件包含模型结构则不需要）
        device: 加载设备，默认 "cpu"
    
    Returns:
        YOLO 模型对象（已包装，符合 ART 格式）
    
    """
    print(f"加载 YOLO 权重文件: {weight_path}")
    
    # 获取项目根目录（当前文件所在目录的上两级）
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))
    project_yolo_path = os.path.join(project_root, "utils", "yolo")
    
    # 尝试自动检测 YOLOv7 代码路径（优先使用项目集成的 utils/yolo）
    if yolov7_root is None:
        possible_paths = [
            project_yolo_path,  # 优先使用项目集成的 utils/yolo
            "/wkm/secai-common/yolo/yolov7",  # 备用路径
            os.path.join(os.path.dirname(weight_path), "..", ".."),
            os.path.join(os.path.dirname(weight_path), ".."),
            os.path.dirname(weight_path),
        ]
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            # 检查是否有 models 目录（可能是符号链接）
            if os.path.exists(abs_path) and (os.path.exists(os.path.join(abs_path, "models")) or 
                                             os.path.islink(os.path.join(abs_path, "models"))):
                yolov7_root = abs_path
                break
    
    # 确定实际的 YOLO 代码路径（如果是符号链接，获取真实路径）
    real_yolo_path = None
    if yolov7_root and os.path.exists(yolov7_root):
        if os.path.islink(os.path.join(yolov7_root, "models")):
            # 如果是项目集成的 utils/yolo，需要获取真实的 YOLO 代码路径
            models_link = os.path.join(yolov7_root, "models")
            real_yolo_path = os.path.dirname(os.path.realpath(models_link))
            print(f"✓ 检测到项目集成的 YOLO 代码（符号链接）")
            print(f"  - 项目路径: {yolov7_root}")
            print(f"  - 真实路径: {real_yolo_path}")
        else:
            # 直接使用 YOLO 代码路径
            real_yolo_path = yolov7_root
    
    # 如果找到 YOLOv7 代码路径，添加到 sys.path
    if real_yolo_path and os.path.exists(real_yolo_path):
        
        # 清理可能冲突的模块（特别是 utils 模块）
        modules_to_clear = []
        for mod_name in list(sys.modules.keys()):
            if mod_name == 'utils' or mod_name.startswith('utils.'):
                # 检查是否是 YOLOv7 的 utils 模块
                mod = sys.modules[mod_name]
                if hasattr(mod, '__file__') and mod.__file__:
                    mod_path = os.path.dirname(os.path.abspath(mod.__file__))
                    # 检查是否是项目自己的 utils（不是 YOLO 的 utils）
                    project_utils_path = os.path.join(project_root, "utils")
                    if os.path.abspath(mod_path).startswith(os.path.abspath(project_utils_path)):
                        # 是项目自己的 utils，不清理
                        continue
                    yolov7_utils_path = os.path.join(real_yolo_path, "utils")
                    if os.path.abspath(mod_path) != os.path.abspath(yolov7_utils_path):
                        # 不是 YOLOv7 的 utils，标记为需要清理
                        modules_to_clear.append(mod_name)
        
        # 清理冲突的模块
        for mod_name in modules_to_clear:
            del sys.modules[mod_name]
        
        # 确保 YOLOv7 代码路径在 sys.path 最前面（优先级最高）
        # 使用真实路径而不是符号链接路径
        if real_yolo_path in sys.path:
            sys.path.remove(real_yolo_path)
        sys.path.insert(0, real_yolo_path)
        
        print(f"✓ 找到 YOLOv7 代码路径: {real_yolo_path}")
        print(f"  - sys.path[0]: {sys.path[0]}")
        if modules_to_clear:
            print(f"  - 已清理冲突模块: {len(modules_to_clear)} 个")
        
        # 在加载权重文件之前，先导入 YOLOv7 的 utils 模块（使用直接路径指定）
        # 这样可以避免加载权重文件时的模块导入错误
        try:
            # 先清理可能存在的冲突模块（但保留项目自己的 utils）
            for mod_name in ['utils.datasets', 'utils.general', 'utils.plots', 'utils.torch_utils', 
                           'models', 'models.common', 'models.yolo']:
                if mod_name in sys.modules:
                    try:
                        del sys.modules[mod_name]
                    except Exception:
                        pass
            
            # 使用 importlib 直接从指定路径加载 YOLOv7 的 utils 模块
            # 使用真实路径
            utils_dir = os.path.join(real_yolo_path, "utils")
            
            # 创建 YOLO 的 utils 命名空间（避免与项目自己的 utils 冲突）
            # 但 YOLO 代码期望直接导入 utils，所以我们需要将真实路径添加到 sys.path
            # 这样 YOLO 代码可以正常导入 utils.datasets 等
            
            # 加载 utils.datasets
            utils_datasets_path = os.path.join(utils_dir, "datasets.py")
            if os.path.exists(utils_datasets_path):
                spec = importlib.util.spec_from_file_location("utils.datasets", utils_datasets_path)
                if spec and spec.loader:
                    utils_datasets_module = importlib.util.module_from_spec(spec)
                    # 将 utils.datasets 模块添加到 sys.modules
                    sys.modules['utils.datasets'] = utils_datasets_module
                    # 执行模块加载
                    spec.loader.exec_module(utils_datasets_module)
                    print("✓ YOLOv7 utils.datasets 模块导入成功（使用项目集成路径）")
                else:
                    raise ImportError(f"无法创建模块规范: {utils_datasets_path}")
            else:
                raise FileNotFoundError(f"utils.datasets 文件不存在: {utils_datasets_path}")
            
            # 同样处理 utils.general（models.common 也需要）
            utils_general_path = os.path.join(utils_dir, "general.py")
            if os.path.exists(utils_general_path) and 'utils.general' not in sys.modules:
                spec = importlib.util.spec_from_file_location("utils.general", utils_general_path)
                if spec and spec.loader:
                    utils_general_module = importlib.util.module_from_spec(spec)
                    sys.modules['utils.general'] = utils_general_module
                    spec.loader.exec_module(utils_general_module)
                    print("✓ YOLOv7 utils.general 模块导入成功（使用项目集成路径）")
            
            # 同样处理其他可能需要的 utils 模块
            for utils_module_name in ['utils.plots', 'utils.torch_utils']:
                module_file = utils_module_name.replace('.', '/') + '.py'
                module_path = os.path.join(utils_dir, module_file.replace('utils/', ''))
                if os.path.exists(module_path) and utils_module_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(utils_module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[utils_module_name] = module
                        spec.loader.exec_module(module)
                        print(f"✓ YOLOv7 {utils_module_name} 模块导入成功（使用项目集成路径）")
            
        except Exception as e:
            print(f"⚠️  YOLOv7 utils 模块导入失败: {e}")
            print(f"  当前 sys.path[0]: {sys.path[0] if sys.path else 'N/A'}")
            print(f"  YOLOv7 utils 路径: {os.path.join(real_yolo_path, 'utils', 'datasets.py')}")
            print(f"  文件是否存在: {os.path.exists(os.path.join(real_yolo_path, 'utils', 'datasets.py'))}")
            # 即使导入失败，也尝试继续（可能权重文件不需要这个模块）
            print("  尝试继续加载权重文件...")
    
    # 加载权重文件
    try:
        # PyTorch 2.6+ 默认使用 weights_only=True，但 YOLO 权重文件包含 numpy 对象
        # 需要设置 weights_only=False 来加载（仅当信任权重文件来源时）
        # 注意：加载权重文件时会尝试 unpickle 模型类，需要确保 YOLOv7 代码路径已设置
        
        # 在加载前，确保 YOLOv7 路径在 sys.path 最前面
        if yolov7_root and yolov7_root in sys.path:
            if sys.path[0] != yolov7_root:
                sys.path.remove(yolov7_root)
                sys.path.insert(0, yolov7_root)
        
        # 再次清理可能冲突的模块（在加载前，非常重要）
        if yolov7_root:
            yolov7_root_abs = os.path.abspath(yolov7_root)
            # 清理所有可能冲突的模块（除了已经正确加载的 YOLOv7 模块）
            modules_to_clear_before_load = []
            for mod_name in list(sys.modules.keys()):
                if mod_name in ['utils', 'models'] or mod_name.startswith('utils.') or mod_name.startswith('models.'):
                    mod = sys.modules[mod_name]
                    if hasattr(mod, '__file__') and mod.__file__:
                        mod_path = os.path.dirname(os.path.abspath(mod.__file__))
                        if not mod_path.startswith(yolov7_root_abs):
                            # 不是 YOLOv7 的模块，标记为需要清理
                            modules_to_clear_before_load.append(mod_name)
            
            # 清理冲突的模块
            for mod_name in modules_to_clear_before_load:
                try:
                    del sys.modules[mod_name]
                except Exception:
                    pass
            
            if modules_to_clear_before_load:
                print(f"  - 加载前清理冲突模块: {len(modules_to_clear_before_load)} 个")
        
        # 加载权重文件（与 load_model.py 一致）
        try:
            # PyTorch 2.6+ 默认使用 weights_only=True，但某些权重文件可能包含其他对象
            # 需要设置 weights_only=False 来加载（仅当信任权重文件来源时）
            try:
                checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
            except Exception:
                # 如果 weights_only=True 失败，尝试 weights_only=False
                checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"加载权重文件失败: {str(e)}") from e
        
        # 检查权重文件格式
        if 'model' in checkpoint:
            # 如果包含模型结构
            yolo_model = checkpoint['model']
            print("✓ 从权重文件加载模型结构")
        elif isinstance(checkpoint, nn.Module):
            # 如果整个文件就是模型
            yolo_model = checkpoint
            print("✓ 权重文件本身就是模型")
        else:
            # 如果只是权重字典，需要模型定义文件或 YOLOv7 代码
            if model_path is not None and model_class is not None:
                # 使用模型定义文件加载（与 load_model.py 一致）
                # 1. 动态导入模型定义文件
                model_def_path = Path(model_path).resolve()
                if not model_def_path.exists():
                    raise FileNotFoundError(f"模型定义文件不存在: {model_def_path}")
                
                # 创建模块规范
                spec = importlib.util.spec_from_file_location("custom_model", model_def_path)
                if spec is None:
                    raise ImportError(f"无法导入文件: {model_def_path}")
                
                # 加载模块
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 2. 获取模型类
                if not hasattr(module, model_class):
                    raise AttributeError(f"模块 {model_def_path} 中未找到类 {model_class}")
                model_class_obj = getattr(module, model_class)
                
                # 3. 实例化模型（使用空参数字典）
                try:
                    yolo_model = model_class_obj()
                except Exception as e:
                    # 增强错误提示
                    valid_args = inspect.getfullargspec(model_class_obj.__init__).args[1:]  # 排除self
                    raise ValueError(
                        f"模型初始化参数错误，有效参数为: {valid_args}\n"
                        f"当前参数: []"
                    ) from e
                
                # 4. 加载权重（与 load_model.py 一致）
                try:
                    # 如果 checkpoint 是字典且包含 'model' 键，提取模型部分
                    state_dict = checkpoint
                    if isinstance(state_dict, dict) and 'model' in state_dict:
                        state_dict = state_dict['model'].state_dict() if hasattr(state_dict['model'], 'state_dict') else state_dict['model']
                    
                    yolo_model.load_state_dict(state_dict)
                    yolo_model.eval()
                    print("✓ 从模型定义文件加载模型")
                except Exception as e:
                    raise RuntimeError(f"加载权重失败: {str(e)}") from e
            elif real_yolo_path and os.path.exists(os.path.join(real_yolo_path, "models", "yolo.py")):
                # 尝试使用 YOLOv7 官方代码加载
                try:
                    # 确保使用真实路径导入
                    if real_yolo_path not in sys.path:
                        sys.path.insert(0, real_yolo_path)
                    from models.yolo import Model
                    
                    # 尝试从权重文件中获取配置信息
                    if 'model' in checkpoint:
                        model_obj = checkpoint.get('model')
                        if hasattr(model_obj, 'yaml'):
                            cfg = model_obj.yaml
                        else:
                            # 使用默认配置
                            cfg_paths = [
                                os.path.join(real_yolo_path, "cfg", "training", "yolov7.yaml"),
                                os.path.join(real_yolo_path, "yolov7.yaml"),
                            ]
                            cfg = None
                            for cfg_path in cfg_paths:
                                if os.path.exists(cfg_path):
                                    cfg = cfg_path
                                    break
                            if cfg is None:
                                raise FileNotFoundError(f"未找到 YOLOv7 配置文件，尝试过的路径: {cfg_paths}")
                        
                        yolo_model = Model(cfg=cfg)
                        # 加载权重（与 load_model.py 一致）
                        state_dict = checkpoint['model']
                        if isinstance(state_dict, nn.Module):
                            state_dict = state_dict.state_dict()
                        yolo_model.load_state_dict(state_dict)
                    else:
                        # 只有权重，需要配置文件
                        cfg_paths = [
                            os.path.join(real_yolo_path, "cfg", "training", "yolov7.yaml"),
                            os.path.join(real_yolo_path, "yolov7.yaml"),
                        ]
                        cfg = None
                        for cfg_path in cfg_paths:
                            if os.path.exists(cfg_path):
                                cfg = cfg_path
                                break
                        if cfg is None:
                            raise FileNotFoundError(f"未找到 YOLOv7 配置文件，尝试过的路径: {cfg_paths}")
                        
                        yolo_model = Model(cfg=cfg)
                        # 加载权重（与 load_model.py 一致）
                        state_dict = checkpoint
                        if isinstance(state_dict, dict) and 'model' in state_dict:
                            state_dict = state_dict['model'].state_dict() if hasattr(state_dict['model'], 'state_dict') else state_dict['model']
                        yolo_model.load_state_dict(state_dict)
                    print("✓ 使用 YOLOv7 官方代码加载模型")
                except Exception as e:
                    raise ValueError(
                        f"使用 YOLOv7 官方代码加载失败: {e}\n"
                        f"请确保 YOLOv7 代码路径正确，或提供 model_path 和 model_class"
                    )
            else:
                raise ValueError(
                    "权重文件不包含模型结构，需要提供以下之一：\n"
                    "1. model_path 和 model_class（模型定义文件）\n"
                    "2. yolov7_root（YOLOv7 代码路径，会自动加载）\n"
                    "3. 确保权重文件包含完整的模型结构"
                )
        
        # 设置为评估模式
        yolo_model.eval()
        
        # 包装模型以符合 ART 格式
        wrapped_model = YOLOWrapper(yolo_model)
        
        # 设置所有参数的 requires_grad=True，以便优化器可以正常工作
        # YOLO 模型在加载时所有参数的 requires_grad 可能都是 False
        for param in wrapped_model.parameters():
            param.requires_grad = True
        
        print("✓ YOLO 模型加载成功")
        return wrapped_model
        
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        raise RuntimeError(f"加载 YOLO 模型失败: {str(e)}") from e


def load_yolov7_model(weight_path: str, yolov7_root: str = None, device: str = "cpu") -> nn.Module:
    """
    从权重文件加载 YOLOv7 模型（便捷函数）
    
    Args:
        weight_path: YOLOv7 权重文件路径（.pt 文件）
        yolov7_root: YOLOv7 代码根目录（可选，会自动检测）
        device: 加载设备，默认 "cpu"
    
    Returns:
        YOLOv7 模型对象（已包装，符合 ART 格式）
    """
    return load_yolo_model(weight_path, yolov7_root=yolov7_root, device=device)
