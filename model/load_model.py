import importlib
from pathlib import Path

import torch

from model.net.cifar10 import Tudui


def load_model(
        model_def_path: str,  # 模型定义文件路径（如 "model_definitions/tudui.py"）
        class_name: str,  # 模型类名（如 "Tudui"）
        weight_path: str,  # 权重文件路径
        ** model_args  # 模型初始化参数（如 input_size=32）
) -> torch.nn.Module:
    """
    根据模型定义路径动态加载模型并加载权重

    Args:
        model_def_path: 模型定义文件的绝对/相对路径
        class_name:     模型类的名称
        weight_path:    预训练权重路径
        model_args:     模型初始化参数

    Returns:
        加载完成的模型实例（处于eval模式）
    """
    # 1. 动态导入模型定义文件
    model_path = Path(model_def_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型定义文件不存在: {model_path}")

    # 创建模块规范
    spec = importlib.util.spec_from_file_location("custom_model", model_path)
    if spec is None:
        raise ImportError(f"无法导入文件: {model_path}")

    # 加载模块
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 2. 获取模型类
    if not hasattr(module, class_name):
        raise AttributeError(f"模块 {model_path} 中未找到类 {class_name}")
    model_class = getattr(module, class_name)

    # 3. 实例化模型
    try:
        model = model_class(**model_args)
    except Exception as e:
        raise ValueError(f"模型初始化失败: {str(e)}") from e

    # 4. 加载权重
    try:
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"加载权重失败: {str(e)}") from e

    return model
