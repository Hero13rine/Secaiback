from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Any


class DataLoaderLoadError(RuntimeError):
    pass


def load_dataloader(
    dataloader_def_path: str | Path,
    func_name: str = "load_data_robustness",
) -> Callable[..., Any]:
    """
    Dynamically load a dataloader function from an external Python file.

    This factory:
    - DOES NOT read any YAML
    - DOES NOT call the dataloader
    - ONLY returns the dataloader function itself

    Args:
        dataloader_def_path:
            Path to dataloader Python file (e.g. "./load_dataset.py")
        func_name:
            Name of the dataloader function to load (default: load_dataset)

    Returns:
        Callable dataloader function (e.g. load_dataset)

    Usage:
        load_dataset = load_dataloader("./load_dataset.py")
        _, _, test_loader = load_dataset(test_root=..., batch_size=2)
    """

    module_path = Path(dataloader_def_path).resolve()
    if not module_path.exists():
        raise DataLoaderLoadError(
            f"dataloader 定义文件不存在: {module_path}"
        )

    module_name = f"dataloader_{module_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise DataLoaderLoadError(
            f"无法创建模块规范: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise DataLoaderLoadError(
            f"执行 dataloader 模块失败: {module_path}\n"
            f"错误类型: {type(e).__name__}\n"
            f"错误信息: {e}"
        ) from e

    if not hasattr(module, func_name):
        raise DataLoaderLoadError(
            f"{module_path} 中未定义函数 {func_name}"
        )

    fn = getattr(module, func_name)
    if not callable(fn):
        raise DataLoaderLoadError(
            f"{func_name} 不是可调用对象"
        )

    return fn
