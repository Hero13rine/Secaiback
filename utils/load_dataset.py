from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Any

import yaml


class DataLoaderConfigError(RuntimeError):
    pass


class DataLoaderLoadError(RuntimeError):
    pass


def load_dataset() -> Callable[..., Any]:
    """
    Dataloader factory.

    Responsibilities:
    - Read dataloader.yaml
    - Load dataloader module
    - Return the dataloader function itself

    IMPORTANT:
    - This function DOES NOT call load_dataset
    - Caller is responsible for passing parameters
    """

    cfg = _load_yaml_config()
    return _load_dataloader_function(cfg)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _load_yaml_config() -> Dict[str, Any]:
    yaml_path = Path("config/dataloader/dataloader.yaml").resolve()
    if not yaml_path.exists():
        raise DataLoaderConfigError(
            f"dataloader 配置文件不存在: {yaml_path}"
        )

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "dataloader" not in data:
        raise DataLoaderConfigError(
            "dataloader.yaml 必须包含顶级字段 'dataloader'"
        )

    return data["dataloader"]


def _load_dataloader_function(cfg: Dict[str, Any]) -> Callable:
    if "definition" not in cfg:
        raise DataLoaderConfigError(
            "dataloader.yaml 缺少字段: definition"
        )

    module_path = Path(cfg["definition"]).resolve()
    if not module_path.exists():
        raise DataLoaderLoadError(
            f"dataloader 定义文件不存在: {module_path}"
        )

    func_name = cfg.get("function", "load_dataset")

    module_name = f"dataloader_{module_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise DataLoaderLoadError(
            f"无法加载 dataloader 模块: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

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
