"""Expose YOLO model modules and mirror original package name."""

import sys

# 让外部模块可以使用 `import models.*` 的老写法
sys.modules.setdefault("models", sys.modules[__name__])

# 预加载关键子模块，方便直接使用
from . import common  # noqa: F401
from . import experimental  # noqa: F401
from . import yolo  # noqa: F401
from .yolo import Model  # noqa: F401

__all__ = ["common", "experimental", "yolo", "Model"]
