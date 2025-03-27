# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.63"

import os
import functools
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Union

# 设置环境变量(在所有导入之前)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # 默认值，减少训练过程中的CPU使用率

# 定义延迟加载器
class LazyLoader:
    """延迟加载器，避免循环导入问题"""
    
    def __init__(self):
        self._cache = {}
    
    def __call__(self, name: str) -> Any:
        """延迟加载指定的模块或属性"""
        if name not in self._cache:
            if name == "YOLO":
                from ultralytics.models.yolo.model import YOLO as _YOLO
                self._cache[name] = _YOLO
            elif name == "ASSETS":
                from ultralytics.utils import ASSETS as _ASSETS
                self._cache[name] = _ASSETS
            elif name == "SETTINGS":
                from ultralytics.utils import SETTINGS as _SETTINGS
                self._cache[name] = _SETTINGS
            elif name == "checks":
                from ultralytics.utils.checks import check_yolo as _checks
                self._cache[name] = _checks
            elif name == "download":
                from ultralytics.utils.downloads import download as _download
                self._cache[name] = _download
            else:
                raise ValueError(f"Unknown module or attribute: {name}")
        return self._cache[name]

# 创建延迟加载器实例
_lazy_loader = LazyLoader()

# 导出公共API
def YOLO(*args, **kwargs):
    """YOLO模型工厂函数"""
    return _lazy_loader("YOLO")(*args, **kwargs)

@property
def ASSETS():
    """Ultralytics资源目录"""
    return _lazy_loader("ASSETS")

@property
def settings():
    """Ultralytics设置"""
    return _lazy_loader("SETTINGS")

def checks(*args, **kwargs):
    """运行系统检查"""
    return _lazy_loader("checks")(*args, **kwargs)

def download(*args, **kwargs):
    """下载资源"""
    return _lazy_loader("download")(*args, **kwargs)

# 定义导出的公共API
__all__ = ("__version__", "YOLO", "ASSETS", "settings", "checks", "download")
