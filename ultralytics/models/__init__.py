# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Ultralytics 模型模块

该模块提供Ultralytics的所有模型类，并负责各种模型任务的注册。
为避免循环导入问题，使用延迟导入技术。
"""

from typing import Dict, Type, Any, Callable, Optional

# 通用构造函数，避免循环导入
def _create_model_factory(task_name: str) -> Callable:
    """为指定任务创建模型工厂函数"""
    def factory(*args, **kwargs):
        # 仅当需要时才导入模型类
        if task_name == "YOLO":
            from ultralytics.models.yolo.model import YOLO as ModelClass
        elif task_name == "detect_cracks":
            # 确保裂缝检测模型已注册
            from ultralytics.models.yolo.detect_cracks.model import CrackDetectionModelWrapper as ModelClass
            # 注册任务
            _ensure_task_registered()
        else:
            raise ValueError(f"未知任务: {task_name}")
        return ModelClass(*args, **kwargs)
    return factory

def _ensure_task_registered():
    """确保裂缝检测任务已注册到YOLO模型中"""
    from ultralytics.models.yolo.detect_cracks.model import register_crack_detection
    register_crack_detection()

# 创建公共API
YOLO = _create_model_factory("YOLO")

# 注册自定义任务(在首次导入时调用)
def register_custom_tasks():
    """注册所有自定义任务到模型系统"""
    _ensure_task_registered()

# 导出公共API
__all__ = ["YOLO", "register_custom_tasks"]

# 自动注册自定义任务
register_custom_tasks()
