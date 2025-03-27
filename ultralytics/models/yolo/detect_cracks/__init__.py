# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
裂缝检测模块

该模块提供了基于YOLO的裂缝检测功能，适用于道路、桥梁等基础设施的检测任务。
"""

from .model import CrackDetectionModel, CrackDetectionModelWrapper, register_crack_detection
from ultralytics.models.yolo.detect import DetectionValidator, DetectionPredictor

# 将重要类导出
__all__ = [
    "CrackDetectionModel",
    "CrackDetectionModelWrapper", 
    "DetectionValidator", 
    "DetectionPredictor",
    "register_crack_detection"
]

# 自动注册裂缝检测任务
register_crack_detection() 