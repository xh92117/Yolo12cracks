# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Ultralytics YOLO模块

该模块提供YOLO模型及其变体，支持各种计算机视觉任务。
"""

# 基本模块和重要类导入
from ultralytics.models.yolo import classify, detect, segment, pose, obb, detect_cracks
from ultralytics.models.yolo.model import YOLO

# 定义公共API - 使用列表形式而非逗号分隔的字符串
__all__ = ["classify", "detect", "segment", "pose", "obb", "detect_cracks", "YOLO"]
