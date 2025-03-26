# Ultralytics YOLO 🚀, AGPL-3.0 license

from .model import CrackDetectionModel, CrackDetectionModelWrapper, register_crack_detection

# 导出公共组件
__all__ = ['CrackDetectionModel', 'CrackDetectionModelWrapper', 'register_crack_detection']

# 自动注册裂缝检测任务
register_crack_detection() 