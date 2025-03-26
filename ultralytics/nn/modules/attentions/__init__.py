# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.nn.modules.attentions.msae import ScaleAwareAttention, MultiScaleAttentionEnhancement, AdaptiveScaleHead

__all__ = [
    'ScaleAwareAttention',  # 尺度感知注意力模块
    'MultiScaleAttentionEnhancement',  # 多尺度注意力增强模块
    'AdaptiveScaleHead',  # 自适应尺度检测头
] 