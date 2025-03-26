# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
注意力机制模块配置
用于控制消融实验的配置参数
"""

# 全局开关
enable_msae = True  # 是否启用多尺度注意力增强
enable_adaptive_head = True  # 是否启用自适应检测头
enable_size_aware_loss = True  # 是否启用尺寸感知损失

# 特性参数
msae_config = {
    "width_mult": 1.0,  # 宽度乘数
    "scale_levels": 4,  # 尺度划分级别
    "enable_cross_scale": True,  # 是否启用跨尺度信息交换
}

# 自适应检测头参数
adaptive_head_config = {
    "enable_scale_adaptation": True,  # 是否启用尺度自适应
    "small_object_anchor_ratio": [0.8, 1.2],  # 小物体锚框宽高调整比例
}

# 尺寸感知损失参数
loss_config = {
    "small_obj_weight": 1.5,  # 小目标权重
    "medium_obj_weight": 1.0,  # 中等目标权重
    "large_obj_weight": 0.8,  # 大目标权重
    "small_threshold": 32*32,  # 小目标面积阈值
    "large_threshold": 96*96,  # 大目标面积阈值
}

# 获取当前配置信息
def get_config():
    """获取当前注意力机制配置"""
    config = {
        "enable_msae": enable_msae,
        "enable_adaptive_head": enable_adaptive_head,
        "enable_size_aware_loss": enable_size_aware_loss,
        "msae_config": msae_config,
        "adaptive_head_config": adaptive_head_config,
        "loss_config": loss_config,
    }
    return config

# 设置配置
def set_config(config_dict):
    """设置注意力机制配置"""
    global enable_msae, enable_adaptive_head, enable_size_aware_loss
    global msae_config, adaptive_head_config, loss_config
    
    if 'enable_msae' in config_dict:
        enable_msae = config_dict['enable_msae']
    if 'enable_adaptive_head' in config_dict:
        enable_adaptive_head = config_dict['enable_adaptive_head']
    if 'enable_size_aware_loss' in config_dict:
        enable_size_aware_loss = config_dict['enable_size_aware_loss']
    
    # 更新子配置
    if 'msae_config' in config_dict:
        msae_config.update(config_dict['msae_config'])
    if 'adaptive_head_config' in config_dict:
        adaptive_head_config.update(config_dict['adaptive_head_config'])
    if 'loss_config' in config_dict:
        loss_config.update(config_dict['loss_config']) 