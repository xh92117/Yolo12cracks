# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入配置
from ultralytics.nn.modules.attentions.config import (
    enable_msae, 
    enable_adaptive_head, 
    msae_config,
    adaptive_head_config
)


class ScaleAwareAttention(nn.Module):
    """
    尺度感知注意力模块，专为处理目标尺寸差异大的情况设计。
    
    该模块通过学习不同尺度的特征图权重，动态调整对不同大小目标的响应，特别适合裂缝检测场景。
    
    参数:
        in_channels (int): 输入通道数
        reduction (int): 通道降维比例
        scale_levels (int): 尺度划分级别
    """
    
    def __init__(self, in_channels, reduction=16, scale_levels=None):
        super().__init__()
        self.in_channels = in_channels
        # 使用配置中的尺度级别数，如未指定则使用默认值
        self.scale_levels = scale_levels or msae_config.get("scale_levels", 4)
        
        # 通道注意力分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP，但有多组权重用于不同尺度
        self.channel_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
            ) for _ in range(self.scale_levels)
        ])
        
        # 选择权重
        self.scale_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.scale_levels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # 尺度自适应激活
        self.scale_act = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, _, h, w = x.shape
        
        if not enable_msae:
            # 如果禁用MSAE，直接返回输入
            return x
            
        # 计算全局特征
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        global_feat = torch.cat([avg_out, max_out], dim=1)
        
        # 计算尺度权重
        scale_weights = self.scale_attention(global_feat)  # B x scale_levels x 1 x 1
        
        # 多尺度通道注意力
        channel_attn = torch.zeros_like(avg_out)
        for i in range(self.scale_levels):
            scale_attn = self.channel_mlp[i](avg_out) + self.channel_mlp[i](max_out)
            channel_attn += scale_weights[:, i:i+1] * scale_attn
        
        channel_attn = self.scale_act(channel_attn)
        
        # 空间注意力，捕获裂缝的空间形态
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_attn = self.scale_act(self.spatial_conv(spatial_feat))
        
        # 结合通道和空间注意力
        refined = x * channel_attn * spatial_attn
        
        return refined


class MultiScaleAttentionEnhancement(nn.Module):
    """
    多尺度注意力增强模块(MSAE)，用于平衡不同尺寸目标的检测性能。
    
    该模块自适应地处理从小到大各种尺寸的裂缝，提高小裂缝的检测精度的同时保持对大裂缝的检测能力。
    
    参数:
        in_channels (list): 不同特征层的通道数列表
        width_mult (float): 宽度乘数，控制模块复杂度
    """
    
    def __init__(self, in_channels, width_mult=None):
        super().__init__()
        self.n_layers = len(in_channels)
        
        # 使用配置中的宽度乘数
        self.width_mult = width_mult or msae_config.get("width_mult", 1.0)
        # 是否启用跨尺度信息交换
        self.enable_cross_scale = msae_config.get("enable_cross_scale", True)
        
        # 针对P3/P4/P5特征层的尺度注意力增强
        self.scale_attentions = nn.ModuleList([
            ScaleAwareAttention(
                in_channels[i], 
                reduction=max(int(16 / self.width_mult), 1),
                scale_levels=msae_config.get("scale_levels", 4)
            ) for i in range(self.n_layers)
        ])
        
        if self.enable_cross_scale:
            # 跨层交互，小尺度特征辅助大尺度特征，反之亦然
            self.down_convs = nn.ModuleList([
                nn.Conv2d(in_channels[i], in_channels[i+1], 
                        kernel_size=3, stride=2, padding=1)
                for i in range(self.n_layers-1)
            ])
            
            self.up_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels[i], in_channels[i-1], 
                            kernel_size=1, stride=1)
                ) for i in range(1, self.n_layers)
            ])
            
            # 特征融合
            self.fusions = nn.ModuleList([
                nn.Conv2d(in_channels[i] * 2, in_channels[i], 
                        kernel_size=1, stride=1)
                for i in range(1, self.n_layers-1)
            ])
            
            # P3层特殊处理，只接收来自P4的特征
            self.fusion_p3 = nn.Conv2d(in_channels[0] * 2, in_channels[0], 
                                    kernel_size=1, stride=1)
            
            # P5层特殊处理，只接收来自P4的特征
            self.fusion_p5 = nn.Conv2d(in_channels[-1] * 2, in_channels[-1], 
                                    kernel_size=1, stride=1)
            
            # 动态权重调节器，根据输入动态调整不同尺度的重要性
            self.scale_weights = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels[i], 1, kernel_size=1),
                    nn.Sigmoid()
                ) for i in range(self.n_layers)
            ])
    
    def forward(self, features):
        """
        前向传播
        
        参数:
            features (list): 特征图列表，通常对应P3, P4, P5特征层
            
        返回:
            list: 增强后的特征图列表
        """
        # 如果MSAE被禁用，直接返回输入
        if not enable_msae:
            return features
            
        # 确保输入格式正确
        if isinstance(features, list) and len(features) == 1:
            features = features[0]  # 处理嵌套列表情况
            
        assert len(features) == self.n_layers, \
            f"Expected {self.n_layers} feature maps, got {len(features)}"
        
        # 先应用尺度注意力
        enhanced = [attn(feat) for feat, attn in zip(features, self.scale_attentions)]
        
        # 如果不启用跨尺度信息交换，直接返回经过注意力增强的特征
        if not self.enable_cross_scale:
            return enhanced
            
        # 特征自下而上传递（小目标特征帮助大目标）
        down_feats = [enhanced[0]]
        for i in range(self.n_layers - 1):
            down_feat = self.down_convs[i](down_feats[-1])
            down_feats.append(down_feat)
        
        # 特征自上而下传递（大目标特征帮助小目标）
        up_feats = [enhanced[-1]]
        for i in range(self.n_layers - 1, 0, -1):
            up_feat = self.up_convs[self.n_layers-1-i](up_feats[-1])
            up_feats.append(up_feat)
        
        up_feats = up_feats[::-1]  # 反转顺序使其与enhanced对应
        
        # 特征融合
        results = []
        
        # P3特殊处理
        p3_weight = self.scale_weights[0](enhanced[0])
        p3_enhanced = torch.cat([enhanced[0], up_feats[0]], dim=1)
        p3_enhanced = self.fusion_p3(p3_enhanced)
        results.append(enhanced[0] + p3_weight * p3_enhanced)
        
        # 中间层融合：同时接收上下层信息
        for i in range(1, self.n_layers-1):
            fusion_feat = torch.cat([enhanced[i], 
                                    down_feats[i] + up_feats[i]], dim=1)
            fusion_feat = self.fusions[i-1](fusion_feat)
            weight = self.scale_weights[i](enhanced[i])
            results.append(enhanced[i] + weight * fusion_feat)
        
        # P5特殊处理
        p5_weight = self.scale_weights[-1](enhanced[-1])
        p5_enhanced = torch.cat([enhanced[-1], down_feats[-1]], dim=1)
        p5_enhanced = self.fusion_p5(p5_enhanced)
        results.append(enhanced[-1] + p5_weight * p5_enhanced)
        
        return results


class AdaptiveScaleHead(nn.Module):
    """
    自适应尺度检测头，针对裂缝检测中的尺寸差异问题特别优化
    
    特点：
    1. 针对不同尺度的目标动态调整预测策略
    2. 对小目标增强特征表达
    3. 学习不同尺度目标的最佳锚框分配策略
    
    参数:
        in_channels (list): 输入特征层通道数列表
        nc (int): 类别数量
        anchors (list): 锚框配置
        stride (list): 特征图相对原图的步长
    """
    
    def __init__(self, in_channels, nc=1, anchors=None, stride=None):
        super().__init__()
        self.nc = nc  # 类别数
        self.na = len(anchors[0]) // 2 if anchors is not None else 3  # 每层锚框数
        self.no = nc + 5  # 每个锚框的输出数量 (类别+置信度+xywh)
        
        # 检查是否启用自适应头部
        self.enable_adaptation = adaptive_head_config.get("enable_scale_adaptation", True)
        
        # 使用多尺度注意力增强模块
        self.msae = MultiScaleAttentionEnhancement(in_channels)
        
        # 轻量级特征提取器，减少参数量
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], in_channels[i], 
                          kernel_size=3, padding=1, groups=in_channels[i]),
                nn.BatchNorm2d(in_channels[i]),
                nn.SiLU(),
                nn.Conv2d(in_channels[i], in_channels[i], kernel_size=1),
            ) for i in range(len(in_channels))
        ])
        
        # 预测头，每个尺度一个
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels[i], self.no * self.na, kernel_size=1)
            for i in range(len(in_channels))
        ])
        
        # 尺度自适应模块，学习最佳锚点分配和尺度权重
        if self.enable_adaptation:
            self.scale_adapter = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels[i], 2, kernel_size=1),
                    nn.Sigmoid()
                ) for i in range(len(in_channels))
            ])
        
        # 锚框调整矩阵，对小目标的锚框进行特殊调整
        small_object_ratio = adaptive_head_config.get("small_object_anchor_ratio", [0.8, 1.2])
        self.register_buffer('anchor_adjust', torch.ones(len(in_channels), 2))
        # P3层锚框宽高调整系数，针对细小裂缝
        self.anchor_adjust[0, :] = torch.tensor(small_object_ratio)  # 小裂缝倾向于更窄更长
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (list): 输入特征图列表，通常为P3, P4, P5
            
        返回:
            tuple: 检测结果
        """
        # 如果自适应头部被禁用，则退化为普通检测头
        if not enable_adaptive_head:
            outputs = []
            # 仅对常规特征进行预测，不进行自适应调整
            for i, feature in enumerate(x):
                # 特征提取
                feat = self.extractors[i](feature)
                # 预测
                pred = self.heads[i](feat)
                # 重塑输出形状
                bs, _, ny, nx = pred.shape
                pred = pred.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                outputs.append(pred)
            return outputs
        
        # 检查x的格式，可能已经被增强
        if not isinstance(x, list) or (isinstance(x, list) and len(x) != 3):
            # 可能是从MSAE传来的已经增强过的特征
            features = x
        else:
            # 应用多尺度注意力增强
            features = self.msae(x)
        
        outputs = []
        for i, feature in enumerate(features):
            # 特征提取
            feat = self.extractors[i](feature)
            
            # 预测
            pred = self.heads[i](feat)
            
            # 重塑输出形状
            bs, _, ny, nx = pred.shape
            pred = pred.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 如果启用自适应调整
            if self.enable_adaptation:
                # 应用尺度自适应调整系数
                scale_factor = self.scale_adapter[i](feature)
                
                # 根据尺度调整预测结果
                pred[..., 0:2] = pred[..., 0:2] * scale_factor[..., 0].view(bs, 1, 1, 1, 1)  # 中心点位置
                pred[..., 2:4] = pred[..., 2:4] * scale_factor[..., 1].view(bs, 1, 1, 1, 1)  # 宽高
            
            outputs.append(pred)
            
        return outputs 