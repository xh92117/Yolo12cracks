# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import Loss as DetectionLoss
from ultralytics.models.yolo.segment.train import Loss as SegmentationLoss
from ultralytics.nn.modules.block import A2C2f, C3k2
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.nn.modules.attention import MultiScaleAttentionEnhancement, AdaptiveScaleHead
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import FocalLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class DetectSegmentationModel(DetectionModel):
    """YOLOv12模型，同时具有目标检测和实例分割能力
    
    这个模型扩展了标准YOLOv12检测模型，添加了分割头，可同时输出目标检测边界框和实例分割掩码。
    特别适合裂缝检测，能同时获得裂缝位置和精确形状。
    
    新增功能:
    - 多尺度注意力增强(MSAE)，解决标注框尺寸差异过大的问题
    - 自适应检测头，针对不同尺寸的裂缝动态调整检测策略
    """
    
    def __init__(self, cfg='yolov12_detect_segment.yaml', ch=3, nc=None, verbose=True):
        """
        初始化DetectSegmentationModel
        
        Args:
            cfg (str | dict): 模型配置文件路径或配置字典
            ch (int): 输入通道数
            nc (int, optional): 类别数量，如果为None则使用配置文件中的值
            verbose (bool): 是否打印详细信息
        """
        super().__init__(cfg, ch, nc, verbose)  # 继承DetectionModel初始化
        
        # 检查是否有多尺度注意力增强模块
        self.has_msae = False
        if 'msae' in self.yaml:
            self.msae = self._build_msae()
            self.has_msae = True
        
        # 分离头部模块
        is_crack_detection = cfg.endswith('yolov12_crack_detection.yaml')
        
        if is_crack_detection and self.has_msae:
            # 使用自适应检测头
            self.detect_head = self._build_adaptive_head('detect_head')
            self.segment_head = self._build_head('segment_head')
        else:
            # 使用常规检测和分割头
            self.detect_head = self._build_head('detect_head')
            self.segment_head = self._build_head('segment_head')
        
        # 初始化权重
        self._init_weights()
        
        if verbose:
            self._info()

    def _build_msae(self):
        """构建多尺度注意力增强模块"""
        m = nn.Sequential(*(self._build_block(x) for x in self.yaml['msae']))
        return m

    def _build_adaptive_head(self, head_name):
        """构建针对裂缝检测特别优化的自适应检测头"""
        m = nn.Sequential(*(self._build_block(x) for x in self.yaml[head_name]))
        return m

    def _build_head(self, head_name):
        """
        构建检测头或分割头
        
        Args:
            head_name (str): 头部名称，'detect_head'或'segment_head'
            
        Returns:
            nn.Module: 构建的头部模块
        """
        if head_name == 'detect_head':
            m = nn.Sequential(*(self._build_block(x) for x in self.yaml[head_name]))
            if hasattr(m[-1], 'bias') and isinstance(m[-1].bias, torch.Tensor):
                # 使用custom prior初始化检测头
                m[-1].bias.data[:] = 1.0  # 初始化obj confidence
            return m
        elif head_name == 'segment_head':
            m = nn.Sequential(*(self._build_block(x) for x in self.yaml[head_name]))
            return m
        else:
            raise ValueError(f"未知的头部类型: {head_name}")

    def _build_block(self, layer_cfg):
        """
        构建网络块
        
        Args:
            layer_cfg (list): 层配置
            
        Returns:
            nn.Module: 构建的网络块
        """
        from_layer, num_modules, module_name, args = layer_cfg
        
        # 特殊模块处理
        special_modules = ['Detect', 'Segment', 'MultiScaleAttentionEnhancement', 'AdaptiveScaleHead']
        
        if isinstance(from_layer, list) and module_name in special_modules:
            # 获取输入特征层
            from_layers = [self.save[x] if x < 0 else x for x in from_layer]
            
            if module_name == 'Detect':
                return Detect(nc=self.nc, ch=self.ch, args=self.args)
            elif module_name == 'Segment':
                return Segment(nc=self.nc, nm=args[1], npr=args[2], ch=self.ch, args=self.args)
            elif module_name == 'MultiScaleAttentionEnhancement':
                # 处理多尺度注意力增强模块
                from ultralytics.nn.modules.attention import MultiScaleAttentionEnhancement
                return MultiScaleAttentionEnhancement(args[0])
            elif module_name == 'AdaptiveScaleHead':
                # 自适应尺度检测头
                from ultralytics.nn.modules.attention import AdaptiveScaleHead
                return AdaptiveScaleHead(args[0], nc=args[1])
        else:
            # 使用原生的构建块实现
            return super()._build_block(layer_cfg)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            tuple: 检测和分割结果
        """
        # 获取骨干网络和颈部特征
        y = []  # 存储中间特征图
        for m in self.model:
            if m.f != -1:  # 如果不是使用上一层作为输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从早期层获取特征
            x = m(x)  # 前向传播
            y.append(x if m.i in self.save else None)  # 保存输出
        
        # 准备检测和分割的特征
        p3, p4, p5 = y[14], y[17], y[20]  # 提取P3, P4, P5特征
        features = [p3, p4, p5]
        
        # 应用多尺度注意力增强（如果有）
        if self.has_msae:
            enhanced_features = self.msae([features])
            # 分别获取检测和分割结果（使用增强后的特征）
            detect_out = self.detect_head(enhanced_features)
            segment_out = self.segment_head(enhanced_features)
        else:
            # 直接使用原始特征
            detect_out = self.detect_head(features)
            segment_out = self.segment_head(features)
        
        return detect_out, segment_out


class DetectSegmentLoss(nn.Module):
    """
    检测分割联合损失函数
    
    该类结合了目标检测损失和实例分割损失，用于联合训练。
    可以通过超参数调整两个任务的权重。
    
    支持针对不同尺寸目标的自适应损失调整。
    """
    
    def __init__(self, model):
        """
        初始化联合损失函数
        
        Args:
            model (DetectSegmentationModel): 目标检测分割模型
        """
        super().__init__()
        
        self.det_loss = DetectionLoss(model)
        self.seg_loss = SegmentationLoss(model)
        
        # 获取权重配置
        loss_weights = getattr(model.args, 'loss_weights', None)
        if loss_weights is None:
            # 默认权重
            self.box_weight = 1.0  # 检测权重
            self.mask_weight = 1.0  # 分割权重
        else:
            # 从配置加载权重
            self.box_weight = loss_weights.get('box', 1.0)
            self.mask_weight = loss_weights.get('mask', 1.0)
            
        # 尺寸自适应损失增强
        self.size_aware_loss = model.yaml.get('size_aware_loss', True)
    
    def __call__(self, preds, batch):
        """
        计算联合损失
        
        Args:
            preds (tuple): 模型预测，(detection_results, segmentation_results)
            batch (dict): 包含图像和标注的字典
            
        Returns:
            tuple: (总损失, 各组件损失字典)
        """
        # 分离检测和分割预测
        det_preds, seg_preds = preds
        
        # 计算检测损失
        det_loss, det_components = self.det_loss(det_preds, batch)
        
        # 计算分割损失
        seg_loss, seg_components = self.seg_loss(seg_preds, batch)
        
        # 如果启用尺寸感知损失
        if self.size_aware_loss and 'bboxes' in batch:
            # 计算目标框的面积
            boxes = batch['bboxes']
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # 根据面积划分尺寸类别
            small_box = areas < 32*32  # 小目标 (小于32x32)
            medium_box = (areas >= 32*32) & (areas < 96*96)  # 中等目标
            large_box = areas >= 96*96  # 大目标
            
            # 针对小目标增加权重（解决标注框尺寸差异过大问题）
            small_weight = 1.5  # 小目标权重增强
            medium_weight = 1.0  # 中等目标标准权重
            large_weight = 0.8  # 大目标权重略微降低
            
            # 计算每个样本的权重
            weights = torch.ones_like(areas)
            weights[small_box] = small_weight
            weights[medium_box] = medium_weight
            weights[large_box] = large_weight
            
            # 应用权重到检测损失
            # 这只是一个概念示范，实际上需要对损失函数进行修改
            # 这里假设det_components中有一个item_loss表示每个目标的损失
            if 'item_loss' in det_components:
                weighted_loss = det_components['item_loss'] * weights.to(det_components['item_loss'].device)
                det_loss = weighted_loss.sum() / max(1, weights.sum())
        
        # 计算加权总损失
        loss = self.box_weight * det_loss + self.mask_weight * seg_loss
        
        # 合并损失组件
        components = {**det_components, **{f'seg_{k}': v for k, v in seg_components.items()}}
        components['box_weight'] = self.box_weight  # 添加权重信息
        components['mask_weight'] = self.mask_weight
        
        return loss, components 