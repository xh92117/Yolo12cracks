# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLO检测模型模块

该模块提供了YOLO检测模型的实现，用于目标检测任务。
主要包含DetectionModel类及其辅助函数。
"""

from pathlib import Path
import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f, DFL, Proto
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.nn.tasks import BaseModel, attempt_load_one_weight
from ultralytics.utils import ASSETS, LOGGER, RANK, yaml_load

class DetectionModel(BaseModel):
    """YOLO检测模型"""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        """
        初始化YOLO检测模型
        
        Args:
            cfg (str | dict): 模型配置文件路径或配置字典
            ch (int): 输入通道数，默认为3（RGB图像）
            nc (int, optional): 类别数量。如果为None，则使用配置文件中指定的值
            verbose (bool): 是否显示详细日志
        """
        super().__init__(cfg, ch, nc, verbose)  # 初始化父类
        self.inplace = self.yaml.get('inplace', True)  # 设置inplace操作标志

    def _build_network(self):
        """构建YOLO检测网络的主干和检测头"""
        self.head = self._build_head()  # 构建检测头
        self.model = self._build_backbone() + self.head  # 组合主干和检测头形成完整模型

    def _build_backbone(self):
        """构建YOLO主干网络，返回层列表"""
        return self._build_backbone_common()  # 使用基类中的通用主干构建方法

    def _build_head(self):
        """构建YOLO检测头"""
        y = []  # 输出通道列表
        for m in self.model:
            if m.f != -1:  # 如果不是来自上一层
                # 获取前一层的输出通道数
                y.append(m.f < 0 and y[m.f] or self.output_shapes[m.f] * (m.f > 0))
                
        # 计算DFL的最小通道数
        c2 = max((16, y[-1] // self.stride[-1]))
        
        # 返回检测头网络结构：两个卷积层和一个检测层
        return [
            Conv(y[-1], c2, 3),  # 第一个3x3卷积层
            Conv(c2, c2, 3),     # 第二个3x3卷积层
            Detect(nc=self.nc, ch=c2)  # 检测层
        ]

    def forward(self, x):
        """
        模型前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, channels, height, width)
            
        Returns:
            (torch.Tensor): 模型输出，形状取决于检测头的配置
        """
        return self._forward_once(x)  # 使用基类的标准前向传播方法
        
    def _info(self, verbose=False):
        """打印模型信息"""
        super()._info(verbose)  # 调用父类方法打印基本信息
        
        if verbose and hasattr(self, 'head') and len(self.head) > 0 and hasattr(self.head[-1], 'dfl'):
            # 附加信息
            if self.head[-1].dfl:
                self.info("YOLO头使用DFL结构. 参见 https://arxiv.org/abs/2211.00481")
            self.info(f"锚点数量: {self.head[-1].na}")
            self.info(f"类别数: {self.head[-1].nc}")
            self.info(f"缩放系数: {self.head[-1].scale}")
            
    def get_validator(self):
        """返回对应的验证器类"""
        # 按需导入，避免循环导入问题
        from ultralytics.models.yolo.detect.val import DetectionValidator
        return DetectionValidator 