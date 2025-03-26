# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import torch
import torch.nn as nn

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ASSETS, LOGGER, RANK, yaml_load

from ultralytics.models.yolo.detect import DetectionModel, DetectionTrainer, DetectionValidator, DetectionPredictor
from ultralytics.nn.modules.attention import ScaleAwareAttention, MultiScaleAttentionEnhancement, AdaptiveScaleHead
from ultralytics.nn.modules.attentions.config import enable_msae, enable_adaptive_head, enable_size_aware_loss


class CrackDetectionModel(DetectionModel):
    """
    裂缝检测模型，继承自DetectionModel，增加了多尺度注意力机制
    
    特点:
    1. 多尺度注意力增强(MSAE)，解决裂缝粗细差异大的问题
    2. 自适应检测头，对不同尺寸裂缝动态调整检测策略
    3. 尺度感知模块，针对小裂缝特征增强
    """
    
    def __init__(self, cfg='yolov12_cracks.yaml', ch=3, nc=None, verbose=True):
        """
        初始化裂缝检测模型
        
        Args:
            cfg (str | dict): 模型配置文件路径或配置字典
            ch (int): 输入通道数，默认为3
            nc (int, optional): 类别数量，默认为配置文件中的值
            verbose (bool): 是否打印详细信息
        """
        super().__init__(cfg, ch, nc, verbose)
        
        # 检查是否有多尺度注意力增强模块(MSAE)
        if hasattr(self.model[-1], 'msae') and enable_msae:
            self.has_msae = True
            if verbose:
                LOGGER.info("已启用多尺度注意力增强模块(MSAE)")
        else:
            self.has_msae = False
            
        # 检查是否有自适应尺度检测头
        if hasattr(self.model[-1], 'enable_adaptation') and enable_adaptive_head:
            self.has_adaptive_head = True
            if verbose:
                LOGGER.info("已启用自适应尺度检测头")
        else:
            self.has_adaptive_head = False
            
        # 初始化权重
        self._init_weights()
        
        if verbose:
            self._info()
    
    def _build_msae(self):
        """构建多尺度注意力增强模块"""
        if 'msae' in self.yaml:
            m = nn.Sequential(*(self._build_block(x) for x in self.yaml['msae']))
            return m
        return None
            
    def _build_block(self, layer_cfg):
        """
        构建网络块
        
        Args:
            layer_cfg (list): 层配置 [from_layer, repeat, module, args]
            
        Returns:
            nn.Module: 构建的网络块
        """
        from_layer, num_modules, module_name, args = layer_cfg
        
        # 特殊模块处理
        if module_name in ['MultiScaleAttentionEnhancement', 'AdaptiveScaleHead']:
            # 获取输入特征层
            if isinstance(from_layer, list):
                channels = [self.save[x].shape[1] if x < 0 else self.save[x].shape[1] for x in from_layer]
                
                if module_name == 'MultiScaleAttentionEnhancement':
                    return MultiScaleAttentionEnhancement(channels)
                elif module_name == 'AdaptiveScaleHead':
                    return AdaptiveScaleHead(channels, nc=args[0])
        
        # 使用基类的实现处理其他块
        return super()._build_block(layer_cfg)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            tuple: 检测结果
        """
        # 调用DetectionModel的forward方法
        return super().forward(x)


class CrackDetectionModelWrapper(Model):
    """
    裂缝检测模型包装类，提供更简洁的接口
    """
    
    def __init__(self, model='yolov12_cracks.pt', task=None, verbose=False):
        """
        初始化裂缝检测模型
        
        Args:
            model (str): 模型路径或名称
            task (str): 任务类型，自动设置为'detect'
            verbose (bool): 是否打印详细信息
        """
        # 检查是否是配置文件路径
        if Path(model).suffix == '.yaml':
            super().__init__(model=model, task='detect', verbose=verbose)
            self._new(model, verbose=verbose)
        else:
            super().__init__(model=model, task='detect', verbose=verbose)
            # 加载权重
            self._load(model, verbose=verbose)
    
    def _new(self, cfg, verbose=True):
        """从配置创建新模型"""
        self.model = CrackDetectionModel(cfg, verbose=verbose)
        
    def _load(self, weights, task=None, verbose=True):
        """从权重加载模型"""
        if verbose:
            LOGGER.info(f'正在加载{weights}...')
        
        # 加载检查点
        self.ckpt = attempt_load_one_weight(weights)
        self.task = task or self.ckpt.get('task') or 'detect'
        
        # 构建模型
        if self.ckpt.get('model', None) is None:
            raise ValueError("模型权重无效，缺少模型配置")
            
        if 'ema' in self.ckpt:
            self.model = CrackDetectionModel(self.ckpt['model'].yaml)
            state_dict = self.ckpt['ema'].float().state_dict()
        else:
            self.model = CrackDetectionModel(self.ckpt['model'].yaml)
            state_dict = self.ckpt['model'].float().state_dict()
        
        # 加载模型权重
        self.model.load_state_dict(state_dict, strict=True)
        
        if verbose:
            LOGGER.info(f'已加载裂缝检测模型: {weights}')
    
    def train(self, **kwargs):
        """训练模型"""
        self.predictor = None
        if 'task' in kwargs:
            kwargs['task'] = 'detect'  # 确保任务类型为检测
        
        # 启用尺度感知损失
        if enable_size_aware_loss:
            if 'size_aware_loss' not in kwargs:
                kwargs['size_aware_loss'] = True
        
        # 使用标准的DetectionTrainer
        self.trainer = DetectionTrainer(overrides=self.overrides, **kwargs)
        if RANK in (-1, 0):
            self.model = self.trainer.best_model
            return self.trainer.metrics
        
        return None
        
    def val(self, **kwargs):
        """验证模型"""
        self.predictor = None
        
        if self.validator is None:
            self.validator = DetectionValidator(args=dict(model=self.model, verbose=False), **kwargs)
        else:
            self.validator.args.update(kwargs)
        
        return self.validator.validate()
        
    def predict(self, **kwargs):
        """使用模型进行预测"""
        self.validator = None
        
        if self.predictor is None:
            self.predictor = DetectionPredictor(overrides=dict(model=self.model, verbose=False), **kwargs)
        else:
            self.predictor.args.update(kwargs)
            
        return self.predictor.predict()


# 将任务注册到YOLOv12可用任务中
def register_crack_detection():
    """
    注册裂缝检测任务到YOLOv12任务系统
    """
    from ultralytics.models.yolo.model import YOLO
    
    if not hasattr(YOLO, 'detect_cracks'):
        def detect_cracks_method(self, **kwargs):
            """特定于裂缝检测的方法"""
            kwargs['task'] = 'detect_cracks'
            return self._smart_load('detect_cracks', **kwargs)
        
        # 动态添加方法
        setattr(YOLO, 'detect_cracks', detect_cracks_method)
        
        # 将裂缝检测任务注册为检测任务的特例
        from ultralytics.engine.model import task_map
        if 'detect_cracks' not in task_map:
            task_map['detect_cracks'] = ('detect', 'CrackDetectionModelWrapper')
            
        # 添加到文档
        detect_cracks_method.__doc__ = """
        使用YOLO模型进行裂缝检测。
        该方法继承自detect，但增加了多尺度注意力机制，更适合检测不同尺寸的裂缝。
        
        Returns:
            (List[ultralytics.engine.results.Results]): 预测结果列表
        """ 