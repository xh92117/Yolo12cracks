# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel, attempt_load_one_weight
from ultralytics.utils import ASSETS, LOGGER, RANK, yaml_load
from ultralytics.engine.model import Model

from ultralytics.models.yolo.detect import DetectionModel, DetectionTrainer, DetectionValidator, DetectionPredictor
from ultralytics.nn.modules.attention import ScaleAwareAttention, MultiScaleAttentionEnhancement, AdaptiveScaleHead
from ultralytics.nn.modules.attentions.config import enable_msae, enable_adaptive_head, enable_size_aware_loss

# 全局任务映射字典
TASK_MAP = {
    'detect_cracks': ('detect', 'CrackDetectionModel')
}

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
            ch (int): 输入通道数，默认为3（RGB图像）
            nc (int, optional): 类别数量，默认为配置文件中的值
            verbose (bool): 是否打印详细信息
        """
        # 调用父类初始化
        super().__init__(cfg, ch, nc, verbose)
        
        # 初始化多尺度注意力增强模块
        self._init_attention_modules(verbose)
        
        # 打印模型信息
        if verbose:
            self._info()
    
    def _init_attention_modules(self, verbose=True):
        """
        初始化并配置注意力增强模块
        
        Args:
            verbose (bool): 是否打印详细日志
        """
        # 检查是否有多尺度注意力增强模块(MSAE)
        if enable_msae and hasattr(self, 'yaml') and 'msae' in self.yaml:
            self.has_msae = True
            self.msae = self._build_msae()
            if verbose:
                LOGGER.info("已启用多尺度注意力增强模块(MSAE)")
        else:
            self.has_msae = False
            
        # 检查是否有自适应尺度检测头
        if enable_adaptive_head and hasattr(self, 'yaml') and 'adaptive_head' in self.yaml:
            self.has_adaptive_head = True
            if verbose:
                LOGGER.info("已启用自适应尺度检测头")
        else:
            self.has_adaptive_head = False
    
    def _build_msae(self):
        """
        构建多尺度注意力增强模块
        
        Returns:
            nn.Module: 多尺度注意力增强模块
        """
        if 'msae' in self.yaml:
            msae_config = self.yaml['msae']
            # 构建模块序列
            m = nn.Sequential(*(self._build_attention_block(x) for x in msae_config))
            return m
        return None
            
    def _build_attention_block(self, layer_cfg):
        """
        构建注意力网络块
        
        Args:
            layer_cfg (list): 层配置 [from_layer, repeat, module, args]
            
        Returns:
            nn.Module: 构建的网络块
        """
        from_layer, num_modules, module_name, args = layer_cfg
        
        # 特殊模块处理
        if module_name == 'MultiScaleAttentionEnhancement':
            # 获取输入特征层通道数
            if isinstance(from_layer, list):
                channels = [self.get_layer_channels(x) for x in from_layer]
                return MultiScaleAttentionEnhancement(channels)
                
        elif module_name == 'AdaptiveScaleHead':
            # 获取输入特征层通道数
            if isinstance(from_layer, list):
                channels = [self.get_layer_channels(x) for x in from_layer]
                return AdaptiveScaleHead(channels, nc=args[0] if args else self.nc)
                
        elif module_name == 'ScaleAwareAttention':
            # 单一尺度注意力
            channel = self.get_layer_channels(from_layer)
            return ScaleAwareAttention(channel, reduction=args[0] if args else 16)
        
        # 使用标准方法构建其他块
        return self._build_standard_block(layer_cfg)
        
    def _build_block(self, layer_cfg):
        """
        构建网络块，重写父类方法
        
        Args:
            layer_cfg (list): 层配置 [from_layer, repeat, module, args]
            
        Returns:
            nn.Module: 构建的网络块
        """
        from_layer, num_modules, module_name, args = layer_cfg
        
        # 检查是否是注意力模块
        if module_name in ['MultiScaleAttentionEnhancement', 'AdaptiveScaleHead', 'ScaleAwareAttention']:
            return self._build_attention_block(layer_cfg)
        
        # 使用父类方法构建普通模块
        return super()._build_block(layer_cfg)
        
    def _build_standard_block(self, layer_cfg):
        """
        构建标准网络块，调用父类方法
        
        Args:
            layer_cfg (list): 层配置
            
        Returns:
            nn.Module: 构建的网络块
        """
        return super()._build_block(layer_cfg)
        
    def get_layer_channels(self, layer_idx):
        """
        获取特定层的通道数
        
        Args:
            layer_idx (int): 层索引
            
        Returns:
            int: 通道数
        """
        if hasattr(self, 'save') and layer_idx < len(self.save):
            # 从保存的特征图获取通道数
            return self.save[layer_idx].shape[1] if layer_idx >= 0 else self.save[layer_idx].shape[1]
        elif hasattr(self, 'model') and layer_idx < len(self.model):
            # 从模型层获取输出通道数
            return getattr(self.model[layer_idx], 'out_channels', 0)
        else:
            # 默认通道数
            return 256
    
    def _build_head(self):
        """
        构建裂缝检测头，支持自适应尺度
        
        Returns:
            list: 检测头层列表
        """
        # 首先使用父类方法构建标准检测头
        head = super()._build_head()
        
        # 如果启用了自适应头，修改检测头
        if self.has_adaptive_head and hasattr(self, 'yaml') and 'adaptive_head' in self.yaml:
            # 读取自适应头配置
            adaptive_config = self.yaml['adaptive_head']
            
            # 根据配置修改检测头的最后一层
            if isinstance(adaptive_config, dict) and 'type' in adaptive_config:
                head_type = adaptive_config['type']
                if head_type == 'multi_scale':
                    # 替换最后一层检测头为多尺度检测头
                    channels = head[-2].out_channels
                    head[-1] = AdaptiveScaleHead([channels], nc=self.nc)
        
        return head
    
    def forward(self, x):
        """
        模型前向传播，支持多尺度注意力增强
        
        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: 模型输出
        """
        # 使用父类的前向传播获取基本特征
        y = super().forward(x)
        
        # 如果启用了多尺度注意力增强，应用MSAE
        if self.has_msae and hasattr(self, 'msae') and self.msae is not None:
            # 增强特征图
            if isinstance(y, list):
                # 对每个输出特征图应用注意力
                enhanced_y = [self.msae(feature) for feature in y]
                return enhanced_y
            else:
                # 对单一输出应用注意力
                return self.msae(y)
                
        return y
        
    def _info(self, verbose=False):
        """打印模型详细信息"""
        # 调用父类方法打印基本信息
        super()._info(verbose)
        
        if not verbose:
            return
            
        # 打印裂缝检测特定信息
        if self.has_msae:
            LOGGER.info("裂缝检测增强: 多尺度注意力增强模块")
        if self.has_adaptive_head:
            LOGGER.info("裂缝检测增强: 自适应尺度检测头")
        if enable_size_aware_loss:
            LOGGER.info("裂缝检测增强: 尺度感知损失函数")
            
    def get_validator(self):
        """返回对应的验证器类"""
        # 按需导入，避免循环导入问题
        return DetectionValidator


class CrackDetectionModelWrapper(Model):
    """
    裂缝检测模型包装类，提供统一接口
    """
    
    def __init__(self, model='yolov12_cracks.pt', task=None, verbose=False):
        """
        初始化裂缝检测模型
        
        Args:
            model (str): 模型路径或名称
            task (str): 任务类型，自动设置为'detect_cracks'
            verbose (bool): 是否打印详细信息
        """
        # 首先调用父类初始化
        task = task or 'detect'  # 确保有任务类型
        super().__init__(model=model, task=task, verbose=verbose)
        
        # 设置任务为裂缝检测
        self.task = 'detect_cracks'
        
        # 重新加载模型（如果需要）
        if not isinstance(self.model, CrackDetectionModel):
            self._setup_model()
        
    def _setup_model(self):
        """设置模型"""
        # 检查是否是配置文件路径
        if Path(self.model).suffix == '.yaml':
            self._new(self.model, verbose=self.verbose)
        else:
            self._load(self.model, verbose=self.verbose)
    
    def _new(self, cfg, verbose=True):
        """从配置创建新模型"""
        self.model = CrackDetectionModel(cfg=cfg, verbose=verbose)
        
    def _load(self, weights, verbose=True):
        """从权重加载模型"""
        if verbose:
            LOGGER.info(f'正在加载{weights}...')
        
        # 加载检查点
        self.ckpt = attempt_load_one_weight(weights)
        
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
        """
        训练模型
        
        Args:
            **kwargs: 训练参数
            
        Returns:
            dict: 训练指标
        """
        # 重置预测器
        self.predictor = None
        
        # 确保任务类型为裂缝检测
        kwargs['task'] = 'detect_cracks'
        
        # 启用尺度感知损失
        if enable_size_aware_loss and 'size_aware_loss' not in kwargs:
            kwargs['size_aware_loss'] = True
        
        # 创建训练器
        self.trainer = DetectionTrainer(overrides={'model': self.model}, **kwargs)
        
        # 主进程获取最佳模型
        if RANK in (-1, 0):
            self.model = self.trainer.best_model
            return self.trainer.metrics
        
        return None
        
    def val(self, **kwargs):
        """
        验证模型
        
        Args:
            **kwargs: 验证参数
            
        Returns:
            dict: 验证指标
        """
        # 重置预测器
        self.predictor = None
        
        # 创建或更新验证器
        if self.validator is None:
            self.validator = DetectionValidator(args={'model': self.model, 'verbose': self.verbose}, **kwargs)
        else:
            self.validator.args.update(kwargs)
        
        # 执行验证
        return self.validator.validate()
        
    def predict(self, **kwargs):
        """
        使用模型进行预测
        
        Args:
            **kwargs: 预测参数
            
        Returns:
            list: 预测结果
        """
        # 重置验证器
        self.validator = None
        
        # 创建或更新预测器
        if self.predictor is None:
            self.predictor = DetectionPredictor(overrides={'model': self.model, 'verbose': self.verbose}, **kwargs)
        else:
            self.predictor.args.update(kwargs)
            
        # 执行预测
        return self.predictor.predict()


# 将任务注册到YOLO可用任务中
def register_crack_detection():
    """
    注册裂缝检测任务到YOLO任务系统
    """
    from ultralytics.models.yolo.model import YOLO
    
    if not hasattr(YOLO, 'detect_cracks'):
        def detect_cracks_method(self, **kwargs):
            """
            特定于裂缝检测的方法
            
            Args:
                **kwargs: 参数
                
            Returns:
                object: 根据操作模式返回不同结果
            """
            kwargs['task'] = 'detect_cracks'
            return self._smart_load('detect_cracks', **kwargs)
        
        # 动态添加方法
        setattr(YOLO, 'detect_cracks', detect_cracks_method)
        
        # 将裂缝检测任务注册为检测任务的特例
        if 'detect_cracks' not in TASK_MAP:
            TASK_MAP['detect_cracks'] = ('detect', 'CrackDetectionModelWrapper')
            
        # 添加到文档
        detect_cracks_method.__doc__ = """
        使用YOLO模型进行裂缝检测。
        
        该方法专门针对裂缝检测任务，增加了多尺度注意力机制，更适合检测不同尺寸的裂缝。
        支持以下增强功能：
        1. 多尺度注意力增强(MSAE)：解决裂缝粗细差异大的问题
        2. 自适应检测头：对不同尺寸裂缝动态调整检测策略
        3. 尺度感知损失：针对小裂缝检测优化训练过程
        
        Returns:
            根据操作模式返回不同结果：
            - train: 训练指标字典
            - val: 验证指标字典
            - predict: 预测结果列表
        """ 