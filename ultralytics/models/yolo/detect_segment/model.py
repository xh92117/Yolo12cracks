# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ASSETS, LOGGER, RANK, yaml_load

from ultralytics.models.yolo.detect import DetectionModel, DetectionTrainer, DetectionValidator, DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationModel, SegmentationTrainer, SegmentationValidator, SegmentationPredictor


class DetectSegmentModel(Model):
    """
    YOLO模型，同时支持目标检测和实例分割任务。
    
    这个模型继承自基础Model类，充分利用检测和分割功能，同时输出边界框和分割掩码。
    特别适用于裂缝检测场景，能够同时提供裂缝的位置和精确形状。
    """

    def __init__(self, model='yolov12n_detect_segment.pt', task=None, verbose=False):
        """
        初始化DetectSegmentModel。
        
        Args:
            model (str): 模型路径或名称，支持指向.pt文件的路径或者.yaml配置文件
            task (str, optional): 任务类型，自动判断为`detect_segment`
            verbose (bool): 是否打印详细信息
        """
        super().__init__(model=model, task='detect_segment', verbose=verbose)
        
        # 加载配置
        self.ckpt = None
        self.cfg = None
        self.ckpt_path = None
        self.validator = None
        self.predictor = None
        self.overrides = {'task': 'detect_segment'}  # 任务类型覆盖
        
        # 获取模型
        suffix = Path(model).suffix
        if suffix == '.yaml':
            self.cfg = model
            self._new(model, verbose=verbose)
        elif suffix in ('.pt', '.pth'):
            self.ckpt_path = model
            self._load(model, task='detect_segment', verbose=verbose)
        else:
            raise NotImplementedError(f'Model format {suffix} not supported yet.')

    def _new(self, cfg, verbose=True):
        """
        从YAML配置创建一个新的模型
        """
        cfg = yaml_load(cfg)
        if verbose:
            LOGGER.info(f'创建{self.task}模型: {self.model}')
        
        self.model = DetectSegmentationModel(cfg)  # 使用自定义的联合模型类

    def _load(self, weights, task=None, verbose=True):
        """
        从权重文件加载模型
        """
        if verbose:
            LOGGER.info(f'正在加载{weights}...')
        
        # 加载检查点
        self.ckpt = attempt_load_one_weight(weights)
        self.task = task or self.ckpt.get('task') or 'detect_segment'
        
        # 根据检查点构建模型
        if 'ema' in self.ckpt:
            self.model = DetectSegmentationModel(self.ckpt['model'].yaml)
            state_dict = self.ckpt['ema'].float().state_dict()
        else:
            self.model = DetectSegmentationModel(self.ckpt['model'].yaml)
            state_dict = self.ckpt['model'].float().state_dict()
        
        # 加载模型权重
        self.model.load_state_dict(state_dict, strict=True)
        
        if verbose:
            LOGGER.info(f'已加载{self.task}模型: {weights}')
    
    def train(self, **kwargs):
        """
        训练模型，同时训练检测和分割任务
        """
        self.predictor = None
        if 'task' in kwargs:
            kwargs['task'] = 'detect_segment'  # 确保任务类型正确
        
        # 使用自定义的DetectSegmentTrainer
        self.trainer = DetectSegmentTrainer(overrides=self.overrides, **kwargs)
        if RANK in (-1, 0):
            self.model = self.trainer.best_model
            return self.trainer.metrics  # 返回训练指标
        
        return None
        
    def val(self, **kwargs):
        """
        验证检测和分割结果
        """
        self.predictor = None
        
        # 如果没有验证器则创建一个
        if self.validator is None:
            self.validator = DetectSegmentValidator(args=dict(model=self.model, verbose=False), **kwargs)
        else:
            self.validator.args.update(kwargs)
        
        return self.validator.validate()  # 返回验证指标
        
    def predict(self, **kwargs):
        """
        使用模型进行预测，返回检测和分割结果
        """
        self.validator = None
        
        if self.predictor is None:
            self.predictor = DetectSegmentPredictor(overrides=dict(model=self.model, verbose=False), **kwargs)
        else:
            self.predictor.args.update(kwargs)
            
        return self.predictor.predict()  # 返回预测结果 