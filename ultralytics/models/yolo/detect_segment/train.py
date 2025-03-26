# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import RANK, colorstr

from .network import DetectSegmentationModel, DetectSegmentLoss


class DetectSegmentTrainer(BaseTrainer):
    """
    联合训练类，结合目标检测和实例分割任务
    
    继承自BaseTrainer，专门处理同时进行目标检测和实例分割的训练。
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        初始化联合训练器
        
        Args:
            cfg (str, optional): 配置文件路径或文件名
            overrides (dict, optional): 覆盖默认配置的参数
            _callbacks (list, optional): 回调函数列表
        """
        # 设置默认任务为detect_segment
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect_segment"
        
        # 初始化基类
        super().__init__(cfg, overrides, _callbacks)
        
        # 确保数据集配置正确
        if hasattr(self.args, "data"):
            self.data = self.args.data
        self.validator = None
        
        # 自定义数据加载
        self.train_loader = self.get_dataloader(self.trainset, "train", rank=RANK)
        
        # 创建核心模型
        self.model = self.get_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_lr_scheduler()
        self.loss = DetectSegmentLoss(self.model)
        
        self.args.model = self.model
        self.args.lr0 = self.args.lr0 * self.batch_size / 16

        # 加载预训练权重
        if not self.resume:
            self.load_pretrained_weights()
        
        self.best_fitness = 0.0
        self.best_epoch = -1
        
    def get_model(self):
        """
        加载或创建模型
        
        Returns:
            DetectSegmentationModel: 联合检测分割模型
        """
        model = DetectSegmentationModel(cfg=self.args.model or "yolov12_detect_segment.yaml", nc=self.args.nc)
        
        return model
    
    def preprocess_batch(self, batch):
        """
        预处理批次数据
        
        Args:
            batch (dict): 包含图像和标签的批次数据
            
        Returns:
            dict: 预处理后的批次数据
        """
        # 确保每个样本同时有边界框和实例掩码
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["cls"] = batch["cls"].to(self.device)
        batch["bboxes"] = batch["bboxes"].to(self.device)
        
        # 处理分割掩码
        if "masks" in batch:
            batch["masks"] = batch["masks"].to(self.device).float()
        else:
            # 如果没有掩码，使用边界框生成粗略掩码（训练阶段）
            if self.args.overlap_mask and "batch_idx" in batch:
                h, w = batch["img"].shape[2:]
                masks = torch.zeros((batch["bboxes"].shape[0], h, w), device=self.device)
                for i, box in enumerate(batch["bboxes"]):
                    x1, y1, x2, y2 = map(int, box[:4])
                    masks[i, y1:y2, x1:x2] = 1
                batch["masks"] = masks
        
        return batch
    
    def _do_train_epoch(self, epoch):
        """
        执行一个训练epoch
        
        Args:
            epoch (int): 当前epoch编号
            
        Returns:
            dict: 损失和指标字典
        """
        # 设置训练模式
        self.model.train()
        self.model.requires_grad_(True)
        
        # 训练循环
        pbar = self.progress_bar(self.train_loader, description=self.epoch_progress_string.format(epoch))
        self.lr_scheduler.last_epoch = epoch - 1  # do not move
        
        for batch_idx, batch in enumerate(pbar):
            # 预处理
            ni = batch_idx + self.epoch_size * epoch
            batch = self.preprocess_batch(batch)
            
            # 前向传播和损失计算
            self.optimizer.zero_grad()
            preds = self.model(batch["img"])
            loss, loss_items = self.loss(preds, batch)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新学习率
            self.lr_scheduler.step_after_batch(ni)
            
            # 记录损失
            if RANK == 0:
                self.loss_items.update({k: v for k, v in loss_items.items()})
                self.log_train_batch(batch_idx, ni, {**{"lr": self.optimizer.param_groups[0]["lr"]}, **loss_items})
                
                # 更新进度条描述
                memory = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"  # nvidia only
                base_str = f"Epoch {epoch}/{self.epochs - 1 if self.epochs > 1 else 1}"
                pbar.set_description(f"{base_str}: {memory}")
                
        return {**{"lr": self.optimizer.param_groups[0]["lr"]}, **self.loss_items.result()}
    
    def get_validator(self):
        """
        获取验证器
        
        Returns:
            DetectSegmentValidator: 联合检测分割验证器
        """
        if self.validator is None:
            self.validator = DetectSegmentValidator(
                dataloader=self.get_dataloader(self.testset, "val", rank=RANK),
                save_dir=self.save_dir,
                args=copy(self.args),
            )
        return self.validator
    
    def load_pretrained_weights(self):
        """
        加载预训练权重
        
        可以加载检测和分割预训练权重并合并
        """
        if self.args.pretrained.lower() == "false":
            return
        
        # 检查是否有联合预训练权重
        try:
            # 尝试直接加载联合模型权重
            weights = attempt_load_one_weight(self.args.pretrained)
            ckpt = weights.float().state_dict()
            self.model.load_state_dict(ckpt, strict=False)
            print(f"成功加载预训练权重: {colorstr('bold', 'green', self.args.pretrained)}")
        except Exception:
            # 尝试单独加载检测和分割权重
            print(f"无法直接加载预训练权重，尝试分别加载检测和分割权重...")
            
            # 加载检测权重
            try:
                detect_weights = attempt_load_one_weight("yolov12n.pt")
                detect_ckpt = detect_weights.float().state_dict()
                
                # 仅加载骨干网络和检测头部分
                model_state_dict = self.model.state_dict()
                for k, v in detect_ckpt.items():
                    if k in model_state_dict and (not k.startswith("segment_head")):
                        model_state_dict[k] = v
                
                self.model.load_state_dict(model_state_dict, strict=False)
                print(f"成功加载检测预训练权重")
            except Exception as e:
                print(f"加载检测权重失败: {e}")
            
            # 加载分割权重
            try:
                segment_weights = attempt_load_one_weight("yolov8n-seg.pt")
                segment_ckpt = segment_weights.float().state_dict()
                
                # 对分割头部分重命名键并加载
                model_state_dict = self.model.state_dict()
                for k, v in segment_ckpt.items():
                    if k.startswith("model.28"):  # 分割头
                        new_k = k.replace("model.28", "segment_head.0")
                        if new_k in model_state_dict:
                            model_state_dict[new_k] = v
                
                self.model.load_state_dict(model_state_dict, strict=False)
                print(f"成功加载分割预训练权重")
            except Exception as e:
                print(f"加载分割权重失败: {e}")


class DetectSegmentValidator:
    """
    联合检测分割验证器
    
    用于验证检测和分割任务的性能，计算综合指标。
    """
    
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """
        初始化验证器
        
        Args:
            dataloader: 验证数据加载器
            save_dir: 保存结果的目录
            args: 参数
            _callbacks: 回调函数列表
        """
        self.args = args or {}
        
        # 创建检测和分割验证器
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        
        self.det_validator = DetectionValidator(dataloader, save_dir, args, _callbacks)
        self.seg_validator = SegmentationValidator(dataloader, save_dir, args, _callbacks)
        
    def __call__(self, trainer=None, model=None):
        """验证模型性能"""
        return self.validate(trainer, model)
    
    def validate(self, trainer=None, model=None):
        """
        执行验证
        
        Args:
            trainer: 训练器实例
            model: 模型实例
            
        Returns:
            dict: 验证指标
        """
        model = model or (trainer.model if trainer else None)
        metrics = {}
        
        # 执行检测验证
        det_metrics = self.det_validator(trainer, model)
        
        # 执行分割验证
        seg_metrics = self.seg_validator(trainer, model)
        
        # 合并指标
        for k, v in det_metrics.items():
            metrics[f"detect_{k}"] = v
        
        for k, v in seg_metrics.items():
            metrics[f"segment_{k}"] = v
        
        # 计算组合指标（可以根据需要调整权重）
        box_map = det_metrics["metrics/mAP50-95(B)"]
        mask_map = seg_metrics["metrics/mAP50-95(M)"]
        metrics["fitness"] = 0.6 * box_map + 0.4 * mask_map  # 组合指标
        
        return metrics


class DetectSegmentPredictor:
    """
    联合检测分割预测器
    
    用于生成检测和分割结果。
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        初始化预测器
        
        Args:
            cfg: 配置文件路径或文件名
            overrides: 覆盖默认配置的参数
            _callbacks: 回调函数列表
        """
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor
        
        # 创建检测和分割预测器
        overrides = overrides or {}
        self.det_predictor = DetectionPredictor(cfg, overrides, _callbacks)
        self.seg_predictor = SegmentationPredictor(cfg, overrides, _callbacks)
        
        # 存储参数
        self.overrides = overrides
        self.args = self.det_predictor.args
    
    def predict(self, source=None, model=None, verbose=False, stream=False):
        """
        生成预测结果
        
        Args:
            source: 输入源
            model: 模型实例
            verbose: 是否显示详细信息
            stream: 是否流式处理
            
        Returns:
            list: 预测结果
        """
        # 执行检测预测
        det_results = self.det_predictor.predict(source, model, verbose, stream)
        
        # 执行分割预测
        seg_results = self.seg_predictor.predict(source, model, verbose, stream)
        
        # 合并结果
        results = []
        for det_res, seg_res in zip(det_results, seg_results):
            # 创建合并结果
            combined_res = copy(det_res)
            combined_res.masks = seg_res.masks  # 添加分割掩码
            results.append(combined_res)
        
        return results 