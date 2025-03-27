#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
裂缝检测模型训练脚本
该脚本使用YOLOv12结合多尺度注意力机制训练裂缝检测模型
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import random
import numpy as np

# 添加项目根目录到PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 标准依赖项
import torch
from tqdm import tqdm

# 延迟导入YOLO类避免循环导入
def import_yolo():
    from ultralytics import YOLO
    return YOLO

# 自定义依赖项
from ultralytics.nn.modules.attentions.config import set_config
from ultralytics.utils.files import increment_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12裂缝检测模型训练')
    parser.add_argument('--data', type=str, default='cracks.yaml', help='数据集配置文件')
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/v12/yolov12_cracks.yaml', help='模型配置文件')
    parser.add_argument('--weights', type=str, default='', help='预训练权重路径(为空则从头训练)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='训练图像尺寸 (像素)')
    parser.add_argument('--device', type=str, default='', help='cuda设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作线程数')
    parser.add_argument('--project', type=str, default='runs/train', help='保存结果的项目名称')
    parser.add_argument('--name', type=str, default='yolov12_cracks', help='保存结果的运行名称')
    parser.add_argument('--exist-ok', action='store_true', help='是否使用已存在的运行目录，否则递增')
    parser.add_argument('--msae', type=int, default=1, help='是否启用多尺度注意力增强 (0=禁用，1=启用)')
    parser.add_argument('--adaptive', type=int, default=1, help='是否启用自适应检测头 (0=禁用，1=启用)')
    parser.add_argument('--size-aware-loss', type=int, default=1, help='是否启用尺寸感知损失 (0=禁用，1=启用)')
    
    # 学习率和优化器设置
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率 = lr0 * lrf')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD动量/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='优化器权重衰减')
    
    # 增强设置
    parser.add_argument('--hsv-h', type=float, default=0.015, help='图像HSV-Hue增强')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='图像HSV-Saturation增强')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='图像HSV-Value增强')
    parser.add_argument('--degrees', type=float, default=0.0, help='图像旋转 (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='图像平移 (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='图像缩放 (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='图像剪切 (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='图像透视变换')
    parser.add_argument('--flipud', type=float, default=0.0, help='图像上下翻转增强')
    parser.add_argument('--fliplr', type=float, default=0.5, help='图像左右翻转增强')
    parser.add_argument('--mosaic', type=float, default=1.0, help='图像马赛克增强')
    parser.add_argument('--mixup', type=float, default=0.0, help='图像混合增强')
    parser.add_argument('--copy-paste', type=float, default=0.0, help='分割复制-粘贴增强')
    
    return parser.parse_args()


def setup_config(args):
    """设置MSAE和自适应检测头配置"""
    # 设置多尺度注意力增强和自适应检测头
    config = {
        'enable_msae': bool(args.msae),
        'enable_adaptive_head': bool(args.adaptive),
        'enable_size_aware_loss': bool(args.size_aware_loss),
        'msae_config': {
            'width_mult': 0.5,
            'scale_levels': 4,
            'enable_cross_scale': True
        },
        'adaptive_head_config': {
            'enable_scale_adaptation': bool(args.adaptive),
            'small_object_anchor_ratio': [0.8, 1.2]  # 针对细小裂缝的特殊处理
        },
        'loss_config': {
            'small_obj_weight': 1.5,  # 小目标权重
            'medium_obj_weight': 1.0,  # 中等目标权重
            'large_obj_weight': 0.8,  # 大目标权重
            'small_threshold': 32*32,  # 小目标面积阈值
            'large_threshold': 96*96   # 大目标面积阈值
        }
    }
    
    # 应用配置
    set_config(config)


def create_data_yaml(args):
    """创建或检查数据集配置文件"""
    if not Path(args.data).exists():
        # 如果指定的数据配置文件不存在，创建一个模板
        dataset_path = Path(args.data)
        dataset_dir = dataset_path.parent
        
        if not dataset_dir.exists():
            os.makedirs(dataset_dir, exist_ok=True)
        
        data_dict = {
            'path': './datasets/cracks',  # 数据集根目录
            'train': 'images/train',      # 训练图像相对路径
            'val': 'images/val',          # 验证图像相对路径
            
            'names': {
                0: 'crack'                # 裂缝类别
            },
            
            'nc': 1,                      # 类别数
            'task': 'detect_cracks'       # 任务类型：裂缝检测
        }
        
        # 写入YAML文件
        with open(args.data, 'w') as f:
            yaml.dump(data_dict, f, sort_keys=False)
        
        print(f"已创建数据集配置模板: {args.data}")
        print("请根据您的数据集结构修改此配置文件")
        
        # 创建数据集目录结构
        dataset_root = Path(data_dict['path'])
        if not dataset_root.exists():
            for split in ['train', 'val']:
                (dataset_root / 'images' / split).mkdir(parents=True, exist_ok=True)
                (dataset_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
            print(f"已创建数据集目录结构: {dataset_root}")
            print("请将您的图像和标注放入对应目录")
        
        return False  # 指示用户需要填充数据集
    
    return True  # 数据集配置文件存在


def train(args):
    """训练裂缝检测模型"""
    # 设置注意力增强和自适应检测头配置
    setup_config(args)
    
    # 创建保存目录
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据配置文件获取任务类型（使用obb任务替代detect_cracks）
    with open(args.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    # 使用obb任务，因为它已经在TASK2MODEL和TASK2METRIC中定义
    task = 'obb'
    
    # 初始化模型
    if args.weights:
        model = import_yolo()(args.weights, task=task)
    else:
        model = import_yolo()(args.cfg, task=task)
    
    # 训练模型
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        verbose=True
    )
    
    return results, save_dir


def main(args):
    """主函数"""
    print(f"准备训练YOLOv12裂缝检测模型...")
    
    # 检查数据集配置文件
    data_ready = create_data_yaml(args)
    if not data_ready:
        print("请填充数据集后重新运行训练脚本")
        return
    
    # 检查模型配置文件
    if not Path(args.cfg).exists():
        print(f"错误: 模型配置文件 {args.cfg} 不存在")
        return
    
    # 训练模型
    print(f"开始训练 {args.cfg} 模型...")
    results, save_dir = train(args)
    
    # 输出训练结果
    print(f"训练完成! 模型保存在 {save_dir}")
    
    # 验证模型
    if Path(save_dir / 'weights' / 'best.pt').exists():
        print("\n开始验证最佳模型...")
        best_model = import_yolo()(str(save_dir / 'weights' / 'best.pt'))
        best_model.val(data=args.data)
    

if __name__ == "__main__":
    args = parse_args()
    main(args) 