# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
裂缝检测模型消融实验脚本 - 优化版
用于对比测试不同优化组合的效果:
1. 原始YOLOv12模型 (基准)
2. 仅添加tanh图像增强
3. 仅添加实例分割检测头
4. 添加tanh图像增强 + 多尺度注意力机制
5. 完整优化 (tanh图像增强 + 多尺度注意力机制 + 实例分割检测头)
"""

import argparse
from pathlib import Path
import yaml
import torch
import sys

# 添加项目路径到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv12项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.attentions.config import set_config
from ultralytics.utils.files import increment_path

# 实验配置
EXPERIMENT_CONFIGS = {
    'base': {
        'name': '基础模型',
        'tanh': False,
        'msae': False,
        'segment': False,
        'cfg_key': 'cfg_base'
    },
    'tanh': {
        'name': 'tanh图像增强',
        'tanh': True,
        'msae': False,
        'segment': False,
        'cfg_key': 'cfg_base'
    },
    'segment': {
        'name': '实例分割',
        'tanh': False,
        'msae': False,
        'segment': True,
        'cfg_key': 'cfg_segmentation'
    },
    'tanh_msae': {
        'name': 'tanh增强+多尺度注意力',
        'tanh': True,
        'msae': True,
        'segment': False,
        'cfg_key': 'cfg_cracks'
    },
    'full': {
        'name': '完整优化',
        'tanh': True,
        'msae': True,
        'segment': True,
        'cfg_key': 'cfg_segmentation'
    }
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12裂缝检测模型消融实验')
    
    # 数据和模型配置
    parser.add_argument('--data', type=str, default='datasets/cracks.yaml', help='数据集配置文件')
    parser.add_argument('--cfg-base', type=str, default='ultralytics/cfg/models/v12/yolov12n.yaml', help='基础模型配置')
    parser.add_argument('--cfg-segmentation', type=str, default='ultralytics/cfg/models/v12/yolov12_detect_segment.yaml', help='分割模型配置')
    parser.add_argument('--cfg-cracks', type=str, default='ultralytics/cfg/models/v12/yolov12_cracks.yaml', help='裂缝检测特化模型配置')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='训练图像尺寸')
    parser.add_argument('--device', type=str, default='', help='cuda设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作线程数')
    
    # 实验管理
    parser.add_argument('--project', type=str, default='runs/ablation', help='结果保存目录')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='是否覆盖已存在的实验目录')
    parser.add_argument('--experiments', type=str, default='all', help='要运行的实验，用逗号分隔(base,tanh,segment,tanh_msae,full,all)')
    
    # 优化器设置
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率系数')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减系数')
    
    # tanh增强参数
    parser.add_argument('--tanh-gain', type=float, default=0.5, help='tanh增强强度')
    parser.add_argument('--tanh-threshold', type=float, default=0.2, help='tanh增强阈值')
    
    return parser.parse_args()


def setup_experiment_config(exp_config):
    """设置实验相关配置"""
    # 多尺度注意力配置
    msae_config = {
        'enable_msae': exp_config['msae'],
        'enable_adaptive_head': exp_config['msae'],
        'enable_size_aware_loss': exp_config['msae'],
        'msae_config': {
            'width_mult': 0.5,
            'scale_levels': 4,
            'enable_cross_scale': True
        },
        'adaptive_head_config': {
            'enable_scale_adaptation': exp_config['msae'],
            'small_object_anchor_ratio': [0.8, 1.2]
        },
        'loss_config': {
            'small_obj_weight': 1.5,
            'medium_obj_weight': 1.0,
            'large_obj_weight': 0.8,
            'small_threshold': 32*32,
            'large_threshold': 96*96
        }
    }
    set_config(msae_config)
    return msae_config


def run_experiment(args, exp_id):
    """
    运行单个消融实验
    
    参数:
        args: 命令行参数
        exp_id: 实验ID (base, tanh, segment等)
    
    返回:
        实验结果和保存路径
    """
    exp_config = EXPERIMENT_CONFIGS[exp_id]
    exp_name = f"{args.name}_{exp_id}"
    model_cfg = getattr(args, exp_config['cfg_key'])
    
    # 创建实验目录
    exp_dir = Path(args.project) / exp_name
    exp_dir = increment_path(exp_dir, exist_ok=args.exist_ok)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置实验配置
    setup_experiment_config(exp_config)
    
    # 保存实验配置
    with open(exp_dir / "experiment_config.yaml", 'w') as f:
        config = {
            'name': exp_config['name'],
            'model_config': model_cfg,
            'use_tanh': exp_config['tanh'],
            'use_msae': exp_config['msae'],
            'use_segmentation': exp_config['segment'],
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'image_size': args.imgsz,
            'learning_rate': args.lr0
        }
        yaml.dump(config, f, sort_keys=False)
    
    # 打印实验信息
    print(f"\n{'='*80}")
    print(f"开始实验: {exp_config['name']} ({exp_id})")
    print(f"模型配置: {model_cfg}")
    print(f"使用tanh增强: {'是' if exp_config['tanh'] else '否'}")
    print(f"使用多尺度注意力增强: {'是' if exp_config['msae'] else '否'}")
    print(f"使用实例分割: {'是' if exp_config['segment'] else '否'}")
    print(f"{'='*80}\n")
    
    # 初始化模型
    model = YOLO(model_cfg)
    
    # 准备训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': exp_dir.parent,
        'name': exp_dir.name,
        'exist_ok': True,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'verbose': True
    }
    
    # 添加tanh增强相关参数
    if exp_config['tanh']:
        train_args.update({
            'use_tanh': True,
            'tanh_gain': args.tanh_gain,
            'tanh_threshold': args.tanh_threshold,
            'tanh_channels': 'all'
        })
    
    # 训练模型
    results = model.train(**train_args)
    
    # 保存训练结果摘要
    with open(exp_dir / 'results_summary.txt', 'w') as f:
        f.write(f"实验名称: {exp_config['name']}\n")
        f.write(f"模型配置: {model_cfg}\n")
        f.write(f"使用tanh增强: {'是' if exp_config['tanh'] else '否'}\n")
        f.write(f"使用多尺度注意力增强: {'是' if exp_config['msae'] else '否'}\n")
        f.write(f"使用实例分割: {'是' if exp_config['segment'] else '否'}\n")
        f.write("\n训练结果摘要:\n")
        if hasattr(results, 'keys'):
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(f"结果: {results}\n")
    
    # 验证模型
    if Path(exp_dir / 'weights' / 'best.pt').exists():
        print(f"\n验证最佳模型...")
        val_model = YOLO(str(exp_dir / 'weights' / 'best.pt'))
        val_results = val_model.val(data=args.data)
        
        # 保存验证结果
        with open(exp_dir / 'val_results.txt', 'w') as f:
            f.write("验证结果摘要:\n")
            if hasattr(val_results, 'keys'):
                for key, value in val_results.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"结果: {val_results}\n")
    
    return results, exp_dir


def run_ablation_experiments(args):
    """运行消融实验系列"""
    experiments = args.experiments.split(',')
    if 'all' in experiments:
        experiments = list(EXPERIMENT_CONFIGS.keys())
    
    results = {}
    
    # 运行选中的实验
    for exp_id in experiments:
        if exp_id in EXPERIMENT_CONFIGS:
            results[exp_id] = run_experiment(args, exp_id)
    
    # 生成实验比较报告
    generate_comparison_report(args, results)
    
    return results


def generate_comparison_report(args, results):
    """生成各实验比较报告"""
    report_path = Path(args.project) / f"{args.name}_comparison_report.md"
    
    with open(report_path, 'w') as f:
        # 报告标题和基本信息
        f.write("# 裂缝检测模型优化消融实验报告\n\n")
        f.write("## 实验配置\n\n")
        f.write(f"- 数据集: `{args.data}`\n")
        f.write(f"- 训练轮数: {args.epochs}\n")
        f.write(f"- 批次大小: {args.batch_size}\n")
        f.write(f"- 图像尺寸: {args.imgsz}\n")
        f.write(f"- 学习率: {args.lr0}\n\n")
        
        # 实验组合表格
        f.write("## 实验组合\n\n")
        f.write("| 实验名称 | tanh图像增强 | 多尺度注意力机制 | 实例分割检测头 |\n")
        f.write("|---------|------------|--------------|-------------|\n")
        
        for exp_id in results.keys():
            config = EXPERIMENT_CONFIGS[exp_id]
            f.write(f"| {config['name']} | {'✅' if config['tanh'] else '❌'} | "
                   f"{'✅' if config['msae'] else '❌'} | {'✅' if config['segment'] else '❌'} |\n")
        
        # 性能比较表格
        f.write("\n## 性能比较\n\n")
        f.write("| 实验名称 | mAP@0.5 | mAP@0.5:0.95 | 推理速度 (ms) | 模型大小 (MB) |\n")
        f.write("|---------|---------|--------------|--------------|-------------|\n")
        
        for exp_id, (_, exp_dir) in results.items():
            # 提取性能指标
            metrics = {'mAP50': 'N/A', 'mAP50-95': 'N/A', 'inference_time': 'N/A', 'model_size': 'N/A'}
            
            # 从验证结果文件读取指标
            val_results_file = exp_dir / 'val_results.txt'
            if val_results_file.exists():
                with open(val_results_file, 'r') as vf:
                    content = vf.read()
                    for line in content.splitlines():
                        if 'metrics/mAP50(B)' in line:
                            metrics['mAP50'] = line.split(':')[-1].strip()
                        elif 'metrics/mAP50-95(B)' in line:
                            metrics['mAP50-95'] = line.split(':')[-1].strip()
                        elif 'speed' in line and 'ms' in line:
                            # 尝试提取推理速度
                            metrics['inference_time'] = line.split(':')[-1].strip()
            
            # 计算模型大小
            model_path = exp_dir / 'weights' / 'best.pt'
            if model_path.exists():
                metrics['model_size'] = f"{model_path.stat().st_size / (1024 * 1024):.2f}"
            
            # 写入表格行
            f.write(f"| {EXPERIMENT_CONFIGS[exp_id]['name']} | {metrics['mAP50']} | {metrics['mAP50-95']} | "
                   f"{metrics['inference_time']} | {metrics['model_size']} |\n")
        
        # 结论部分 (留给用户填写)
        f.write("\n## 结论\n\n")
        f.write("*请根据实验结果分析各优化组合的效果...*\n")
    
    print(f"\n比较报告已生成: {report_path}\n")


def main():
    """主函数"""
    args = parse_args()
    
    print("开始裂缝检测模型消融实验...")
    print(f"将运行以下实验: {args.experiments}")
    
    # 创建项目目录
    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行消融实验
    results = run_ablation_experiments(args)
    
    print("\n所有实验完成!")
    print(f"结果保存在: {project_dir.resolve()}")


if __name__ == "__main__":
    main() 