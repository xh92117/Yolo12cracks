# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
裂缝检测示例，使用YOLOv12结合多尺度注意力机制
该模型针对不同尺寸裂缝有更好的检测效果
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

# 添加项目路径到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv12项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加ROOT到PATH

from ultralytics import YOLO
from ultralytics.nn.modules.attentions.config import set_config
from ultralytics.utils.plotting import Annotator, colors


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12裂缝检测')
    parser.add_argument('--source', type=str, default='assets/crack_examples', help='图像路径、视频路径或目录')
    parser.add_argument('--model', type=str, default='yolov12n_cracks.pt', help='模型路径或名称')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', type=str, default='', help='cuda设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--save-txt', action='store_true', help='保存结果到*.txt文件')
    parser.add_argument('--msae', type=int, default=1, help='是否启用多尺度注意力增强 (0=禁用，1=启用)')
    parser.add_argument('--adaptive', type=int, default=1, help='是否启用自适应检测头 (0=禁用，1=启用)')
    
    # 高级配置
    parser.add_argument('--imgsz', type=int, default=640, help='推理尺寸 (像素)')
    parser.add_argument('--max-det', type=int, default=300, help='每张图像的最大检测数量')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    
    return parser.parse_args()


def setup_config(args):
    """设置MSAE和自适应检测头配置"""
    # 设置多尺度注意力增强和自适应检测头
    config = {
        'ENABLE_MSAE': bool(args.msae),
        'ENABLE_ADAPTIVE_HEAD': bool(args.adaptive),
        'MSAE_CONFIG': {
            'width_mult': 0.5,
            'scale_levels': 4,
            'enable_cross_scale': True
        },
        'ADAPTIVE_HEAD_CONFIG': {
            'enable_scale_adaptation': bool(args.adaptive),
            'small_object_anchor_ratio': [0.8, 1.2]  # 针对细小裂缝的特殊处理
        }
    }
    
    # 应用配置
    set_config(config)


def run_detection(args):
    """运行裂缝检测"""
    # 初始化模型
    model = YOLO(args.model)
    
    # 推理
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        save_txt=args.save_txt,
        verbose=True
    )
    
    # 打印检测结果
    for r in results:
        boxes = r.boxes  # 检测到的目标边界框
        
        if len(boxes) > 0:
            print(f"在 {r.path} 中检测到 {len(boxes)} 条裂缝")
            # 打印每个检测到的裂缝的置信度
            for box in boxes:
                print(f"- 裂缝: 置信度 {box.conf.item():.4f}, 位置: {box.xyxy[0].tolist()}")
        else:
            print(f"在 {r.path} 中未检测到裂缝")
    
    return results


def main(args):
    """主函数"""
    print(f"正在使用 {args.model} 进行裂缝检测...")
    
    # 设置注意力增强和自适应检测头配置
    setup_config(args)
    
    # 获取源文件信息
    source = Path(args.source)
    is_file = source.is_file()
    is_url = source.as_posix().startswith(('http:/', 'https:/'))
    is_dir = source.is_dir()
    
    if not any([is_file, is_url, is_dir]) and not source.exists():
        print(f"错误: 源 {source} 不存在")
        return
    
    # 运行检测
    results = run_detection(args)
    
    # 显示结果
    if results and args.source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img = cv2.imread(args.source)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotator = Annotator(img)
        
        boxes = results[0].boxes
        for box in boxes:
            b = box.xyxy[0].tolist()  # 获取边界框坐标 (x1, y1, x2, y2)
            c = box.cls.item()  # 获取类别索引
            annotator.box_label(b, f'Crack {box.conf.item():.2f}', color=colors(c, True))
        
        img = annotator.result()
        img = Image.fromarray(img)
        img.show()
    
    print("裂缝检测完成!")


if __name__ == "__main__":
    args = parse_args()
    main(args) 