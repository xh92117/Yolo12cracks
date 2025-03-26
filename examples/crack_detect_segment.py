"""
YOLOv12 裂缝检测与分割示例

本示例展示如何使用YOLOv12新的联合检测分割模型进行裂缝识别
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import DetectSegment  # 导入联合检测分割模型

# 创建示例目录
SAVE_DIR = Path("runs/detect_segment")
os.makedirs(SAVE_DIR, exist_ok=True)


def train_model():
    """训练联合检测分割模型"""
    # 使用自定义配置文件
    model = DetectSegment("ultralytics/cfg/models/v12/yolov12_detect_segment.yaml")
    
    # 训练模型
    # 注意：需要准备好同时包含边界框和分割掩码标注的数据集
    results = model.train(
        data="ultralytics/cfg/datasets/crack-seg.yaml",  # 数据集
        epochs=50,                    # 训练轮数
        imgsz=640,                   # 输入尺寸
        batch=16,                    # 批次大小
        device=0,                    # GPU设备
        plots=True,                  # 保存绘图
        save=True,                   # 保存模型
        project=SAVE_DIR,            # 保存路径
        name="train",                # 实验名称
        custom_aug=0.5,              # 使用自定义增强
    )
    
    return model


def validate_model(model=None):
    """验证模型性能"""
    if model is None:
        # 从已训练的权重加载模型
        model = DetectSegment(SAVE_DIR / "train" / "weights" / "best.pt")
    
    # 验证性能
    metrics = model.val(
        data="ultralytics/cfg/datasets/crack-seg.yaml",  # 数据集
        imgsz=640,                   # 输入尺寸
        batch=16,                    # 批次大小
        device=0,                    # GPU设备
        project=SAVE_DIR,            # 保存路径
        name="val",                  # 实验名称
    )
    
    print(f"验证结果: 检测mAP={metrics['detect_metrics/mAP50-95(B)']:.3f}, "
          f"分割mAP={metrics['segment_metrics/mAP50-95(M)']:.3f}")
    
    return model


def predict_and_visualize(model=None, source=None):
    """预测并可视化结果"""
    if model is None:
        # 从已训练的权重加载模型
        model = DetectSegment(SAVE_DIR / "train" / "weights" / "best.pt")
    
    if source is None:
        # 使用默认图像或测试集的第一张图像
        source = "ultralytics/assets/crack1.jpg"
    
    # 进行预测
    results = model.predict(
        source=source,               # 图像路径
        imgsz=640,                   # 输入尺寸
        conf=0.25,                   # 置信度阈值
        device=0,                    # GPU设备
        project=SAVE_DIR,            # 保存路径
        name="predict",              # 实验名称
        save=True,                   # 保存结果
        show=False,                  # 不显示结果
    )
    
    # 可视化结果
    for idx, result in enumerate(results):
        img = result.orig_img.copy()
        
        # 获取边界框和掩码
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') else None
        
        # 绘制结果
        plt.figure(figsize=(16, 10))
        
        # 原图
        plt.subplot(1, 3, 1)
        plt.title("原始图像")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        # 检测结果
        plt.subplot(1, 3, 2)
        plt.title("目标检测结果")
        det_img = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        # 分割结果
        plt.subplot(1, 3, 3)
        plt.title("实例分割结果")
        seg_img = img.copy()
        if masks is not None:
            for mask in masks:
                color = np.random.randint(0, 255, 3).tolist()
                mask_img = np.zeros_like(img)
                mask_img[mask[0]] = color
                seg_img = cv2.addWeighted(seg_img, 1, mask_img, 0.5, 0)
        
        plt.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(SAVE_DIR / "predict" / f"visualization_{idx}.png")
        plt.close()

    return results


def main():
    """主函数"""
    # 训练模型
    model = train_model()
    
    # 验证模型
    validate_model(model)
    
    # 预测并可视化
    predict_and_visualize(model)
    
    print(f"所有结果已保存到: {SAVE_DIR}")


if __name__ == "__main__":
    main() 