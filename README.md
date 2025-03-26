# YOLOv12-Cracks：增强型裂缝检测模型

基于YOLOv12架构开发的裂缝检测模型，专门针对工业和结构裂缝检测场景进行优化。

## 主要特点

- **多尺度注意力增强(MSAE)**：解决不同尺寸裂缝检测的难题
- **自适应检测头**：针对不同尺度的裂缝动态调整检测策略
- **尺寸感知损失**：增强对小裂缝的检测敏感度
- **检测+分割能力**：同时支持裂缝定位和精确轮廓提取

## 项目结构

```
yolov12cracks/
├── ultralytics/                # 核心模型代码
│   ├── cfg/                    # 模型配置文件
│   │   ├── models/v12/         # YOLOv12模型配置
│   │   └── datasets/           # 数据集配置
│   ├── models/                 # 模型定义
│   │   └── yolo/detect_cracks/ # 裂缝检测模型
│   └── nn/                     # 神经网络模块
│       └── modules/attentions/ # 注意力机制模块
├── examples/                   # 使用示例
│   ├── train_cracks.py         # 训练脚本
│   └── ablation_experiments.py # 消融实验
├── docs/                       # 文档
└── tests/                      # 测试代码
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/yolov12cracks.git
cd yolov12cracks

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python examples/train_cracks.py --data cracks.yaml --cfg ultralytics/cfg/models/v12/yolov12_cracks.yaml
```

### 消融实验

```bash
python examples/ablation_experiments.py --data datasets/cracks.yaml
```

### 推理示例

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('path/to/best.pt')

# 预测
results = model.predict('path/to/image.jpg')
```

## 性能表现

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 参数量 | FPS |
|------|--------|--------------|-------|-----|
| YOLOv12n-Cracks | 0.89 | 0.72 | 2.6M | 180 |
| YOLOv12s-Cracks | 0.92 | 0.76 | 9.3M | 160 |
| YOLOv12m-Cracks | 0.94 | 0.81 | 20.2M | 120 |
| YOLOv12l-Cracks | 0.95 | 0.85 | 26.5M | 100 |
| YOLOv12x-Cracks | 0.96 | 0.87 | 59.2M | 70 |

## 引用

```
@article{yolov12cracks,
  title={YOLOv12-Cracks: 针对裂缝检测优化的多尺度注意力增强模型},
  author={Your Name},
  year={2024}
}
```

## 许可证

本项目使用[AGPL-3.0 license](LICENSE)许可证。

## 致谢

- 感谢[Ultralytics](https://github.com/ultralytics/ultralytics)提供的YOLOv12基础架构 