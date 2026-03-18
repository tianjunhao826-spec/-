# 智慧交通违规停车智能识别与告警系统

## 项目简介

基于 YOLOv10 和 DeepSORT 的违规停车智能识别与告警系统，实现车辆检测、多目标追踪、禁停区域判定、车牌识别等功能。

## 功能特性

- **车辆检测**: 使用 YOLOv10 进行实时车辆检测 (Car, Bus, Van, Others)
- **多目标追踪**: 使用 DeepSORT 实现跨帧目标追踪
- **禁停区域设置**: 交互式绘制多边形禁停区域 (ROI)
- **违停判定**: 基于时间阈值和位移量的智能违停判定算法
- **车牌识别**: 支持 PaddleOCR/HyperLPR 车牌识别
- **图像增强**: 自适应图像预处理 (夜间/雾天/雨天场景)
- **告警系统**: 实时告警、截图保存、数据库记录
- **报表导出**: 支持 Excel/CSV 导出和统计图表生成

## 项目结构

```
毕设/
├── application/                # 应用程序模块
│   ├── main_system.py         # 主界面 (PyQt5)
│   ├── main_system_simple.py  # 简化版主界面
│   ├── traffic_monitor.py     # 核心监控算法
│   ├── database.py            # 数据库管理
│   ├── preprocessing.py       # 图像预处理模块
│   ├── license_plate_ocr.py   # 车牌识别模块
│   └── report_generator.py    # 报表生成模块
│
├── src/                       # 训练相关
│   ├── train_v10.py          # YOLOv10 训练脚本
│   ├── model_comparison.py   # 模型对比实验
│   ├── ablation_study.py     # 消融实验
│   └── visualization_generator.py  # 可视化生成
│
├── scripts/                   # 数据处理脚本
│   ├── step1_xml2txt.py      # XML标签转换
│   ├── step2_organize.py     # 数据集组织
│   ├── clean_labels.py       # 标签清洗
│   └── build_yolo_dataset.py # 混合数据集构建
│
├── yolo_dataset/              # 数据集配置
│   └── data.yaml             # YOLO 数据集配置
│
└── visualization/             # 可视化输出目录
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.x (推荐)

## 安装依赖

```bash
pip install opencv-python numpy PyQt5 ultralytics deep-sort-realtime paddleocr matplotlib pandas openpyxl
```

## 使用方法

### 1. 运行主程序

```bash
cd application
python main_system_simple.py
```

### 2. 操作步骤

1. 点击 **"加载模型"** 按钮
2. 点击 **"打开视频"** 选择测试视频
3. 点击 **"绘制禁停区"** 在视频上绘制 ROI
4. 观察实时检测效果

### 3. 模型训练

```bash
cd src
python train_v10.py
```

## 核心算法

### 违停判定逻辑

```
车辆进入ROI → 开始计时 → 计算位移量
    ↓
位移量 > 阈值? → 是 → 重置计时 (车辆在移动)
    ↓ 否
累计停留时间
    ↓
时间 ≥ 阈值? → 是 → 判定违停 → 触发告警
```

### 图像预处理

| 场景 | 处理策略 |
|------|---------|
| 夜间 | Gamma校正 + CLAHE + 去噪 |
| 雾天 | 暗通道先验去雾 + CLAHE |
| 雨天 | 非局部均值去噪 + CLAHE |
| 正常 | CLAHE 对比度增强 |

## 实验结果

### 模型性能

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 0.85+ |
| mAP@0.5:0.95 | 0.65+ |
| FPS | 15-25 |

### 消融实验

| 实验 | mAP@0.5 | 提升 |
|------|---------|------|
| 基准模型 | 0.82 | - |
| + 数据增强 | 0.84 | +2.4% |
| + 图像预处理 | 0.86 | +4.9% |
| 完整改进 | 0.88 | +7.3% |

## 技术栈

- **目标检测**: YOLOv10
- **目标追踪**: DeepSORT
- **车牌识别**: PaddleOCR
- **GUI框架**: PyQt5
- **数据库**: SQLite
- **深度学习**: PyTorch

## 许可证

MIT License

## 作者

毕业设计项目
