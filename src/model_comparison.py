import os
import sys
import time
import json
from typing import Dict, List, Optional

import torch
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelComparator:
    """
    模型对比实验类
    用于比较不同YOLO模型的性能指标
    """

    def __init__(self, output_dir: str = "visualization/model_comparison"):
        """
        初始化模型对比器
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        self.results = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def add_model(self, name: str, model_path: str):
        """
        添加待对比的模型
        Args:
            name: 模型名称
            model_path: 模型权重路径
        """
        if name not in self.results:
            self.results[name] = {'path': model_path}

    def evaluate_model(self, model_name: str, data_yaml: str) -> Dict:
        """
        评估单个模型
        Args:
            model_name: 模型名称
            data_yaml: 数据集配置文件路径
        Returns:
            metrics: 评估指标字典
        """
        from ultralytics import YOLO

        model_path = self.results[model_name]['path']
        print(f"\n[Comparison] 正在评估模型: {model_name}")
        print(f"[Comparison] 模型路径: {model_path}")

        model = YOLO(model_path)

        print(f"[Comparison] 执行验证集评估...")
        metrics = model.val(
            data=data_yaml,
            split='val',
            imgsz=640,
            batch=16,
            conf=0.001,
            plots=True,
            project=self.output_dir,
            name=model_name,
            exist_ok=True
        )

        self.results[model_name]['metrics'] = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }

        self.results[model_name]['params'] = sum(p.numel() for p in model.model.parameters())

        print(f"[Comparison] {model_name} 评估完成:")
        print(f"  - mAP@0.5: {self.results[model_name]['metrics']['mAP50']:.4f}")
        print(f"  - mAP@0.5:0.95: {self.results[model_name]['metrics']['mAP50-95']:.4f}")
        print(f"  - Precision: {self.results[model_name]['metrics']['precision']:.4f}")
        print(f"  - Recall: {self.results[model_name]['metrics']['recall']:.4f}")

        return self.results[model_name]['metrics']

    def measure_fps(self, model_name: str, test_images: List[str],
                    warmup: int = 10, iterations: int = 100) -> float:
        """
        测量模型FPS
        Args:
            model_name: 模型名称
            test_images: 测试图片路径列表
            warmup: 预热迭代次数
            iterations: 测试迭代次数
        Returns:
            fps: 平均FPS
        """
        from ultralytics import YOLO

        model_path = self.results[model_name]['path']
        model = YOLO(model_path)

        if not test_images:
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_images = [dummy_img]

        print(f"[Comparison] 测量 {model_name} FPS...")

        for _ in range(warmup):
            _ = model.predict(test_images[0], verbose=False)

        start_time = time.time()
        for i in range(iterations):
            img = test_images[i % len(test_images)]
            _ = model.predict(img, verbose=False)
        end_time = time.time()

        total_time = end_time - start_time
        fps = iterations / total_time

        self.results[model_name]['fps'] = fps
        print(f"[Comparison] {model_name} FPS: {fps:.2f}")

        return fps

    def run_full_comparison(self, data_yaml: str, test_images_dir: str = None) -> Dict:
        """
        运行完整的模型对比实验
        Args:
            data_yaml: 数据集配置文件
            test_images_dir: 测试图片目录 (用于FPS测试)
        Returns:
            comparison_results: 对比结果字典
        """
        test_images = []
        if test_images_dir and os.path.exists(test_images_dir):
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                test_images.extend(
                    [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)
                     if f.endswith(ext.replace('*', ''))]
                )

        for model_name in self.results.keys():
            self.evaluate_model(model_name, data_yaml)

            if test_images:
                self.measure_fps(model_name, test_images[:10])

        self.save_results()

        return self.get_comparison_results()

    def get_comparison_results(self) -> Dict:
        """获取对比结果"""
        comparison = {}
        for name, data in self.results.items():
            comparison[name] = {
                **data.get('metrics', {}),
                'fps': data.get('fps', 0),
                'params': data.get('params', 0)
            }
        return comparison

    def save_results(self, filename: str = "comparison_results.json"):
        """保存对比结果到JSON文件"""
        output_path = os.path.join(self.output_dir, filename)

        serializable_results = {}
        for name, data in self.results.items():
            serializable_results[name] = {
                'path': data.get('path', ''),
                'metrics': data.get('metrics', {}),
                'fps': data.get('fps', 0),
                'params': int(data.get('params', 0))
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"[Comparison] 结果已保存: {output_path}")

    def load_results(self, filename: str = "comparison_results.json"):
        """加载已有的对比结果"""
        output_path = os.path.join(self.output_dir, filename)

        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                for name, data in loaded.items():
                    self.results[name] = data
            print(f"[Comparison] 已加载结果: {output_path}")
            return True
        return False

    def generate_comparison_chart(self, output_filename: str = "model_comparison.png"):
        """生成对比图表"""
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not self.results:
            print("[Comparison] 没有可用的对比结果")
            return None

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        model_names = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        metrics_config = [
            ('mAP50', 'mAP@0.5'),
            ('mAP50-95', 'mAP@0.5:0.95'),
            ('precision', '精确率'),
            ('recall', '召回率'),
            ('fps', 'FPS'),
            ('params', '参数量(M)')
        ]

        for idx, (metric_key, metric_name) in enumerate(metrics_config):
            ax = axes[idx // 3, idx % 3]

            values = []
            for model in model_names:
                val = self.results[model].get('metrics', {}).get(metric_key, 0)
                if val == 0:
                    val = self.results[model].get(metric_key, 0)
                if metric_key == 'params':
                    val = val / 1e6 if val > 1e6 else val
                values.append(val)

            bars = ax.bar(model_names, values, color=colors[:len(model_names)],
                          edgecolor='white', linewidth=1.5)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{val:.3f}' if metric_key not in ['fps', 'params'] else f'{val:.1f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('模型性能对比实验', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Comparison] 对比图表已生成: {output_path}")
        return output_path

    def generate_comparison_table(self) -> str:
        """生成对比表格"""
        lines = []
        lines.append("=" * 80)
        lines.append("模型性能对比表")
        lines.append("=" * 80)
        lines.append(f"{'模型':<15} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<10} {'FPS':<8}")
        lines.append("-" * 80)

        for name, data in self.results.items():
            metrics = data.get('metrics', {})
            line = f"{name:<15} "
            line += f"{metrics.get('mAP50', 0):.4f}       "
            line += f"{metrics.get('mAP50-95', 0):.4f}          "
            line += f"{metrics.get('precision', 0):.4f}       "
            line += f"{metrics.get('recall', 0):.4f}    "
            line += f"{data.get('fps', 0):.1f}"
            lines.append(line)

        lines.append("=" * 80)

        table = "\n".join(lines)
        print(table)

        table_path = os.path.join(self.output_dir, "comparison_table.txt")
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(table)

        return table


def main():
    """主函数 - 运行模型对比实验"""
    print("=" * 60)
    print("模型对比实验")
    print("=" * 60)

    BASE_DIR = r"C:\Users\86153\ML_Projects\day01\毕设"
    DATA_YAML = os.path.join(BASE_DIR, "yolo_dataset", "data.yaml")

    comparator = ModelComparator(
        output_dir=os.path.join(BASE_DIR, "visualization", "model_comparison")
    )

    comparator.add_model(
        "YOLOv10s",
        os.path.join(BASE_DIR, "src", "Traffic_Project", "yolov10_train_v1", "weights", "best.pt")
    )

    weights_dir = os.path.join(BASE_DIR, "weights")
    if os.path.exists(os.path.join(weights_dir, "yolov8s.pt")):
        comparator.add_model("YOLOv8s", os.path.join(weights_dir, "yolov8s.pt"))
    if os.path.exists(os.path.join(weights_dir, "yolov8n.pt")):
        comparator.add_model("YOLOv8n", os.path.join(weights_dir, "yolov8n.pt"))

    test_images_dir = os.path.join(BASE_DIR, "yolo_dataset", "dataset", "images", "val")

    try:
        results = comparator.run_full_comparison(DATA_YAML, test_images_dir)

        comparator.generate_comparison_chart()
        comparator.generate_comparison_table()

        print("\n[完成] 模型对比实验完成!")
        print(f"结果保存在: {comparator.output_dir}")

    except Exception as e:
        print(f"[错误] 实验执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
