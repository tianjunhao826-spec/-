import os
import sys
import json
import time
from typing import Dict, List, Optional
from datetime import datetime

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AblationStudy:
    """
    消融实验类
    用于验证各改进模块的有效性
    """

    def __init__(self, output_dir: str = "visualization/ablation_study"):
        """
        初始化消融实验
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        self.results = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_experiment(self, name: str, config: Dict, data_yaml: str) -> Dict:
        """
        运行单个实验
        Args:
            name: 实验名称
            config: 实验配置
            data_yaml: 数据集配置文件
        Returns:
            metrics: 实验结果
        """
        from ultralytics import YOLO

        print(f"\n{'=' * 60}")
        print(f"实验: {name}")
        print(f"配置: {config}")
        print(f"{'=' * 60}")

        model_path = config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"[警告] 模型路径不存在: {model_path}")
            return None

        model = YOLO(model_path)

        train_args = {
            'data': data_yaml,
            'epochs': config.get('epochs', 30),
            'batch': config.get('batch', 8),
            'imgsz': 640,
            'device': config.get('device', '0'),
            'project': self.output_dir,
            'name': name,
            'exist_ok': True,
            'save': True,
            'plots': True,
        }

        if config.get('augment', False):
            train_args.update({
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5,
            })

        if config.get('preprocessing', False):
            train_args['pretrained'] = True

        print(f"[实验] 开始训练...")
        results = model.train(**train_args)

        print(f"[实验] 开始验证...")
        metrics = model.val(
            data=data_yaml,
            split='val',
            imgsz=640,
            batch=16,
            conf=0.001,
            plots=True,
            project=self.output_dir,
            name=f"{name}_val",
            exist_ok=True
        )

        self.results[name] = {
            'config': config,
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'training_time': results.train_time if hasattr(results, 'train_time') else 0
        }

        print(f"[实验] 结果:")
        print(f"  - mAP@0.5: {self.results[name]['mAP50']:.4f}")
        print(f"  - mAP@0.5:0.95: {self.results[name]['mAP50-95']:.4f}")
        print(f"  - Precision: {self.results[name]['precision']:.4f}")
        print(f"  - Recall: {self.results[name]['recall']:.4f}")

        return self.results[name]

    def run_full_ablation(self, base_model: str, data_yaml: str) -> Dict:
        """
        运行完整的消融实验
        Args:
            base_model: 基础模型路径
            data_yaml: 数据集配置文件
        Returns:
            all_results: 所有实验结果
        """
        experiments = {
            'baseline': {
                'model_path': base_model,
                'epochs': 30,
                'augment': False,
                'preprocessing': False,
                'description': '基准模型 (无任何改进)'
            },
            'with_augmentation': {
                'model_path': base_model,
                'epochs': 30,
                'augment': True,
                'preprocessing': False,
                'description': '加入数据增强'
            },
            'with_preprocessing': {
                'model_path': base_model,
                'epochs': 30,
                'augment': False,
                'preprocessing': True,
                'description': '加入预处理增强'
            },
            'full_improvements': {
                'model_path': base_model,
                'epochs': 30,
                'augment': True,
                'preprocessing': True,
                'description': '完整改进方案'
            }
        }

        print("\n" + "=" * 70)
        print("开始消融实验")
        print("=" * 70)

        for name, config in experiments.items():
            self.run_experiment(name, config, data_yaml)

        self.save_results()
        self.generate_ablation_chart()

        return self.results

    def measure_inference_speed(self, model_path: str, test_images: List[str],
                                 warmup: int = 10, iterations: int = 100) -> float:
        """
        测量推理速度
        Args:
            model_path: 模型路径
            test_images: 测试图片列表
            warmup: 预热次数
            iterations: 测试次数
        Returns:
            fps: 平均FPS
        """
        from ultralytics import YOLO

        model = YOLO(model_path)

        if not test_images:
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_images = [dummy]

        for _ in range(warmup):
            _ = model.predict(test_images[0], verbose=False)

        start = time.time()
        for i in range(iterations):
            _ = model.predict(test_images[i % len(test_images)], verbose=False)
        end = time.time()

        return iterations / (end - start)

    def evaluate_different_scenes(self, model_path: str, scene_dirs: Dict[str, str]) -> Dict:
        """
        评估不同场景下的检测性能
        Args:
            model_path: 模型路径
            scene_dirs: {场景名: 图片目录}
        Returns:
            scene_results: 各场景性能
        """
        from ultralytics import YOLO
        import glob

        model = YOLO(model_path)
        scene_results = {}

        print("\n[场景评估] 开始评估不同场景...")

        for scene_name, scene_dir in scene_dirs.items():
            image_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                image_files.extend(glob.glob(os.path.join(scene_dir, ext)))

            if not image_files:
                print(f"[场景评估] {scene_name}: 未找到图片")
                continue

            total_detections = 0
            total_confidence = 0

            for img_path in image_files[:50]:
                results = model.predict(img_path, conf=0.25, verbose=False)
                for r in results:
                    boxes = r.boxes
                    total_detections += len(boxes)
                    for box in boxes:
                        total_confidence += float(box.conf[0])

            avg_detections = total_detections / min(len(image_files), 50)
            avg_confidence = total_confidence / max(total_detections, 1)

            scene_results[scene_name] = {
                'images_tested': min(len(image_files), 50),
                'avg_detections': avg_detections,
                'avg_confidence': avg_confidence
            }

            print(f"[场景评估] {scene_name}: 平均检测数={avg_detections:.1f}, "
                  f"平均置信度={avg_confidence:.3f}")

        self.results['scene_evaluation'] = scene_results
        return scene_results

    def save_results(self, filename: str = "ablation_results.json"):
        """保存实验结果"""
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n[消融实验] 结果已保存: {output_path}")

    def load_results(self, filename: str = "ablation_results.json") -> bool:
        """加载已有结果"""
        output_path = os.path.join(self.output_dir, filename)

        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            return True
        return False

    def generate_ablation_chart(self, output_filename: str = "ablation_comparison.png"):
        """生成消融实验对比图"""
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        experiment_names = []
        map50_values = []
        map50_95_values = []
        precision_values = []
        recall_values = []

        for name, data in self.results.items():
            if name == 'scene_evaluation':
                continue
            if isinstance(data, dict) and 'mAP50' in data:
                experiment_names.append(name)
                map50_values.append(data['mAP50'])
                map50_95_values.append(data['mAP50-95'])
                precision_values.append(data['precision'])
                recall_values.append(data['recall'])

        if not experiment_names:
            print("[消融实验] 没有可用的实验结果")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        name_mapping = {
            'baseline': '基准模型',
            'with_augmentation': '+ 数据增强',
            'with_preprocessing': '+ 预处理增强',
            'full_improvements': '完整改进'
        }

        display_names = [name_mapping.get(n, n) for n in experiment_names]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        ax1 = axes[0, 0]
        bars1 = ax1.bar(display_names, map50_values, color=colors[:len(display_names)])
        ax1.set_title('mAP@0.5 对比', fontsize=12, fontweight='bold')
        ax1.set_ylabel('mAP@0.5')
        ax1.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars1, map50_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax2 = axes[0, 1]
        bars2 = ax2.bar(display_names, map50_95_values, color=colors[:len(display_names)])
        ax2.set_title('mAP@0.5:0.95 对比', fontsize=12, fontweight='bold')
        ax2.set_ylabel('mAP@0.5:0.95')
        ax2.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars2, map50_95_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax3 = axes[1, 0]
        bars3 = ax3.bar(display_names, precision_values, color=colors[:len(display_names)])
        ax3.set_title('精确率 对比', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Precision')
        ax3.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars3, precision_values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax4 = axes[1, 1]
        bars4 = ax4.bar(display_names, recall_values, color=colors[:len(display_names)])
        ax4.set_title('召回率 对比', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Recall')
        ax4.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars4, recall_values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        for ax in axes.flat:
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(ax.get_ylim()[1], 1.0))

        plt.suptitle('消融实验结果对比', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[消融实验] 对比图已生成: {output_path}")
        return output_path

    def generate_ablation_table(self) -> str:
        """生成消融实验表格"""
        lines = []
        lines.append("=" * 90)
        lines.append("消融实验结果表")
        lines.append("=" * 90)
        lines.append(f"{'实验配置':<25} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} "
                     f"{'Precision':<12} {'Recall':<10} {'提升':<8}")
        lines.append("-" * 90)

        baseline_map = 0
        for name, data in self.results.items():
            if name == 'scene_evaluation':
                continue
            if isinstance(data, dict) and 'mAP50' in data:
                if name == 'baseline':
                    baseline_map = data['mAP50']
                    improvement = "-"
                else:
                    improvement = f"+{(data['mAP50'] - baseline_map) * 100:.2f}%"

                line = f"{name:<25} "
                line += f"{data['mAP50']:.4f}       "
                line += f"{data['mAP50-95']:.4f}          "
                line += f"{data['precision']:.4f}       "
                line += f"{data['recall']:.4f}    "
                line += f"{improvement}"
                lines.append(line)

        lines.append("=" * 90)

        table = "\n".join(lines)
        print(table)

        table_path = os.path.join(self.output_dir, "ablation_table.txt")
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(table)

        return table


def main():
    """主函数 - 运行消融实验"""
    print("=" * 70)
    print("消融实验")
    print("=" * 70)

    BASE_DIR = r"C:\Users\86153\ML_Projects\day01\毕设"
    DATA_YAML = os.path.join(BASE_DIR, "yolo_dataset", "data.yaml")
    BASE_MODEL = os.path.join(BASE_DIR, "weights", "yolov10s.pt")

    ablation = AblationStudy(
        output_dir=os.path.join(BASE_DIR, "visualization", "ablation_study")
    )

    if ablation.load_results():
        print("[消融实验] 已加载已有结果")
        ablation.generate_ablation_chart()
        ablation.generate_ablation_table()
    else:
        print("[消融实验] 开始新的实验...")

        if os.path.exists(BASE_MODEL):
            ablation.run_full_ablation(BASE_MODEL, DATA_YAML)
        else:
            print(f"[警告] 基础模型不存在: {BASE_MODEL}")
            print("[提示] 请确保模型文件存在,或修改BASE_MODEL路径")

    print("\n[完成] 消融实验完成!")


if __name__ == "__main__":
    main()
