import os
import sys
import glob
import random
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VisualizationGenerator:
    """
    可视化生成器
    用于生成论文所需的各类可视化图表
    """

    def __init__(self, output_dir: str = "visualization"):
        """
        初始化可视化生成器
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

        subdirs = [
            'training_curves',
            'model_comparison',
            'detection_samples',
            'scene_comparison',
            'confusion_matrix',
            'system_demo'
        ]

        for subdir in subdirs:
            path = os.path.join(output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)

    def generate_detection_samples(self, model_path: str, image_dir: str,
                                    num_samples: int = 8,
                                    output_name: str = "detection_samples.png") -> str:
        """
        生成检测样本可视化图
        Args:
            model_path: 模型路径
            image_dir: 图片目录
            num_samples: 样本数量
            output_name: 输出文件名
        Returns:
            output_path: 输出图片路径
        """
        from ultralytics import YOLO

        model = YOLO(model_path)

        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_files:
            print(f"[Vis] 未找到图片: {image_dir}")
            return None

        samples = random.sample(image_files, min(num_samples, len(image_files)))

        cols = 4
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

        class_names = {0: 'Car', 1: 'Bus', 2: 'Van', 3: 'Others'}
        colors = {
            0: '#FF6B6B',
            1: '#4ECDC4',
            2: '#45B7D1',
            3: '#96CEB4'
        }

        for idx, img_path in enumerate(samples):
            if idx >= len(axes):
                break

            results = model.predict(img_path, conf=0.25, verbose=False)
            result = results[0]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[idx].imshow(img)

            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=colors.get(cls_id, 'red'),
                    facecolor='none'
                )
                axes[idx].add_patch(rect)

                label = f"{class_names.get(cls_id, 'Unknown')} {conf:.2f}"
                axes[idx].text(
                    x1, y1 - 5, label,
                    fontsize=8,
                    color='white',
                    bbox=dict(facecolor=colors.get(cls_id, 'red'), alpha=0.7)
                )

            axes[idx].set_title(os.path.basename(img_path), fontsize=10)
            axes[idx].axis('off')

        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('车辆检测效果展示', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'detection_samples', output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Vis] 检测样本图已生成: {output_path}")
        return output_path

    def generate_scene_comparison(self, model_path: str, scene_dirs: Dict[str, str],
                                   output_name: str = "scene_comparison.png") -> str:
        """
        生成不同场景检测效果对比图
        Args:
            model_path: 模型路径
            scene_dirs: {场景名: 图片目录}
            output_name: 输出文件名
        """
        from ultralytics import YOLO

        model = YOLO(model_path)

        num_scenes = len(scene_dirs)
        fig, axes = plt.subplots(2, num_scenes, figsize=(4 * num_scenes, 8))

        if num_scenes == 1:
            axes = axes.reshape(2, 1)

        for idx, (scene_name, scene_dir) in enumerate(scene_dirs.items()):
            image_files = []
            for ext in ['*.jpg', '*.png']:
                image_files.extend(glob.glob(os.path.join(scene_dir, ext)))

            if not image_files:
                continue

            sample_img = random.choice(image_files)

            img = cv2.imread(sample_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[0, idx].imshow(img)
            axes[0, idx].set_title(f'{scene_name} - 原图', fontsize=11)
            axes[0, idx].axis('off')

            results = model.predict(sample_img, conf=0.25, verbose=False)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            axes[1, idx].imshow(annotated)
            axes[1, idx].set_title(f'{scene_name} - 检测结果', fontsize=11)
            axes[1, idx].axis('off')

        plt.suptitle('不同场景检测效果对比', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'scene_comparison', output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Vis] 场景对比图已生成: {output_path}")
        return output_path

    def generate_enhancement_comparison(self, image_path: str,
                                         output_name: str = "enhancement_comparison.png") -> str:
        """
        生成图像增强效果对比图
        Args:
            image_path: 原始图片路径
            output_name: 输出文件名
        """
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from application.preprocessing import ImageEnhancer

        enhancer = ImageEnhancer()

        img = cv2.imread(image_path)
        if img is None:
            print(f"[Vis] 无法读取图片: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        enhanced_night = enhancer.adjust_brightness(img, gamma=1.5)
        enhanced_clahe = enhancer.clahe_enhance(img)
        enhanced_denoise = enhancer.denoise(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        enhanced_denoise = cv2.cvtColor(enhanced_denoise, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(img)
        axes[0, 0].set_title('原始图像', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(enhanced_night)
        axes[0, 1].set_title('亮度增强 (γ=1.5)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(enhanced_clahe)
        axes[1, 0].set_title('CLAHE对比度增强', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(enhanced_denoise)
        axes[1, 1].set_title('去噪处理', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle('图像预处理增强效果对比', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'detection_samples', output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Vis] 增强对比图已生成: {output_path}")
        return output_path

    def generate_system_architecture_diagram(self,
                                              output_name: str = "system_architecture.png") -> str:
        """
        生成系统架构图
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')

        def draw_box(x, y, w, h, text, color='#4ECDC4', fontsize=10):
            rect = patches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', wrap=True)

        def draw_arrow(x1, y1, x2, y2):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))

        draw_box(0.5, 10, 3, 1.5, '视频流采集\n摄像头/视频文件', '#FF6B6B')
        draw_box(5, 10, 3, 1.5, '图像预处理\n去噪/增强/CLAHE', '#FFB347')
        draw_box(9.5, 10, 3, 1.5, 'YOLOv10\n车辆检测', '#4ECDC4')
        draw_box(13, 10, 2.5, 1.5, 'DeepSORT\n目标追踪', '#45B7D1')

        draw_box(9.5, 7, 3, 1.5, 'ROI区域\n禁停区判定', '#96CEB4')
        draw_box(13, 7, 2.5, 1.5, '违停逻辑\n时间+位移', '#DDA0DD')

        draw_box(9.5, 4, 3, 1.5, '车牌识别\nPaddleOCR', '#87CEEB')
        draw_box(13, 4, 2.5, 1.5, '告警触发\n截图保存', '#FF6B6B')

        draw_box(5, 1, 3, 1.5, 'SQLite数据库\n违规记录存储', '#98FB98')
        draw_box(9.5, 1, 3, 1.5, 'PyQt5 GUI\n实时监控界面', '#DDA0DD')
        draw_box(13, 1, 2.5, 1.5, '报表导出\n统计分析', '#FFB347')

        draw_arrow(3.5, 10.75, 5, 10.75)
        draw_arrow(8, 10.75, 9.5, 10.75)
        draw_arrow(12.5, 10.75, 13, 10.75)
        draw_arrow(14.25, 10, 14.25, 8.5)
        draw_arrow(14.25, 7, 12.5, 7.75)
        draw_arrow(11, 7, 11, 5.5)
        draw_arrow(14.25, 4, 14.25, 2.5)
        draw_arrow(12.5, 1.75, 9.5, 1.75)
        draw_arrow(8, 1.75, 6.5, 1.75)

        ax.text(8, 11.5, '智慧交通违规停车智能识别与告警系统架构',
                ha='center', va='center', fontsize=16, fontweight='bold')

        output_path = os.path.join(self.output_dir, 'system_demo', output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Vis] 系统架构图已生成: {output_path}")
        return output_path

    def generate_violation_logic_flowchart(self,
                                            output_name: str = "violation_logic.png") -> str:
        """
        生成违停判定逻辑流程图
        """
        fig, ax = plt.subplots(figsize=(12, 14))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 14)
        ax.axis('off')

        def draw_box(x, y, w, h, text, shape='rect', color='#4ECDC4'):
            if shape == 'diamond':
                diamond = patches.RegularPolygon(
                    (x + w / 2, y + h / 2), numVertices=4,
                    radius=min(w, h) / 1.5,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2
                )
                ax.add_patch(diamond)
            elif shape == 'oval':
                ellipse = patches.Ellipse(
                    (x + w / 2, y + h / 2), w, h,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2
                )
                ax.add_patch(ellipse)
            else:
                rect = patches.FancyBboxPatch(
                    (x, y), w, h,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2
                )
                ax.add_patch(rect)

            ax.text(x + w / 2, y + h / 2, text,
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', wrap=True)

        def draw_arrow(x1, y1, x2, y2, label=''):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            if label:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x + 0.3, mid_y, label, fontsize=9, color='blue')

        draw_box(4.5, 13, 3, 0.8, '开始', 'oval', '#98FB98')

        draw_box(4.5, 11.5, 3, 1, '获取车辆检测框', 'rect', '#4ECDC4')
        draw_arrow(6, 13, 6, 12.5)

        draw_box(4.5, 9.5, 3, 1.2, '中心点\n是否在ROI内?', 'diamond', '#FFB347')
        draw_arrow(6, 11.5, 6, 10.7)

        draw_box(8.5, 9.5, 2.5, 0.8, '重置计时器', 'rect', '#FF6B6B')
        draw_arrow(7.5, 10.1, 8.5, 9.9, '否')

        draw_box(4.5, 7.5, 3, 1, '开始/继续计时', 'rect', '#4ECDC4')
        draw_arrow(6, 9.5, 6, 8.5, '是')

        draw_box(4.5, 5.5, 3, 1.2, '位移量\n> 阈值?', 'diamond', '#FFB347')
        draw_arrow(6, 7.5, 6, 6.7)

        draw_box(8.5, 5.5, 2.5, 0.8, '重置计时', 'rect', '#FF6B6B')
        draw_arrow(7.5, 6.1, 8.5, 5.9, '是')

        draw_box(4.5, 3.5, 3, 1.2, '停留时间\n≥ 阈值?', 'diamond', '#FFB347')
        draw_arrow(6, 5.5, 6, 4.7, '否')

        draw_box(8.5, 3.5, 2.5, 0.8, '显示警告', 'rect', '#FFB347')
        draw_arrow(7.5, 4.1, 8.5, 3.9, '否')

        draw_box(4.5, 1.5, 3, 1, '触发违停告警\n保存记录', 'rect', '#FF6B6B')
        draw_arrow(6, 3.5, 6, 2.5, '是')

        draw_box(8.5, 1.5, 2.5, 0.8, '车牌OCR识别', 'rect', '#87CEEB')
        draw_arrow(7.5, 2, 8.5, 1.9)

        ax.text(6, 0.5, '违停判定逻辑流程图', ha='center', fontsize=14, fontweight='bold')

        output_path = os.path.join(self.output_dir, 'system_demo', output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[Vis] 违停逻辑流程图已生成: {output_path}")
        return output_path

    def generate_all_visualizations(self, model_path: str, data_dir: str):
        """
        生成所有可视化图表
        Args:
            model_path: 模型路径
            data_dir: 数据目录
        """
        print("\n" + "=" * 60)
        print("开始生成可视化图表...")
        print("=" * 60)

        self.generate_system_architecture_diagram()

        self.generate_violation_logic_flowchart()

        val_dir = os.path.join(data_dir, "dataset", "images", "val")
        if os.path.exists(val_dir):
            self.generate_detection_samples(model_path, val_dir)

        data_archive = os.path.join(os.path.dirname(data_dir), "data", "archive (1)", "images")
        if os.path.exists(data_archive):
            foggy_dir = data_archive
            if os.path.exists(foggy_dir):
                scene_dirs = {'雾天场景': foggy_dir}
                self.generate_scene_comparison(model_path, scene_dirs)

        sample_images = glob.glob(os.path.join(val_dir, "*.jpg")) if os.path.exists(val_dir) else []
        if sample_images:
            self.generate_enhancement_comparison(sample_images[0])

        print("\n" + "=" * 60)
        print("所有可视化图表生成完成!")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


def main():
    """主函数"""
    BASE_DIR = r"C:\Users\86153\ML_Projects\day01\毕设"

    generator = VisualizationGenerator(
        output_dir=os.path.join(BASE_DIR, "visualization")
    )

    model_path = os.path.join(BASE_DIR, "src", "Traffic_Project", "yolov10_train_v1", "weights", "best.pt")
    data_dir = os.path.join(BASE_DIR, "yolo_dataset")

    generator.generate_all_visualizations(model_path, data_dir)


if __name__ == "__main__":
    main()
