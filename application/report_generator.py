import os
import datetime
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """
    报表生成器
    支持导出Excel报表和生成统计图表
    """

    def __init__(self, db_manager, output_dir="reports"):
        """
        初始化报表生成器
        Args:
            db_manager: 数据库管理器实例
            output_dir: 报表输出目录
        """
        self.db = db_manager
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_to_excel(self, start_date: str = None, end_date: str = None,
                        filename: str = None) -> str:
        """
        导出Excel报表
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            filename: 自定义文件名
        Returns:
            output_path: 导出文件路径
        """
        try:
            import pandas as pd
        except ImportError:
            print("[Report] 需要安装pandas: pip install pandas openpyxl")
            return None

        if start_date and end_date:
            records = self.db.get_violations_by_date(start_date, end_date)
        else:
            records = self.db.get_all_violations()

        if not records:
            print("[Report] 没有数据可导出")
            return None

        df = pd.DataFrame(records, columns=[
            'ID', '时间', '追踪ID', '车牌号', '车辆类型',
            '位置', '停留时长(秒)', '截图路径', '状态'
        ])

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"违规记录报表_{timestamp}.xlsx"

        output_path = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='违规记录', index=False)

            stats = self.db.get_statistics()
            stats_df = pd.DataFrame([
                {'指标': '总违规次数', '数值': stats['total_violations']},
                {'指标': '独立车辆数', '数值': stats['unique_vehicles']},
                {'指标': '平均停留时长(秒)', '数值': stats['avg_duration']},
            ])
            stats_df.to_excel(writer, sheet_name='统计概览', index=False)

        print(f"[Report] Excel报表已导出: {output_path}")
        return output_path

    def export_to_csv(self, start_date: str = None, end_date: str = None,
                      filename: str = None) -> str:
        """导出CSV格式报表"""
        try:
            import pandas as pd
        except ImportError:
            print("[Report] 需要安装pandas")
            return None

        if start_date and end_date:
            records = self.db.get_violations_by_date(start_date, end_date)
        else:
            records = self.db.get_all_violations()

        if not records:
            return None

        df = pd.DataFrame(records, columns=[
            'ID', '时间', '追踪ID', '车牌号', '车辆类型',
            '位置', '停留时长(秒)', '截图路径', '状态'
        ])

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"违规记录_{timestamp}.csv"

        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"[Report] CSV报表已导出: {output_path}")
        return output_path

    def generate_statistics_chart(self, filename: str = None) -> str:
        """
        生成统计图表
        Returns:
            output_path: 图表文件路径
        """
        stats = self.db.get_statistics()

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"统计图表_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)

        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_daily_trend(ax1, stats)

        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_vehicle_type_pie(ax2, stats)

        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_hourly_distribution(ax3, stats)

        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_summary_cards(ax4, stats)

        plt.suptitle('违规停车统计分析报告', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Report] 统计图表已生成: {output_path}")
        return output_path

    def _plot_daily_trend(self, ax, stats):
        """绘制每日趋势图"""
        daily_data = stats.get('daily_recent', {})

        if daily_data:
            dates = list(daily_data.keys())
            counts = list(daily_data.values())

            ax.plot(dates, counts, marker='o', linewidth=2, markersize=8, color='#2E86AB')
            ax.fill_between(dates, counts, alpha=0.3, color='#2E86AB')

            ax.set_xlabel('日期')
            ax.set_ylabel('违规次数')
            ax.set_title('近7日违规趋势')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=14)
            ax.set_title('近7日违规趋势')

    def _plot_vehicle_type_pie(self, ax, stats):
        """绘制车辆类型饼图"""
        type_data = stats.get('by_type', {})

        if type_data:
            labels = []
            sizes = []
            for k, v in type_data.items():
                labels.append(k if k else '未知')
                sizes.append(v)

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            explode = [0.05] * len(labels)

            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors[:len(labels)], explode=explode,
                shadow=True, startangle=90
            )

            ax.set_title('车辆类型分布')
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=14)
            ax.set_title('车辆类型分布')

    def _plot_hourly_distribution(self, ax, stats):
        """绘制时段分布柱状图"""
        hourly_data = stats.get('by_hour', {})

        if hourly_data:
            hours = sorted([int(h) for h in hourly_data.keys()])
            counts = [hourly_data[str(h)] for h in hours]

            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(hours)))

            bars = ax.bar(hours, counts, color=colors, edgecolor='white', linewidth=0.5)

            ax.set_xlabel('时段 (小时)')
            ax.set_ylabel('违规次数')
            ax.set_title('违规时段分布')
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3, axis='y')

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=14)
            ax.set_title('违规时段分布')

    def _plot_summary_cards(self, ax, stats):
        """绘制统计卡片"""
        ax.axis('off')

        summary_data = [
            ('总违规次数', stats.get('total_violations', 0), '#FF6B6B'),
            ('独立车辆数', stats.get('unique_vehicles', 0), '#4ECDC4'),
            ('平均停留时长', f"{stats.get('avg_duration', 0):.1f}秒", '#45B7D1'),
        ]

        y_positions = [0.7, 0.4, 0.1]

        for (label, value, color), y in zip(summary_data, y_positions):
            ax.add_patch(plt.Rectangle((0.1, y), 0.8, 0.2,
                                        facecolor=color, alpha=0.2,
                                        edgecolor=color, linewidth=2))

            ax.text(0.5, y + 0.13, str(value), ha='center', va='center',
                    fontsize=24, fontweight='bold', color=color)
            ax.text(0.5, y + 0.03, label, ha='center', va='center',
                    fontsize=12, color='gray')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('统计概览')

    def generate_detection_comparison_chart(self, comparison_results: Dict,
                                             filename: str = None) -> str:
        """
        生成模型检测效果对比图
        Args:
            comparison_results: 对比结果字典
            filename: 输出文件名
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"模型对比_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'fps', 'params']
        metric_names = ['mAP@0.5', 'mAP@0.5:0.95', '精确率', '召回率', 'FPS', '参数量(M)']

        model_names = list(comparison_results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 3, idx % 3]

            values = []
            for model in model_names:
                val = comparison_results[model].get(metric, 0)
                if metric == 'params':
                    val = val / 1e6
                values.append(val)

            bars = ax.bar(model_names, values, color=colors[:len(model_names)],
                          edgecolor='white', linewidth=1.5)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{val:.3f}' if metric != 'fps' else f'{val:.1f}',
                        ha='center', va='bottom', fontsize=10)

            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('模型性能对比', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Report] 模型对比图已生成: {output_path}")
        return output_path

    def generate_scene_comparison_chart(self, scene_results: Dict,
                                         filename: str = None) -> str:
        """
        生成不同场景检测效果对比图
        Args:
            scene_results: {场景名: {指标: 值}}
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"场景对比_{timestamp}.png"

        output_path = os.path.join(self.output_dir, filename)

        fig, ax = plt.subplots(figsize=(12, 6))

        scenes = list(scene_results.keys())
        metrics = ['mAP50', 'precision', 'recall']
        x = np.arange(len(scenes))
        width = 0.25

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [scene_results[s].get(metric, 0) for s in scenes]
            bars = ax.bar(x + i * width, values, width, label=metric, color=color)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('场景类型')
        ax.set_ylabel('性能指标')
        ax.set_title('不同场景检测性能对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Report] 场景对比图已生成: {output_path}")
        return output_path


if __name__ == "__main__":
    print("报表生成模块测试...")

    class MockDB:
        def get_all_violations(self):
            return [
                (1, '2024-01-15 10:30:00', 'Car_1', '京A12345', 'car', '禁停区A', 65.5, 'test.jpg', '未处理'),
                (2, '2024-01-15 11:00:00', 'Car_2', '京B67890', 'bus', '禁停区A', 120.0, 'test2.jpg', '已处理'),
            ]

        def get_statistics(self):
            return {
                'total_violations': 100,
                'unique_vehicles': 45,
                'avg_duration': 75.5,
                'by_type': {'car': 60, 'bus': 25, 'van': 15},
                'by_hour': {'08': 10, '09': 15, '10': 20, '14': 25, '18': 30},
                'daily_recent': {
                    '2024-01-10': 12, '2024-01-11': 15, '2024-01-12': 8,
                    '2024-01-13': 20, '2024-01-14': 18, '2024-01-15': 22
                }
            }

    generator = ReportGenerator(MockDB(), output_dir="test_reports")

    chart_path = generator.generate_statistics_chart()
    print(f"生成的图表: {chart_path}")

    print("报表生成模块测试完成!")
