import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QListWidget, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QStatusBar, QProgressBar
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

from database import DatabaseManager
from traffic_monitor import TrafficMonitor
from report_generator import ReportGenerator


class VideoThread(QThread):
    """视频处理线程"""
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, monitor, cap):
        super().__init__()
        self.monitor = monitor
        self.cap = cap
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed = self.monitor.process_frame(frame)
            self.frame_ready.emit(processed)
        self.finished.emit()

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """
    智慧交通违规停车智能识别与告警系统主界面
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧交通违规停车智能识别与告警系统")
        self.setGeometry(50, 50, 1400, 900)

        self.db = DatabaseManager(
            db_name=os.path.join(os.path.dirname(__file__), "traffic_violations.db")
        )

        MODEL_PATH = r"C:\Users\86153\ML_Projects\day01\毕设\src\Traffic_Project\yolov10_train_v1\weights\best.pt"

        config = {
            'parking_threshold': 60,
            'movement_threshold': 30,
            'conf_threshold': 0.4,
            'enable_enhancement': False,  # 禁用图像增强
            'enable_plate_recognition': False  # 禁用车牌识别
        }

        self.monitor = TrafficMonitor(MODEL_PATH, self.db, config)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_thread = None

        self.drawing_roi = False
        self.roi_points = []
        self.temp_point = None
        self.frame_count = 0

        self.report_generator = ReportGenerator(self.db)

        self.init_ui()

        self.load_roi_config()

    def init_ui(self):
        """初始化用户界面"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()

        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=3)

        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

        main_widget.setLayout(main_layout)

        self._create_status_bar()

    def _create_left_panel(self) -> QWidget:
        """创建左侧面板 (视频显示区)"""
        panel = QWidget()
        layout = QVBoxLayout()

        video_group = QGroupBox("实时监控")
        video_layout = QVBoxLayout()

        self.video_label = QLabel("点击'打开视频'开始监控")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px solid #3498db;
                background-color: #1a1a2e;
                color: #ffffff;
                font-size: 16px;
            }
        """)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.mousePressEvent = self.mouse_callback
        self.video_label.mouseMoveEvent = self.mouse_move_callback
        video_layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()

        self.btn_open = QPushButton("📁 打开视频")
        self.btn_open.clicked.connect(self.open_video)
        self.btn_open.setStyleSheet("padding: 8px; font-size: 12px;")

        self.btn_camera = QPushButton("📷 打开摄像头")
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_camera.setStyleSheet("padding: 8px; font-size: 12px;")

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setStyleSheet("padding: 8px; font-size: 12px;")
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)

        video_layout.addLayout(btn_layout)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        roi_group = QGroupBox("禁停区域设置")
        roi_layout = QHBoxLayout()

        self.btn_draw_roi = QPushButton("✏️ 绘制禁停区")
        self.btn_draw_roi.clicked.connect(self.start_drawing)
        self.btn_draw_roi.setStyleSheet("padding: 8px; font-size: 12px;")

        self.btn_clear_roi = QPushButton("🗑️ 清除区域")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_clear_roi.setStyleSheet("padding: 8px; font-size: 12px;")

        self.btn_save_roi = QPushButton("💾 保存配置")
        self.btn_save_roi.clicked.connect(self.save_roi_config)
        self.btn_save_roi.setStyleSheet("padding: 8px; font-size: 12px;")

        roi_layout.addWidget(self.btn_draw_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        roi_layout.addWidget(self.btn_save_roi)
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        settings_group = QGroupBox("系统设置")
        settings_layout = QHBoxLayout()

        threshold_layout = QVBoxLayout()
        threshold_layout.addWidget(QLabel("违停阈值(秒):"))
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(10, 300)
        self.spin_threshold.setValue(60)
        self.spin_threshold.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.spin_threshold)
        settings_layout.addLayout(threshold_layout)

        movement_layout = QVBoxLayout()
        movement_layout.addWidget(QLabel("位移阈值(像素):"))
        self.spin_movement = QSpinBox()
        self.spin_movement.setRange(5, 100)
        self.spin_movement.setValue(30)
        self.spin_movement.valueChanged.connect(self.update_movement_threshold)
        movement_layout.addWidget(self.spin_movement)
        settings_layout.addLayout(movement_layout)

        enhance_layout = QVBoxLayout()
        enhance_layout.addWidget(QLabel("图像增强:"))
        self.check_enhance = QCheckBox("启用")
        self.check_enhance.setChecked(True)
        self.check_enhance.stateChanged.connect(self.toggle_enhancement)
        enhance_layout.addWidget(self.check_enhance)
        settings_layout.addLayout(enhance_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        panel.setLayout(layout)
        return panel

    def _create_right_panel(self) -> QWidget:
        """创建右侧面板 (功能选项卡)"""
        panel = QWidget()
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        alert_tab = self._create_alert_tab()
        self.tab_widget.addTab(alert_tab, "🚨 违规记录")

        stats_tab = self._create_stats_tab()
        self.tab_widget.addTab(stats_tab, "📊 统计分析")

        export_tab = self._create_export_tab()
        self.tab_widget.addTab(export_tab, "📤 导出报表")

        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel

    def _create_alert_tab(self) -> QWidget:
        """创建违规记录选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        self.alert_list = QListWidget()
        self.alert_list.setStyleSheet("""
            QListWidget {
                font-size: 11px;
                border: 1px solid #ccc;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e74c3c;
                color: white;
            }
        """)
        layout.addWidget(QLabel("实时违规告警记录:"))
        layout.addWidget(self.alert_list)

        self.btn_clear_alerts = QPushButton("清空记录列表")
        self.btn_clear_alerts.clicked.connect(self.clear_alert_list)
        layout.addWidget(self.btn_clear_alerts)

        tab.setLayout(layout)
        return tab

    def _create_stats_tab(self) -> QWidget:
        """创建统计分析选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.stats_table)

        self.btn_refresh_stats = QPushButton("刷新统计数据")
        self.btn_refresh_stats.clicked.connect(self.refresh_statistics)
        layout.addWidget(self.btn_refresh_stats)

        tab.setLayout(layout)
        self.refresh_statistics()
        return tab

    def _create_export_tab(self) -> QWidget:
        """创建导出报表选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        export_group = QGroupBox("导出选项")
        export_layout = QVBoxLayout()

        self.btn_export_excel = QPushButton("📥 导出Excel报表")
        self.btn_export_excel.clicked.connect(self.export_excel)
        self.btn_export_excel.setStyleSheet("padding: 10px; font-size: 12px;")

        self.btn_export_csv = QPushButton("📥 导出CSV文件")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setStyleSheet("padding: 10px; font-size: 12px;")

        self.btn_export_chart = QPushButton("📊 生成统计图表")
        self.btn_export_chart.clicked.connect(self.export_chart)
        self.btn_export_chart.setStyleSheet("padding: 10px; font-size: 12px;")

        export_layout.addWidget(self.btn_export_excel)
        export_layout.addWidget(self.btn_export_csv)
        export_layout.addWidget(self.btn_export_chart)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)

        self.fps_label = QLabel("FPS: --")
        self.status_bar.addPermanentWidget(self.fps_label)

        self.violation_count_label = QLabel("违规次数: 0")
        self.status_bar.addPermanentWidget(self.violation_count_label)

    def open_video(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if file_path:
            self._start_capture(file_path)

    def open_camera(self):
        """打开摄像头"""
        self._start_capture(0)

    def _start_capture(self, source):
        """启动视频捕获"""
        self.stop_video()

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频源!")
            return

        self.btn_stop.setEnabled(True)
        self.timer.start(30)

        self.status_label.setText("监控中...")

    def stop_video(self):
        """停止视频播放"""
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_stop.setEnabled(False)
        self.status_label.setText("已停止")

    def update_frame(self):
        """更新视频帧"""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        frame = cv2.resize(frame, (800, 500))

        processed_frame = self.monitor.process_frame(frame)

        if self.drawing_roi and len(self.roi_points) > 0:
            for i, p in enumerate(self.roi_points):
                cv2.circle(processed_frame, p, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(processed_frame, self.roi_points[i - 1], p, (0, 255, 0), 2)

            if self.temp_point:
                cv2.line(processed_frame, self.roi_points[-1], self.temp_point, (0, 255, 0), 1)

        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        if self.frame_count % 30 == 0:
            stats = self.monitor.get_statistics()
            self.violation_count_label.setText(f"违规次数: {stats['total_violations']}")
            self.refresh_alert_list()

        self.frame_count += 1

    def mouse_callback(self, event):
        """鼠标点击事件 - 绘制ROI"""
        if self.drawing_roi:
            x = event.pos().x()
            y = event.pos().y()
            self.roi_points.append((x, y))
            self.status_label.setText(f"已添加点 {len(self.roi_points)}: ({x}, {y})")

    def mouse_move_callback(self, event):
        """鼠标移动事件"""
        if self.drawing_roi:
            self.temp_point = (event.pos().x(), event.pos().y())

    def start_drawing(self):
        """开始绘制ROI"""
        if self.drawing_roi:
            if len(self.roi_points) >= 3:
                self.monitor.set_roi(self.roi_points)
                self.drawing_roi = False
                self.btn_draw_roi.setText("✏️ 绘制禁停区")
                self.status_label.setText(f"ROI设置完成: {len(self.roi_points)} 个点")
            else:
                QMessageBox.warning(self, "提示", "至少需要3个点才能构成区域!")
        else:
            self.drawing_roi = True
            self.roi_points = []
            self.monitor.set_roi([])
            self.btn_draw_roi.setText("✅ 完成绘制")
            self.status_label.setText("请在视频区域点击绘制禁停区域...")

    def clear_roi(self):
        """清除ROI"""
        self.roi_points = []
        self.monitor.set_roi([])
        self.drawing_roi = False
        self.btn_draw_roi.setText("✏️ 绘制禁停区")
        self.status_label.setText("ROI已清除")

    def save_roi_config(self):
        """保存ROI配置"""
        if len(self.roi_points) >= 3:
            self.db.save_roi_config("default", self.roi_points)
            QMessageBox.information(self, "成功", "ROI配置已保存!")
        else:
            QMessageBox.warning(self, "提示", "请先绘制有效的ROI区域!")

    def load_roi_config(self):
        """加载ROI配置"""
        points = self.db.load_roi_config()
        if points and len(points) >= 3:
            self.roi_points = points
            self.monitor.set_roi(points)
            self.status_label.setText(f"已加载ROI配置: {len(points)} 个点")

    def update_threshold(self, value):
        """更新违停阈值"""
        self.monitor.set_parking_threshold(value)
        self.status_label.setText(f"违停阈值已更新: {value}秒")

    def update_movement_threshold(self, value):
        """更新位移阈值"""
        self.monitor.set_movement_threshold(value)
        self.status_label.setText(f"位移阈值已更新: {value}像素")

    def toggle_enhancement(self, state):
        """切换图像增强"""
        enabled = state == Qt.Checked
        self.monitor.set_enhancement(enabled)
        self.status_label.setText(f"图像增强: {'开启' if enabled else '关闭'}")

    def refresh_alert_list(self):
        """刷新告警列表"""
        records = self.db.get_all_violations()
        current_count = self.alert_list.count()

        if len(records) > current_count:
            latest = records[0]
            plate = latest[3] if latest[3] else "未识别"
            duration = f"{latest[6]:.1f}s" if latest[6] else "--"
            item_text = f"[{latest[1]}] {latest[2]} | {plate} | {duration}"
            self.alert_list.insertItem(0, item_text)

    def clear_alert_list(self):
        """清空告警列表显示"""
        self.alert_list.clear()

    def refresh_statistics(self):
        """刷新统计数据"""
        stats = self.db.get_statistics()

        self.stats_table.setRowCount(0)

        stat_items = [
            ("总违规次数", stats.get('total_violations', 0)),
            ("独立车辆数", stats.get('unique_vehicles', 0)),
            ("平均停留时长(秒)", f"{stats.get('avg_duration', 0):.1f}"),
        ]

        for name, value in stat_items:
            row = self.stats_table.rowCount()
            self.stats_table.insertRow(row)
            self.stats_table.setItem(row, 0, QTableWidgetItem(name))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def export_excel(self):
        """导出Excel报表"""
        try:
            output_path = self.report_generator.export_to_excel()
            if output_path:
                QMessageBox.information(self, "成功", f"Excel报表已导出:\n{output_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")

    def export_csv(self):
        """导出CSV文件"""
        try:
            output_path = self.report_generator.export_to_csv()
            if output_path:
                QMessageBox.information(self, "成功", f"CSV文件已导出:\n{output_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")

    def export_chart(self):
        """生成统计图表"""
        try:
            output_path = self.report_generator.generate_statistics_chart()
            if output_path:
                QMessageBox.information(self, "成功", f"统计图表已生成:\n{output_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"生成失败: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_video()
        self.db.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
