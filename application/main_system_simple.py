import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QListWidget, QFileDialog, QGroupBox,
    QSpinBox, QCheckBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QStatusBar
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from database import DatabaseManager
from report_generator import ReportGenerator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧交通违规停车智能识别与告警系统")
        self.setGeometry(50, 50, 1400, 900)

        self.db = DatabaseManager(
            db_name=os.path.join(os.path.dirname(__file__), "traffic_violations.db")
        )

        self.monitor = None
        self.model_loaded = False

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.drawing_roi = False
        self.roi_points = []
        self.temp_point = None
        self.frame_count = 0

        self.report_generator = ReportGenerator(self.db)

        self.init_ui()
        self.load_roi_config()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()

        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=3)

        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

        main_widget.setLayout(main_layout)
        self._create_status_bar()

    def _create_left_panel(self):
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
        video_layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()

        self.btn_load_model = QPushButton("🔄 加载模型")
        self.btn_load_model.clicked.connect(self.load_model)
        btn_layout.addWidget(self.btn_load_model)

        self.btn_open = QPushButton("📁 打开视频")
        self.btn_open.clicked.connect(self.open_video)
        btn_layout.addWidget(self.btn_open)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        video_layout.addLayout(btn_layout)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        roi_group = QGroupBox("禁停区域设置")
        roi_layout = QHBoxLayout()

        self.btn_draw_roi = QPushButton("✏️ 绘制禁停区")
        self.btn_draw_roi.clicked.connect(self.start_drawing)
        roi_layout.addWidget(self.btn_draw_roi)

        self.btn_clear_roi = QPushButton("🗑️ 清除区域")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        roi_layout.addWidget(self.btn_clear_roi)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        panel.setLayout(layout)
        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        alert_tab = QWidget()
        alert_layout = QVBoxLayout()
        self.alert_list = QListWidget()
        alert_layout.addWidget(QLabel("实时违规告警记录:"))
        alert_layout.addWidget(self.alert_list)
        alert_tab.setLayout(alert_layout)
        self.tab_widget.addTab(alert_tab, "🚨 违规记录")

        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.stats_table)
        stats_tab.setLayout(stats_layout)
        self.tab_widget.addTab(stats_tab, "📊 统计分析")

        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("请先加载模型")
        self.status_bar.addWidget(self.status_label, 1)
        self.violation_count_label = QLabel("违规次数: 0")
        self.status_bar.addPermanentWidget(self.violation_count_label)

    def load_model(self):
        try:
            self.status_label.setText("正在加载模型...")
            QApplication.processEvents()

            from traffic_monitor import TrafficMonitor

            MODEL_PATH = r"C:\Users\86153\ML_Projects\day01\毕设\src\Traffic_Project\yolov10_train_v1\weights\best.pt"
            config = {
                'parking_threshold': 60,
                'movement_threshold': 30,
                'conf_threshold': 0.4,
                'enable_enhancement': False,
                'enable_plate_recognition': False
            }

            self.monitor = TrafficMonitor(MODEL_PATH, self.db, config)
            self.model_loaded = True
            self.status_label.setText("模型加载完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.status_label.setText("模型加载失败")

    def open_video(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if file_path:
            self._start_capture(file_path)

    def _start_capture(self, source):
        self.stop_video()
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频源!")
            return

        self.btn_stop.setEnabled(True)
        self.timer.start(33)
        self.status_label.setText("监控中...")

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop.setEnabled(False)
        self.status_label.setText("已停止")

    def update_frame(self):
        if not self.cap or not self.monitor:
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

        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        self.frame_count += 1

    def mouse_callback(self, event):
        if self.drawing_roi:
            x, y = event.pos().x(), event.pos().y()
            self.roi_points.append((x, y))
            self.status_label.setText(f"已添加点 {len(self.roi_points)}: ({x}, {y})")

    def start_drawing(self):
        if self.drawing_roi:
            if len(self.roi_points) >= 3:
                if self.monitor:
                    self.monitor.set_roi(self.roi_points)
                self.drawing_roi = False
                self.btn_draw_roi.setText("✏️ 绘制禁停区")
                self.status_label.setText(f"ROI设置完成: {len(self.roi_points)} 个点")
            else:
                QMessageBox.warning(self, "提示", "至少需要3个点!")
        else:
            self.drawing_roi = True
            self.roi_points = []
            if self.monitor:
                self.monitor.set_roi([])
            self.btn_draw_roi.setText("✅ 完成绘制")
            self.status_label.setText("点击绘制禁停区域...")

    def clear_roi(self):
        self.roi_points = []
        if self.monitor:
            self.monitor.set_roi([])
        self.drawing_roi = False
        self.btn_draw_roi.setText("✏️ 绘制禁停区")
        self.status_label.setText("ROI已清除")

    def load_roi_config(self):
        points = self.db.load_roi_config()
        if points and len(points) >= 3:
            self.roi_points = points
            self.status_label.setText(f"已加载ROI配置: {len(points)} 个点")

    def closeEvent(self, event):
        self.stop_video()
        self.db.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
