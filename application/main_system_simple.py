import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QListWidget, QFileDialog, QGroupBox,
    QSpinBox, QCheckBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QStatusBar, QListWidgetItem
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 确保在同级目录或设置好PYTHONPATH
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
        self.setup_styles()

    def refresh_ui_data(self):
        # 1. 刷新状态栏的违规总数
        if self.monitor:
            stats = self.monitor.get_statistics()
            self.violation_count_label.setText(f"违规次数: {stats.get('total_violations', 0)}")

        # 2. 刷新告警日志列表
        try:
            records = self.db.get_all_violations()

            # 使用字典来跟踪已经在界面上的车辆条目 {car_id: QListWidgetItem}
            if not hasattr(self, 'alert_items_map'):
                self.alert_items_map = {}
                self.alert_list.clear()

            # 遍历最近的 50 条记录（倒序遍历）
            for record in reversed(records[:50]):
                record_time = record[1] if len(record) > 1 else "未知时间"
                car_id = record[2] if len(record) > 2 else "未知车辆"
                plate = record[3] if len(record) > 3 and record[3] else "未识别"
                duration = f"{record[6]:.1f}s" if len(record) > 6 and record[6] else "--"

                # 拼接优美的展示文本 (包含最优车牌)
                item_text = f"[{record_time}] 违规车辆 ID: {car_id} | 车牌: {plate} | 违停时长: {duration}"

                # 原地更新或插入新条目
                if car_id in self.alert_items_map:
                    self.alert_items_map[car_id].setText(item_text)
                else:
                    new_item = QListWidgetItem(item_text)
                    self.alert_list.insertItem(0, new_item)
                    self.alert_items_map[car_id] = new_item

        except Exception as e:
            print(f"刷新告警日志时出错: {e}")

    def setup_styles(self):
        # 统一设置字体
        font = self.font()
        font.setFamily("Segoe UI, Microsoft YaHei")
        QApplication.setFont(font)

        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QWidget#panel_container { background-color: #111827; border-radius: 12px; border: 1px solid #1f2937; }
            QGroupBox { font-size: 15px; font-weight: bold; color: #60a5fa; border: 1px solid #1f2937; border-radius: 8px; margin-top: 20px; padding-top: 20px; background-color: rgba(31, 41, 55, 0.3); }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 15px; top: -5px; padding: 4px 10px; background-color: #1f2937; color: #93c5fd; border-radius: 4px; border: 1px solid #374151; }
            QLabel { color: #d1d5db; }
            QPushButton { background-color: #1f2937; color: #e5e7eb; border-radius: 6px; padding: 10px 15px; font-size: 14px; font-weight: bold; border: 1px solid #374151; }
            QPushButton:hover { background-color: #374151; border: 1px solid #60a5fa; color: #ffffff; }
            QPushButton:pressed { background-color: #111827; }
            QPushButton#primary_btn { background-color: rgba(37, 99, 235, 0.15); color: #60a5fa; border: 1px solid #2563eb; }
            QPushButton#primary_btn:hover { background-color: rgba(37, 99, 235, 0.4); border: 1px solid #93c5fd; color: #ffffff; }
            QPushButton#stop_btn { background-color: rgba(220, 38, 38, 0.1); color: #f87171; border: 1px solid #dc2626; }
            QPushButton#stop_btn:hover { background-color: rgba(220, 38, 38, 0.3); border: 1px solid #fca5a5; color: #ffffff; }
            QLabel#video_display { background-color: #000000; border: 2px solid #1e3a8a; border-radius: 10px; color: #4b5563; font-size: 20px; font-weight: bold; }
            QSpinBox { background-color: #1f2937; color: #60a5fa; border: 1px solid #374151; border-radius: 5px; padding: 6px 10px; font-size: 14px; font-weight: bold; }
            QSpinBox:focus { border: 1px solid #3b82f6; background-color: #111827; }
            QSpinBox::up-button, QSpinBox::down-button { background-color: #374151; border-radius: 3px; width: 24px; margin: 1px; }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #4b5563; }
            QTabWidget::pane { border: 1px solid #1f2937; border-radius: 8px; background-color: #111827; top: -1px; }
            QTabBar::tab { background-color: #1f2937; color: #9ca3af; padding: 12px 25px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px; font-weight: bold; font-size: 14px; }
            QTabBar::tab:selected { background-color: #2563eb; color: #ffffff; }
            QTabBar::tab:hover:!selected { background-color: #374151; }
            QListWidget, QTableWidget { background-color: transparent; border: none; color: #d1d5db; outline: none; font-size: 13px; }
            QListWidget::item { padding: 12px 8px; border-bottom: 1px solid #1f2937; }
            QListWidget::item:hover { background-color: rgba(55, 65, 81, 0.4); border-radius: 6px; }
            QHeaderView::section { background-color: #1f2937; color: #9ca3af; padding: 10px; border: none; font-weight: bold; font-size: 13px; border-bottom: 2px solid #2563eb; }
            QStatusBar { background-color: #111827; color: #9ca3af; border-top: 1px solid #1f2937; font-size: 12px; }
        """)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_panel = self._create_left_panel()
        left_container = QWidget()
        left_container.setObjectName("panel_container")
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.addWidget(left_panel)
        main_layout.addWidget(left_container, stretch=7)

        right_panel = self._create_right_panel()
        right_container = QWidget()
        right_container.setObjectName("panel_container")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.addWidget(right_panel)
        main_layout.addWidget(right_container, stretch=3)

        main_widget.setLayout(main_layout)
        self._create_status_bar()

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        video_group = QGroupBox("核心监控视角")
        video_layout = QVBoxLayout()
        video_layout.setSpacing(15)

        self.video_label = QLabel("SIGNAL LOST\n等待接入监控流...")
        self.video_label.setObjectName("video_display")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.mousePressEvent = self.mouse_callback
        video_layout.addWidget(self.video_label, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.btn_load_model = QPushButton("初始化 AI 引擎")
        self.btn_load_model.setObjectName("primary_btn")
        self.btn_load_model.clicked.connect(self.load_model)

        self.btn_open = QPushButton("开启监控流")
        self.btn_open.setObjectName("primary_btn")
        self.btn_open.clicked.connect(self.open_video)

        self.btn_stop = QPushButton("中断连接")
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_load_model)
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_stop)

        video_layout.addLayout(btn_layout)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        bottom_controls_layout = QHBoxLayout()
        bottom_controls_layout.setSpacing(15)

        params_group = QGroupBox("系统参数实时调节")
        params_layout = QHBoxLayout()
        params_layout.setContentsMargins(15, 20, 15, 15)
        params_layout.setSpacing(20)

        parking_layout = QVBoxLayout()
        parking_layout.setSpacing(8)
        lbl_parking = QLabel("违停报警阈值 (秒):")
        lbl_parking.setStyleSheet("color: #9ca3af; font-size: 13px;")
        self.spin_parking = QSpinBox()
        self.spin_parking.setRange(5, 600)
        self.spin_parking.setValue(60)
        self.spin_parking.valueChanged.connect(self.update_parking_threshold)
        parking_layout.addWidget(lbl_parking)
        parking_layout.addWidget(self.spin_parking)

        movement_layout = QVBoxLayout()
        movement_layout.setSpacing(8)
        lbl_movement = QLabel("位移容忍阈值 (像素):")
        lbl_movement.setStyleSheet("color: #9ca3af; font-size: 13px;")
        self.spin_movement = QSpinBox()
        self.spin_movement.setRange(10, 200)
        self.spin_movement.setValue(60)
        self.spin_movement.valueChanged.connect(self.update_movement_threshold)
        movement_layout.addWidget(lbl_movement)
        movement_layout.addWidget(self.spin_movement)

        params_layout.addLayout(parking_layout)
        params_layout.addLayout(movement_layout)
        params_group.setLayout(params_layout)

        roi_group = QGroupBox("智能布控区域设置")
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(15, 20, 15, 15)
        roi_layout.setSpacing(10)

        self.btn_draw_roi = QPushButton("绘制警戒多边形")
        self.btn_draw_roi.clicked.connect(self.start_drawing)
        self.btn_clear_roi = QPushButton("🗑清除当前配置")
        self.btn_clear_roi.clicked.connect(self.clear_roi)

        roi_layout.addWidget(self.btn_draw_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        roi_group.setLayout(roi_layout)

        bottom_controls_layout.addWidget(params_group, stretch=6)
        bottom_controls_layout.addWidget(roi_group, stretch=4)

        layout.addLayout(bottom_controls_layout)
        panel.setLayout(layout)
        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()

        alert_tab = QWidget()
        alert_layout = QVBoxLayout()
        alert_layout.setContentsMargins(10, 15, 10, 10)

        alert_title = QLabel("实时违规动态抓拍日志")
        alert_title.setStyleSheet("color: #fca5a5; font-weight: bold; font-size: 15px; margin-bottom: 5px;")
        alert_layout.addWidget(alert_title)

        self.alert_list = QListWidget()
        alert_layout.addWidget(self.alert_list)
        alert_tab.setLayout(alert_layout)
        self.tab_widget.addTab(alert_tab, "告警日志")

        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(10, 15, 10, 10)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["监控指标", "实时数据"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setShowGrid(False)
        stats_layout.addWidget(self.stats_table)
        stats_tab.setLayout(stats_layout)
        self.tab_widget.addTab(stats_tab, "数据看板")

        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("系统初始化完成 | 请先加载模型")
        self.status_bar.addWidget(self.status_label, 1)
        self.violation_count_label = QLabel("违规次数: 0")
        self.status_bar.addPermanentWidget(self.violation_count_label)

    def update_parking_threshold(self, value):
        if self.monitor:
            try:
                self.monitor.set_parking_threshold(value)
                self.status_label.setText(f"违停报警阈值已更新为: {value} 秒")
            except AttributeError:
                pass

    def update_movement_threshold(self, value):
        if self.monitor:
            try:
                self.monitor.set_movement_threshold(value)
                self.status_label.setText(f"位移容忍阈值已更新为: {value} 像素")
            except AttributeError:
                pass

    def load_model(self):
        try:
            self.status_label.setText("正在加载双引擎级联模型 (YOLOv10 + YOLOv8 + HyperLPR3)...")
            QApplication.processEvents()

            from traffic_monitor import TrafficMonitor

            # ========================================================
            # 【核心修复】：挂载两个分离的物理权重模型，且强行开启识别开关
            # ========================================================
            VEHICLE_MODEL = r"C:\Users\86153\ML_Projects\day01\runs\detect\train\weights\best.pt"
            PLATE_MODEL = r"C:\Users\86153\ML_Projects\day01\runs\detect\train\weight\best.pt"

            config = {
                'parking_threshold': self.spin_parking.value(),
                'movement_threshold': self.spin_movement.value(),
                'conf_threshold': 0.4,
                'enable_enhancement': False,
                'enable_plate_recognition': True,  # 【修复】：必须设为 True 以挂载 OCR
                'ocr_interval': 0.5,  # 寻优频率控制
                'max_age': 100,
                'max_cosine_distance': 0.5,
                'n_init': 3,
                'skip_frames': 2
            }

            self.monitor = TrafficMonitor(
                model_path=VEHICLE_MODEL,
                plate_weights_path=PLATE_MODEL,
                db_manager=self.db,
                config=config
            )

            self.model_loaded = True
            self.status_label.setText("AI 双引擎加载完成，系统就绪")

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
        self.timer.start(5)
        self.status_label.setText("监控流解析中...")

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop.setEnabled(False)
        self.status_label.setText("监控已中断")

    def update_frame(self):
        if not self.cap or not self.monitor:
            return

        self.timer.stop()

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        try:
            # =========================================================
            # 🚨 终极修复：绝对不能在这里把 frame 压缩成 640x480！
            # 必须把原汁原味的高清原图送给底层大模型，保留物理像素细节！
            # =========================================================
            processed_frame = self.monitor.process_frame(frame)

            if processed_frame is None:
                if self.cap: self.timer.start(1)
                return

            # 如果用户正在画 ROI（处理坐标换算需要对齐你的界面比例）
            if self.drawing_roi and len(self.roi_points) > 0:
                for i, p in enumerate(self.roi_points):
                    cv2.circle(processed_frame, p, 5, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(processed_frame, self.roi_points[i - 1], p, (0, 255, 0), 2)

            # =========================================================
            # ✅ 只在最后推流给 UI 界面显示的时候，才压缩画面大小！
            # 这不会影响 AI 刚才已经处理好的高清识别结果
            # =========================================================
            display_frame = cv2.resize(processed_frame, (640, 480))

            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

            if getattr(self, 'frame_count', 0) % 30 == 0:
                self.refresh_ui_data()
            self.frame_count = getattr(self, 'frame_count', 0) + 1

        except Exception as e:
            print(f"处理出错: {e}")

        if self.cap:
            self.timer.start(1)

    def mouse_callback(self, event):
        if self.drawing_roi:
            ux, uy = event.pos().x(), event.pos().y()
            lw, lh = self.video_label.width(), self.video_label.height()
            target_w, target_h = 640, 480

            real_x = int(round((ux / lw) * target_w))
            real_y = int(round((uy / lh) * target_h))

            real_x = max(0, min(real_x, target_w - 1))
            real_y = max(0, min(real_y, target_h - 1))

            self.roi_points.append((real_x, real_y))
            self.status_label.setText(f"ROI打点 {len(self.roi_points)}: 坐标({real_x}, {real_y})")

    def start_drawing(self):
        if self.drawing_roi:
            if len(self.roi_points) >= 3:
                if self.monitor:
                    self.monitor.set_roi(self.roi_points)
                self.drawing_roi = False
                self.btn_draw_roi.setText("重新绘制警戒区")
                self.status_label.setText(f"警戒区域设置完毕: 共 {len(self.roi_points)} 个控制点")
            else:
                QMessageBox.warning(self, "提示", "至少需要3个点才能构成多边形!")
        else:
            self.drawing_roi = True
            self.roi_points = []
            if self.monitor:
                self.monitor.set_roi([])
            self.btn_draw_roi.setText("确认区域闭合")
            self.status_label.setText("请在上方视频画面内点击鼠标，绘制不规则警戒多边形...")

    def clear_roi(self):
        self.roi_points = []
        if self.monitor:
            self.monitor.set_roi([])
        self.drawing_roi = False
        self.btn_draw_roi.setText("绘制警戒多边形")
        self.status_label.setText("🗑警戒区域配置已重置")

    def load_roi_config(self):
        points = self.db.load_roi_config()
        if points and len(points) >= 3:
            self.roi_points = points
            self.status_label.setText(f"已加载历史ROI配置: {len(points)} 个控制点")

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