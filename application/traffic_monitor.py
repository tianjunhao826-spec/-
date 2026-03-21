import os
import cv2
import numpy as np
import time
import math
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from preprocessing import ImageEnhancer, FramePreprocessor
from license_plate_ocr import LicensePlateRecognizer


class VehicleInfo:
    """车辆信息类"""

    def __init__(self, track_id):
        self.track_id = track_id
        self.start_time = None
        self.last_position = None
        self.anchor_position = None
        self.duration = 0.0
        self.total_movement = 0.0
        self.plate_number = None
        self.plate_confidence = 0.0
        self.vehicle_type = 'unknown'
        self.status = 'normal'
        self.alerted = False
        self.position_history = []


class TrafficMonitor:
    """
    交通监控核心类
    实现车辆检测、追踪、违停判定、车牌识别等功能
    """

    def __init__(self, model_path, db_manager, config=None):
        """
        初始化交通监控器
        Args:
            model_path: YOLO模型路径
            db_manager: 数据库管理器
            config: 配置字典
        """
        default_config = {
            'parking_threshold': 60,
            'movement_threshold': 30,
            'conf_threshold': 0.4,
            'enable_enhancement': False,  # 默认关闭，太耗时
            'enable_plate_recognition': True,
            'max_age': 30,
            'n_init': 3,
            'max_cosine_distance': 0.3,
            'skip_frames': 2  # 跳帧处理，每2帧处理1帧
        }

        self.config = {**default_config, **(config or {})}
        self.frame_count = 0

        print(f"[Monitor] 正在加载模型: {model_path}")
        self.model = YOLO(model_path)

        self.tracker = DeepSort(
            max_age=self.config['max_age'],
            n_init=self.config['n_init'],
            max_cosine_distance=self.config['max_cosine_distance']
        )

        self.db = db_manager
        self.roi_points = []

        self.vehicles = {}
        self.alerted_ids = set()

        self.enhancer = ImageEnhancer()
        self.enhancer.enabled = self.config['enable_enhancement']

        self.preprocessor = FramePreprocessor(enable_enhancement=self.config['enable_enhancement'])

        self.plate_recognizer = None
        if self.config['enable_plate_recognition']:
            try:
                self.plate_recognizer = LicensePlateRecognizer(engine='simple')
            except Exception as e:
                print(f"[Monitor] 车牌识别模块初始化失败: {e}")

        self.class_names = {0: 'car', 1: 'bus', 2: 'van', 3: 'others'}

        self.stats = {
            'total_detections': 0,
            'total_tracks': 0,
            'total_violations': 0,
            'frames_processed': 0
        }

    def set_roi(self, points):
        """设置禁停区域"""
        self.roi_points = points
        print(f"[Monitor] ROI已设置: {len(points)} 个点")

    def is_point_in_roi(self, point):
        """判断点是否在ROI区域内"""
        if not self.roi_points or len(self.roi_points) < 3:
            return False
        return cv2.pointPolygonTest(
            np.array(self.roi_points, dtype=np.int32),
            point,
            False
        ) >= 0

    def process_frame(self, frame):
        """
        处理单帧图像
        Args:
            frame: 输入帧
        Returns:
            processed_frame: 处理后的帧
        """
        self.frame_count += 1
        self.stats['frames_processed'] += 1
        current_time = time.time()

        processed_frame = frame.copy()

        if self.enhancer.enabled and self.frame_count % 10 == 0:
            processed_frame = self.enhancer.clahe_enhance(processed_frame)

        if self.frame_count % (self.config.get('skip_frames', 2) + 1) != 0:
            self._draw_roi(processed_frame)
            for track_id, vehicle in list(self.vehicles.items()):
                if vehicle.last_position:
                    bbox = [vehicle.last_position[0] - 50, vehicle.last_position[1] - 30,
                            vehicle.last_position[0] + 50, vehicle.last_position[1] + 30]
                    self._draw_vehicle_info(processed_frame, vehicle, bbox, False)
            self._draw_stats(processed_frame)
            return processed_frame

        results = self.model(processed_frame, verbose=False)[0]

        detections = self._extract_detections(results)
        self.stats['total_detections'] += len(detections)

        tracks = self.tracker.update_tracks(detections, frame=processed_frame)

        self._draw_roi(processed_frame)

        active_track_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_track_ids.add(track_id)

            bbox = track.to_ltrb()
            center = self._get_center(bbox)
            cls_id = int(track.det_conf) if hasattr(track, 'det_conf') else 0

            in_roi = self.is_point_in_roi(center)

            vehicle = self._get_or_create_vehicle(track_id)
            vehicle.vehicle_type = self.class_names.get(cls_id, 'unknown')

            if in_roi:
                self._update_vehicle_in_roi(vehicle, center, bbox, processed_frame, current_time)
            else:
                self._reset_vehicle_state(vehicle, center)

            self._draw_vehicle_info(processed_frame, vehicle, bbox, in_roi)

        self._cleanup_inactive_vehicles(active_track_ids)

        self._draw_stats(processed_frame)

        return processed_frame

    def _extract_detections(self, results):
        """从YOLO结果中提取检测框"""
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf > self.config['conf_threshold']:
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

        return detections

    def _get_center(self, bbox):
        """计算边界框中心点"""
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def _get_or_create_vehicle(self, track_id):
        """获取或创建车辆信息"""
        if track_id not in self.vehicles:
            self.vehicles[track_id] = VehicleInfo(track_id)
            self.stats['total_tracks'] += 1
        return self.vehicles[track_id]

    def _update_vehicle_in_roi(self, vehicle, center, bbox, frame, current_time):
        """
        更新ROI内车辆的状态
        【优化版】：计算相对于初始锚点的总位移，防止将缓慢移动(堵车)误判为违停
        """
        if vehicle.start_time is None:
            vehicle.start_time = current_time
            vehicle.last_position = center
            vehicle.anchor_position = center # 设置初始锚点
            vehicle.status = 'entering'
            vehicle.position_history = [center]
        else:
            # 【修改重点】：计算当前位置与'最初锚点'的距离，而不是上一帧的距离
            total_displacement = math.sqrt(
                (center[0] - vehicle.anchor_position[0]) ** 2 +
                (center[1] - vehicle.anchor_position[1]) ** 2
            )

            vehicle.position_history.append(center)
            if len(vehicle.position_history) > 30:
                vehicle.position_history.pop(0)

            # 如果离开了最初设定的锚点范围，说明车辆发生明显移动（脱离堵车或重新起步）
            if total_displacement > self.config['movement_threshold']:
                vehicle.start_time = current_time  # 重置计时器
                vehicle.anchor_position = center   # 更新基准锚点
                vehicle.duration = 0
                vehicle.status = 'moving'
            else:
                # 车辆在锚点附近未发生大幅移动，持续累加停车时间
                vehicle.duration = current_time - vehicle.start_time
                vehicle.status = 'parking'

            vehicle.last_position = center

            if vehicle.duration >= self.config['parking_threshold']:
                vehicle.status = 'violation'

                if not vehicle.alerted:
                    self._trigger_violation_alert(vehicle, bbox, frame)
                    vehicle.alerted = True

    def _reset_vehicle_state(self, vehicle, center):
        """重置离开ROI的车辆状态"""
        vehicle.start_time = None
        vehicle.duration = 0
        vehicle.last_position = center
        vehicle.status = 'normal'
        vehicle.position_history = []

    def _trigger_violation_alert(self, vehicle, bbox, frame):
        """触发违停告警"""
        self.stats['total_violations'] += 1
        self.alerted_ids.add(vehicle.track_id)

        if self.plate_recognizer:
            plate, conf = self.plate_recognizer.recognize(frame, bbox)
            vehicle.plate_number = plate
            vehicle.plate_confidence = conf

        save_path = self._save_violation_image(frame, vehicle.track_id)

        self.db.insert_violation(
            car_id=f"Car_{vehicle.track_id}",
            image_path=save_path,
            plate_number=vehicle.plate_number,
            duration=vehicle.duration,
            vehicle_type=vehicle.vehicle_type
        )

        print(f"[ALERT] 违停告警! ID={vehicle.track_id}, "
              f"车牌={vehicle.plate_number}, 停留={vehicle.duration:.1f}秒")

    def _save_violation_image(self, frame, track_id):
        """保存违规截图"""
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)

        filename = f"violation_{track_id}_{int(time.time())}.jpg"
        save_path = os.path.join(captures_dir, filename)

        cv2.imwrite(save_path, frame)
        return save_path

    def _cleanup_inactive_vehicles(self, active_ids):
        """清理不活跃的车辆"""
        inactive_ids = [tid for tid in self.vehicles if tid not in active_ids]
        for tid in inactive_ids:
            if tid in self.vehicles:
                del self.vehicles[tid]

    def _draw_roi(self, frame):
        """绘制ROI区域"""
        if len(self.roi_points) > 2:
            pts = np.array(self.roi_points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def _draw_vehicle_info(self, frame, vehicle, bbox, in_roi):
        """绘制车辆信息"""
        x1, y1, x2, y2 = map(int, bbox)
        center = self._get_center(bbox)

        if vehicle.status == 'violation':
            color = (0, 0, 255)
        elif vehicle.status == 'parking':
            color = (0, 255, 255)
        elif in_roi:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{vehicle.track_id}"
        if vehicle.plate_number:
            label += f" | {vehicle.plate_number}"

        cv2.putText(frame, label, (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if in_roi and vehicle.duration > 0:
            time_text = f"{vehicle.duration:.1f}s"
            if vehicle.status == 'violation':
                time_text = f"VIOLATION! {time_text}"
            cv2.putText(frame, time_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.circle(frame, center, 4, (255, 0, 0), -1)

        if len(vehicle.position_history) > 1:
            for i in range(1, len(vehicle.position_history)):
                cv2.line(frame,
                         vehicle.position_history[i - 1],
                         vehicle.position_history[i],
                         (255, 255, 0), 1)

    def _draw_stats(self, frame):
        """绘制统计信息"""
        stats_text = [
            f"Frames: {self.stats['frames_processed']}",
            f"Tracks: {len(self.vehicles)}",
            f"Violations: {self.stats['total_violations']}"
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    def get_statistics(self):
        """获取统计数据"""
        return {
            **self.stats,
            'active_vehicles': len(self.vehicles),
            'roi_set': len(self.roi_points) >= 3
        }

    def set_enhancement(self, enabled):
        """设置图像增强开关"""
        self.enhancer.enabled = enabled
        self.preprocessor.set_enhancement(enabled)

    def set_parking_threshold(self, seconds):
        """设置违停时间阈值"""
        self.config['parking_threshold'] = seconds

    def set_movement_threshold(self, pixels):
        """设置位移阈值"""
        self.config['movement_threshold'] = pixels


if __name__ == "__main__":
    print("交通监控模块测试...")

    class MockDB:
        def insert_violation(self, **kwargs):
            print(f"[MockDB] 插入记录: {kwargs}")

    monitor = TrafficMonitor(
        model_path=r"C:\Users\86153\ML_Projects\day01\毕设\src\Traffic_Project\yolov10_train_v1\weights\best.pt",
        db_manager=MockDB()
    )

    print(f"配置: {monitor.config}")
    print("交通监控模块测试完成!")
