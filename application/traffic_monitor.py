import os
import cv2
import numpy as np
import time
import math
import warnings
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from preprocessing import ImageEnhancer, FramePreprocessor
from license_plate_ocr import LicensePlateRecognizer


# ================= 核心工具：中文无损渲染 =================
def cv2_put_text_chinese(img, text, position, text_color=(0, 255, 0), text_size=20):
    """解决 OpenCV cv2.putText 无法绘制中文导致的 ??? 乱码问题"""
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img

    draw = ImageDraw.Draw(img_pil)
    font_style = None
    # 按优先级加载系统中文字体，请确保项目根目录下有 simhei.ttf
    font_paths = ["simhei.ttf", "msyh.ttc", "simsun.ttc", "Arial Unicode.ttf"]
    for path in font_paths:
        try:
            font_style = ImageFont.truetype(path, text_size, encoding="utf-8")
            break
        except IOError:
            continue

    if font_style is None:
        font_style = ImageFont.load_default()

    draw.text(position, text, fill=text_color, font=font_style)
    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


class VehicleInfo:
    def __init__(self, track_id):
        self.track_id = track_id
        self.start_time = None
        self.last_position = None
        self.anchor_position = None
        self.duration = 0.0
        self.total_movement = 0.0
        self.vehicle_type = 'unknown'
        self.status = 'normal'
        self.alerted = False
        self.position_history = []

        # ================= 生命期缓存字段 =================
        self.plate_number = None  # 记录该车生命周期内的最佳车牌
        self.plate_confidence = 0.0  # 记录对应的最高置信度
        self.last_ocr_time = 0.0  # 记录上次 OCR 的时间戳 (用于控制频率)


class TrafficMonitor:
    """
    交通监控核心类
    实现车辆检测、追踪、违停判定、生命周期车牌寻优
    """

    def __init__(self, model_path, plate_weights_path, db_manager, config=None):
        """
        初始化交通监控器
        Args:
            model_path: 车辆检测 YOLO 模型路径
            plate_weights_path: 车牌定位 YOLO 模型路径
            db_manager: 数据库管理器
        """
        default_config = {
            'parking_threshold': 60,
            'movement_threshold': 30,
            'conf_threshold': 0.2,
            'enable_enhancement': False,
            'enable_plate_recognition': True,
            'max_age': 50,
            'n_init': 3,
            'max_cosine_distance': 0.3,
            'skip_frames': 0,
            'ocr_interval': 0.5  # 新增：OCR 扫描的冷却时间(秒)
        }

        self.config = {**default_config, **(config or {})}
        self.frame_count = 0

        print(f"[Monitor] 正在加载车辆检测模型: {model_path}")
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
                # 挂载最新的两阶段 OCR 识别引擎
                self.plate_recognizer = LicensePlateRecognizer(yolo_weights_path=plate_weights_path)
                print("[Monitor] 级联车牌识别引擎初始化成功")
            except Exception as e:
                print(f"[Monitor] 车牌识别模块初始化失败: {e}")

        self.class_names = {0: 'car', 1: 'bus', 2: 'van', 3: 'others'}
        self.stats = {'total_detections': 0, 'total_tracks': 0, 'total_violations': 0, 'frames_processed': 0}

    def set_roi(self, points):
        self.roi_points = points
        print(f"[Monitor] ROI已设置: {len(points)} 个点")

    def is_point_in_roi(self, point):
        if not self.roi_points or len(self.roi_points) < 3:
            return False
        return cv2.pointPolygonTest(np.array(self.roi_points, dtype=np.int32), point, False) >= 0

    def process_frame(self, frame):
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
                    processed_frame = self._draw_vehicle_info(processed_frame, vehicle, bbox, False)
            self._draw_stats(processed_frame)
            return processed_frame

        results = self.model(processed_frame, verbose=False)[0]
        detections = self._extract_detections(results)
        self.stats['total_detections'] += len(detections)

        tracks = self.tracker.update_tracks(detections, frame=processed_frame)
        self._draw_roi(processed_frame)

        active_track_ids = set()

        # =========================================================
        # 【算力平滑锁】：初始化为 False
        # 记录当前这一帧的 33 毫秒内，是否已经有车辆占用了 GPU 去跑 OCR
        # =========================================================
        ocr_done_this_frame = False

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_track_ids.add(track_id)

            if track.time_since_update > 0:
                continue

            bbox = track.to_ltrb()
            center = self._get_center(bbox)
            cls_id = int(getattr(track, 'det_conf', 0)) if getattr(track, 'det_conf', 0) is not None else 0

            vehicle = self._get_or_create_vehicle(track_id)
            vehicle.vehicle_type = self.class_names.get(cls_id, 'unknown')

            # =========================================================
            # 【生命周期寻优逻辑（加入负载均衡与熔断限制）】
            # =========================================================
            if self.plate_recognizer and (current_time - vehicle.last_ocr_time > self.config['ocr_interval']):

                # 条件 1：置信度低于 0.85 (未达到完美识别)
                # 条件 2：历史识别次数不到 6 次 (防止死磕模糊车牌)
                # 条件 3：not ocr_done_this_frame (这一帧还没其他车用过 GPU)
                if vehicle.plate_confidence < 0.85 and getattr(vehicle, 'ocr_attempts',
                                                               0) < 6 and not ocr_done_this_frame:
                    # 增加历史尝试次数
                    vehicle.ocr_attempts = getattr(vehicle, 'ocr_attempts', 0) + 1

                    plate, conf = self.plate_recognizer.recognize(processed_frame, bbox)
                    vehicle.last_ocr_time = current_time  # 刷新冷却时间

                    # 锁上大门，这一帧剩余的其他车辆不许再跑 OCR
                    ocr_done_this_frame = True

                    if plate and conf > vehicle.plate_confidence:
                        vehicle.plate_number = plate
                        vehicle.plate_confidence = conf
                        print(
                            f"[寻优进行中] ID:{track_id} 暂存更优车牌: {plate} (置信度 {conf:.2f}) | 尝试次数: {vehicle.ocr_attempts}/6")

                # 如果已经到达 6 次，触发熔断（不再调用模型，放过显卡）
                elif getattr(vehicle, 'ocr_attempts', 0) == 6:
                    vehicle.ocr_attempts += 1  # 加 1 防止这句话被重复打印
                    print(f"[寻优熔断锁定] ID:{track_id} 已达最大尝试次数，强制锁定最终车牌: {vehicle.plate_number}")

            in_roi = self.is_point_in_roi(center)
            if in_roi:
                self._update_vehicle_in_roi(vehicle, center, bbox, processed_frame, current_time)
            else:
                self._reset_vehicle_state(vehicle, center)

            processed_frame = self._draw_vehicle_info(processed_frame, vehicle, bbox, in_roi)

        # 清理驶出画面的车辆释放内存
        self._cleanup_inactive_vehicles(active_track_ids)
        self._draw_stats(processed_frame)

        return processed_frame

    def _extract_detections(self, results):
        detections = []
        if results is None or results.boxes is None:
            return detections
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy() if box.xyxy is not None else None
            conf_val = box.conf[0].cpu().numpy() if box.conf is not None else None
            cls_val = box.cls[0].cpu().numpy() if box.cls is not None else None

            if xyxy is not None and conf_val is not None and cls_val is not None:
                x1, y1, x2, y2 = xyxy
                conf = float(conf_val)
                cls_id = int(cls_val)

                if conf > self.config['conf_threshold']:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
        return detections

    def _get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def _get_or_create_vehicle(self, track_id):
        if track_id not in self.vehicles:
            self.vehicles[track_id] = VehicleInfo(track_id)
            self.stats['total_tracks'] += 1
        return self.vehicles[track_id]

    def _update_vehicle_in_roi(self, vehicle, center, bbox, frame, current_time):
        if vehicle.start_time is None:
            vehicle.start_time = current_time
            vehicle.last_position = center
            vehicle.anchor_position = center
            vehicle.status = 'entering'
            vehicle.position_history = [center]
        else:
            total_displacement = math.sqrt(
                (center[0] - vehicle.anchor_position[0]) ** 2 +
                (center[1] - vehicle.anchor_position[1]) ** 2
            )

            vehicle.position_history.append(center)
            if len(vehicle.position_history) > 30:
                vehicle.position_history.pop(0)

            if total_displacement > self.config['movement_threshold']:
                vehicle.start_time = current_time
                vehicle.anchor_position = center
                vehicle.duration = 0
                vehicle.status = 'moving'
            else:
                vehicle.duration = current_time - vehicle.start_time
                vehicle.status = 'parking'

            vehicle.last_position = center

            if vehicle.duration >= self.config['parking_threshold']:
                vehicle.status = 'violation'
                if not vehicle.alerted:
                    # 触发违停时直接提取缓存，无需再次运行 OCR
                    self._trigger_violation_alert(vehicle, frame)
                    vehicle.alerted = True

    def _reset_vehicle_state(self, vehicle, center):
        vehicle.start_time = None
        vehicle.duration = 0
        vehicle.last_position = center
        vehicle.status = 'normal'
        vehicle.position_history = []

    def _trigger_violation_alert(self, vehicle, frame):
        """触发违停告警，调用生命周期中记录的最高置信度车牌"""
        self.stats['total_violations'] += 1
        self.alerted_ids.add(vehicle.track_id)

        # 核心：直接提取最佳记录
        final_plate = vehicle.plate_number if vehicle.plate_number else "未知车牌(识别失败)"
        save_path = self._save_violation_image(frame, vehicle.track_id)

        self.db.insert_violation(
            car_id=f"Car_{vehicle.track_id}",
            image_path=save_path,
            plate_number=final_plate,
            duration=vehicle.duration,
            vehicle_type=vehicle.vehicle_type
        )

        print(f"[ALERT] 违停告警! ID={vehicle.track_id}, 车牌={final_plate}, 停留={vehicle.duration:.1f}秒")

    def _save_violation_image(self, frame, track_id):
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
        filename = f"violation_{track_id}_{int(time.time())}.jpg"
        save_path = os.path.join(captures_dir, filename)
        cv2.imwrite(save_path, frame)
        return save_path

    def _cleanup_inactive_vehicles(self, active_ids):
        """清理驶离画面车辆，利用 del 实现内存自动回收"""
        inactive_ids = [tid for tid in self.vehicles if tid not in active_ids]
        for tid in inactive_ids:
            if tid in self.vehicles:
                del self.vehicles[tid]

    def _draw_roi(self, frame):
        if len(self.roi_points) > 2:
            pts = np.array(self.roi_points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def _draw_vehicle_info(self, frame, vehicle, bbox, in_roi):
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

        # 核心：使用 PIL 替代 cv2.putText 绘制完美中文
        frame = cv2_put_text_chinese(frame, label, (x1, max(0, y1 - 25)), text_color=color, text_size=20)

        if in_roi and vehicle.duration > 0:
            time_text = f"{vehicle.duration:.1f}s"
            if vehicle.status == 'violation':
                time_text = f"VIOLATION! {time_text}"
            frame = cv2_put_text_chinese(frame, time_text, (x1, max(0, y1 - 45)), text_color=color, text_size=18)

        cv2.circle(frame, center, 4, (255, 0, 0), -1)

        if len(vehicle.position_history) > 1:
            for i in range(1, len(vehicle.position_history)):
                cv2.line(frame, vehicle.position_history[i - 1], vehicle.position_history[i], (255, 255, 0), 1)
        return frame

    def _draw_stats(self, frame):
        stats_text = [
            f"Frames: {self.stats['frames_processed']}",
            f"Tracks: {len(self.vehicles)}",
            f"Violations: {self.stats['total_violations']}"
        ]
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    def get_statistics(self):
        return {
            **self.stats,
            'active_vehicles': len(self.vehicles),
            'roi_set': len(self.roi_points) >= 3
        }

    def set_enhancement(self, enabled):
        self.enhancer.enabled = enabled
        self.preprocessor.set_enhancement(enabled)

    def set_parking_threshold(self, seconds):
        self.config['parking_threshold'] = seconds

    def set_movement_threshold(self, pixels):
        self.config['movement_threshold'] = pixels


if __name__ == "__main__":
    print("交通监控模块测试...")


    class MockDB:
        def insert_violation(self, **kwargs):
            print(f"[MockDB] 插入记录: {kwargs}")


    # 注意初始化时的参数变更，需传入车辆模型与车牌模型两个路径
    monitor = TrafficMonitor(
        model_path=r"C:\Users\86153\ML_Projects\day01\毕设\src\Traffic_Project\yolov10_train_v1\weights\best.pt",
        plate_weights_path=r"C:\Users\86153\ML_Projects\day01\runs\detect\train\weight\best.pt",
        db_manager=MockDB()
    )

    print(f"配置: {monitor.config}")
    print("交通监控模块测试完成!")