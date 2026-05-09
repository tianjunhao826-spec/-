import cv2
import numpy as np
import os
import warnings
from ultralytics import YOLO
import hyperlpr3 as lpr3



warnings.filterwarnings("ignore")
os.environ['GLOG_minloglevel'] = '3'


class LicensePlateRecognizer:
    """
    车牌识别核心模块 (重构版)
    架构: 车辆切片输入 -> YOLOv8 车牌精确定位 -> 影棚画布增强 -> HyperLPR3 识别
    """

    def __init__(self, yolo_weights_path):
        """
        初始化车牌识别流水线
        Args:
            yolo_weights_path: 用于检测车牌的 YOLOv8 模型权重路径
        """
        print("[OCR引擎] 正在加载 YOLOv8 车牌空间定位器...")
        self.plate_detector = YOLO(yolo_weights_path)

        print("[OCR引擎] 正在加载 HyperLPR3 专用识别网络...")
        self.lpr_engine = lpr3.LicensePlateCatcher()

    def bionic_enhance_and_canvas(self, plate_img):
        """
        仿生增强与影棚画布构建 (专攻远端模糊小目标)
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # 1. 亚像素重构平滑放大
        upscaled = cv2.resize(plate_img, (264, 84), interpolation=cv2.INTER_CUBIC)

        # 2. 线性对比度温和提亮
        enhanced = cv2.convertScaleAbs(upscaled, alpha=1.2, beta=10)

        # 3. 构建中性灰画布，规避边缘抑制效应
        canvas_h, canvas_w = 200, 400
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 127

        # 4. 居中融合
        start_y = (canvas_h - 84) // 2
        start_x = (canvas_w - 264) // 2
        canvas[start_y:start_y + 84, start_x:start_x + 264] = enhanced

        return canvas

    def recognize(self, frame, vehicle_bbox):
        """
        执行端到端车牌识别 (强制显存降维 + 边界补偿)
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        h_frame, w_frame = frame.shape[:2]

        # 【技术点 1：抗追踪漂移补偿】
        # 必须外扩 30 像素，防止视频掉帧时 DeepSORT 预测框滞后，切掉车头
        margin = 30
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w_frame, x2 + margin)
        y2 = min(h_frame, y2 + margin)

        vehicle_crop = frame[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            return None, 0.0

        # 【技术点 2：显存强行降维】
        # 必须显式指定 imgsz=320 (甚至可以测 256)。车辆切片很小，用 640 是纯浪费算力
        results = self.plate_detector(vehicle_crop, conf=0.3, imgsz=320, augment=False, verbose=False)

        best_text = None
        best_conf = 0.0

        for result in results:
            for box in result.boxes:
                px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)
                vh, vw = vehicle_crop.shape[:2]

                # 车牌级切片仅保留 2 像素边缘
                plate_crop = vehicle_crop[max(0, py1 - 2):min(vh, py2 + 2), max(0, px1 - 2):min(vw, px2 + 2)]

                # 仿生增强
                canvas_plate = self.bionic_enhance_and_canvas(plate_crop)

                if canvas_plate is not None:
                    # HyperLPR3 识别
                    lpr_results = self.lpr_engine(canvas_plate)

                    if lpr_results and len(lpr_results) > 0:
                        text, conf = lpr_results[0][0], lpr_results[0][1]

                        if conf > best_conf:
                            best_conf = conf
                            best_text = text

        return best_text, best_conf

    def batch_recognize(self, frame, vehicle_bboxes):
        """批量识别接口"""
        results = []
        for bbox in vehicle_bboxes:
            plate, conf = self.recognize(frame, bbox)
            results.append((plate, conf))
        return results


if __name__ == "__main__":
    # 模块独立测试入口
    print("车牌识别流水线自检...")

    # 替换为你的真实权重路径测试
    dummy_weights = r'C:\Users\86153\ML_Projects\day01\runs\detect\train\weight\best.pt'
    if os.path.exists(dummy_weights):
        recognizer = LicensePlateRecognizer(dummy_weights)
        test_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        test_bbox = [100, 100, 300, 250]

        plate, conf = recognizer.recognize(test_frame, test_bbox)
        print(f"自检完成: 车牌={plate}, 置信度={conf}")
    else:
        print("未找到权重文件，请在实际系统中挂载模型。")