import cv2
import numpy as np
import re


class LicensePlateRecognizer:
    """
    车牌识别模块
    支持多种OCR引擎: PaddleOCR, HyperLPR, 或简单的轮廓检测
    """

    def __init__(self, engine='paddle'):
        """
        初始化车牌识别器
        Args:
            engine: OCR引擎类型 ('paddle', 'hyperlpr', 'simple')
        """
        self.engine = engine
        self.ocr = None
        self._init_engine()

    def _init_engine(self):
        """初始化OCR引擎"""
        if self.engine == 'paddle':
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
                print("[OCR] PaddleOCR 初始化成功")
            except ImportError:
                print("[OCR] PaddleOCR 未安装, 尝试使用简单模式")
                self.engine = 'simple'
        elif self.engine == 'hyperlpr':
            try:
                import hyperlpr3 as lpr3
                self.recognizer = lpr3.LicensePlateCatcher()
                print("[OCR] HyperLPR 初始化成功")
            except ImportError:
                print("[OCR] HyperLPR 未安装, 尝试使用简单模式")
                self.engine = 'simple'

        if self.engine == 'simple':
            print("[OCR] 使用简单轮廓检测模式")

    def detect_plate_region(self, frame, vehicle_bbox):
        """
        从车辆边界框中提取车牌候选区域
        Args:
            frame: 原始图像
            vehicle_bbox: 车辆边界框 [x1, y1, x2, y2]
        Returns:
            plate_roi: 车牌候选区域
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        vehicle_roi = frame[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            return None

        h, w = vehicle_roi.shape[:2]

        plate_y1 = int(h * 0.5)
        plate_y2 = int(h * 0.85)
        plate_x1 = int(w * 0.1)
        plate_x2 = int(w * 0.9)

        plate_roi = vehicle_roi[plate_y1:plate_y2, plate_x1:plate_x2]

        return plate_roi

    def recognize(self, frame, vehicle_bbox):
        """
        识别车牌号码
        Args:
            frame: 原始图像
            vehicle_bbox: 车辆边界框
        Returns:
            plate_number: 车牌号码 (识别失败返回None)
            confidence: 置信度
        """
        plate_roi = self.detect_plate_region(frame, vehicle_bbox)

        if plate_roi is None or plate_roi.size == 0:
            return None, 0.0

        if self.engine == 'paddle':
            return self._recognize_paddle(plate_roi)
        elif self.engine == 'hyperlpr':
            return self._recognize_hyperlpr(plate_roi)
        else:
            return self._recognize_simple(plate_roi)

    def _recognize_paddle(self, plate_roi):
        """使用PaddleOCR识别"""
        try:
            result = self.ocr.ocr(plate_roi, cls=True)
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    if self._is_valid_plate(text):
                        return self._format_plate(text), conf
        except Exception as e:
            print(f"[OCR] PaddleOCR识别错误: {e}")
        return None, 0.0

    def _recognize_hyperlpr(self, plate_roi):
        """使用HyperLPR识别"""
        try:
            results = self.recognizer(plate_roi)
            if results:
                for plate_info in results:
                    plate_number, conf, plate_type = plate_info
                    if self._is_valid_plate(plate_number):
                        return plate_number, float(conf)
        except Exception as e:
            print(f"[OCR] HyperLPR识别错误: {e}")
        return None, 0.0

    def _recognize_simple(self, plate_roi):
        """简单模式 - 仅返回占位符"""
        return "未识别", 0.0

    def _is_valid_plate(self, text):
        """
        验证是否为有效车牌格式
        支持普通车牌和新能源车牌
        """
        text = text.replace('·', '').replace('-', '').replace(' ', '').upper()

        patterns = [
            r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5}$',
            r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{6}$',
            r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9]{5,6}$',
        ]

        for pattern in patterns:
            if re.match(pattern, text):
                return True
        return False

    def _format_plate(self, text):
        """格式化车牌号"""
        text = text.replace('·', '').replace('-', '').replace(' ', '').upper()
        return text

    def batch_recognize(self, frame, vehicle_bboxes):
        """
        批量识别多个车辆的车牌
        Args:
            frame: 原始图像
            vehicle_bboxes: 车辆边界框列表
        Returns:
            results: [(plate_number, confidence), ...]
        """
        results = []
        for bbox in vehicle_bboxes:
            plate, conf = self.recognize(frame, bbox)
            results.append((plate, conf))
        return results


class PlateDetector:
    """
    车牌检测器 - 使用OpenCV进行车牌定位
    作为OCR的前置步骤
    """

    def __init__(self):
        self.min_plate_ratio = 2.0
        self.max_plate_ratio = 6.0
        self.min_area = 500

    def detect_plates(self, frame):
        """
        检测图像中的车牌区域
        Returns:
            plates: [(x, y, w, h, plate_image), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plates = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                aspect_ratio = float(w) / h if h > 0 else 0
                area = w * h

                if (self.min_plate_ratio < aspect_ratio < self.max_plate_ratio and
                        area > self.min_area):
                    plate_img = frame[y:y + h, x:x + w]
                    plates.append((x, y, w, h, plate_img))

        return plates


if __name__ == "__main__":
    print("车牌识别模块测试...")

    recognizer = LicensePlateRecognizer(engine='simple')
    print(f"使用引擎: {recognizer.engine}")

    test_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    test_bbox = [100, 100, 300, 250]

    plate, conf = recognizer.recognize(test_frame, test_bbox)
    print(f"测试结果: 车牌={plate}, 置信度={conf}")

    print("车牌识别模块测试完成!")
