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
                # 【核心修复】：移除了旧版的 show_log 和 use_angle_cls，使用最新 API
                self.ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
                print("[OCR] PaddleOCR 底层大模型初始化成功！")
            except ImportError:
                print("[OCR] ❌ PaddleOCR 未安装, 尝试使用简单模式")
                self.engine = 'simple'
            except Exception as e:
                # 【核心修复】：增加全局异常捕获，防止 API 变化导致模块直接猝死
                print(f"[OCR] 🚨 PaddleOCR 启动时发生异常: {e}")
                print("[OCR] ⚠️ 将降级使用简单模式")
                self.engine = 'simple'

        elif self.engine == 'hyperlpr':
            try:
                import hyperlpr3 as lpr3
                self.recognizer = lpr3.LicensePlateCatcher()
                print("[OCR] HyperLPR 初始化成功")
            except Exception as e:
                print(f"[OCR] HyperLPR 初始化失败: {e}, 降级使用简单模式")
                self.engine = 'simple'

        if self.engine == 'simple':
            print("[OCR] 使用简单轮廓检测模式 (仅占位，无真实识别能力)")

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

    def enhance_plate(self, plate_img):
        """
        【毕设创新点：车牌局部图像增强】
        针对夜间、低分辨率、反光车牌进行强化
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img

        # 1. 线性插值放大两倍 (解决远距离车牌像素过低，OCR无法提取特征的问题)
        plate_img = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 2. 转换为灰度图
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE (对比度受限自适应直方图均衡化) - 解决局部反光和过暗问题
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced_gray = clahe.apply(gray)

        # 4. 转回 BGR 格式以兼容 PaddleOCR 和 HyperLPR
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def recognize(self, frame, vehicle_bbox):
        """
        识别车牌号码 (入口Debug打桩版)
        """
        print(f"\n[OCR-总调度] 1. 收到违停车辆！当前激活的引擎是: '{self.engine}'")
        print(f"[OCR-总调度] 2. 传入的车辆坐标(bbox): {vehicle_bbox}")

        # 获取车牌初步切片
        plate_roi = self.detect_plate_region(frame, vehicle_bbox)

        if plate_roi is None or plate_roi.size == 0:
            print("[OCR-总调度] 3. ❌ 提前终止！车辆切片失败 (ROI为空)，可能是车辆超出画面边界。")
            return None, 0.0

        print(f"[OCR-总调度] 4. 车牌区域切片成功！准备移交给 {self.engine} 引擎处理...")

        if self.engine == 'paddle':
            return self._recognize_paddle(plate_roi)
        elif self.engine == 'hyperlpr':
            return self._recognize_hyperlpr(plate_roi)
        else:
            return self._recognize_simple(plate_roi)

    def _recognize_paddle(self, plate_roi):
        """使用PaddleOCR识别 (深度Debug打桩版)"""
        import cv2
        import time
        import os

        # 【Debug 1：视觉截获】把即将送进 OCR 的那块像素保存下来，看看切得对不对
        debug_dir = "debug_crops"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # 使用时间戳命名，防止覆盖
        timestamp = int(time.time() * 1000)
        debug_filename = os.path.join(debug_dir, f"crop_{timestamp}.jpg")
        cv2.imwrite(debug_filename, plate_roi)
        print(f"\n[OCR-拦截] =========================================")
        print(f"[OCR-拦截] 1. 已保存待识别图像切片至: {debug_filename}")

        try:
            print("[OCR-拦截] 2. 正在调用底层 Paddle 模型推理...")
            # 确保这里没有 cls=True
            result = self.ocr.ocr(plate_roi)

            # 【Debug 2：张量截获】无修饰地打印模型吐出来的最原始数据
            print(f"[OCR-拦截] 3. 模型原始输出张量/列表: {result}")

            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    print(f"[OCR-拦截] 4. 物理层解析文本: '{text}' | 置信度: {conf:.2f}")

                    # 【Debug 3：逻辑层截获】查看正则是否把正确的字给杀了
                    if self._is_valid_plate(text):
                        final_plate = self._format_plate(text)
                        print(f"[OCR-拦截] 5. ✅ 正则校验通过! 提交数据库的车牌: {final_plate}")
                        print(f"[OCR-拦截] =========================================\n")
                        return final_plate, conf
                    else:
                        print(f"[OCR-拦截] 5. ❌ 触发正则熔断！字符串 '{text}' 不符合车牌格式被强行丢弃。")
            else:
                print("[OCR-拦截] 4. ⚠️ 结论：大模型在此切片中未发现任何文字特征。")

            print(f"[OCR-拦截] =========================================\n")

        except Exception as e:
            print(f"[OCR-拦截] 🚨 发生底层崩溃/异常: {e}")
            print(f"[OCR-拦截] =========================================\n")

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
        text = text.replace('·', '').replace('-', '').replace(' ', '').upper()
        # 去掉正则前后的 ^ 和 $
        pattern = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5,6}'
        if re.search(pattern, text):
            return True
        return False

    def _format_plate(self, text):
        text = text.replace('·', '').replace('-', '').replace(' ', '').upper()
        pattern = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5,6}'
        match = re.search(pattern, text)
        if match:
            return match.group(0) # 只提取车牌部分，剥离多余的噪点字符
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
