import cv2
import numpy as np


class ImageEnhancer:
    """
    图像增强处理器 - 适应复杂光照环境
    用于处理夜间低照度、雨天、雾天等复杂场景
    """

    def __init__(self):
        self.enabled = True
        self.denoise_strength = 10
        self.clahe_clip_limit = 2.0

    def denoise(self, frame, strength=None):
        """
        去噪处理 - 使用非局部均值去噪
        Args:
            frame: 输入图像
            strength: 去噪强度 (默认使用self.denoise_strength)
        """
        if strength is None:
            strength = self.denoise_strength
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)

    def adjust_brightness(self, frame, gamma=1.0):
        """
        Gamma校正 - 亮度调节
        Args:
            frame: 输入图像
            gamma: Gamma值 (>1变暗, <1变亮)
        """
        if gamma <= 0:
            return frame
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def auto_enhance(self, frame):
        """
        自动增强 - 根据图像亮度自适应调整
        自动判断场景并应用相应的增强策略
        """
        if not self.enabled:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        if avg_brightness < 50:
            return self.adjust_brightness(frame, gamma=1.8)
        elif avg_brightness < 80:
            return self.adjust_brightness(frame, gamma=1.4)
        elif avg_brightness > 200:
            return self.adjust_brightness(frame, gamma=0.7)
        elif avg_brightness > 180:
            return self.adjust_brightness(frame, gamma=0.85)
        return frame

    def clahe_enhance(self, frame):
        """
        CLAHE - 对比度受限自适应直方图均衡化
        增强图像对比度，同时避免过度增强噪声
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def dehaze(self, frame, omega=0.95, t0=0.1):
        """
        简单去雾算法 - 基于暗通道先验
        Args:
            frame: 输入图像
            omega: 去雾强度 (0-1)
            t0: 最小透射率
        """
        if not self.enabled:
            return frame

        min_channel = np.min(frame, axis=2)
        dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        atmospheric_light = np.max(dark_channel)

        transmission = 1 - omega * (dark_channel.astype(float) / atmospheric_light)
        transmission = np.maximum(transmission, t0)
        transmission = np.stack([transmission] * 3, axis=2)

        result = (frame.astype(float) - atmospheric_light) / transmission + atmospheric_light
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def enhance_for_detection(self, frame, scene_type='auto'):
        """
        针对目标检测优化的增强流程
        Args:
            frame: 输入图像
            scene_type: 场景类型 ('auto', 'night', 'foggy', 'rainy', 'normal')
        """
        if not self.enabled:
            return frame

        if scene_type == 'auto':
            scene_type = self._detect_scene(frame)

        if scene_type == 'night':
            enhanced = self.adjust_brightness(frame, gamma=1.5)
            enhanced = self.clahe_enhance(enhanced)
            enhanced = self.denoise(enhanced, strength=8)
        elif scene_type == 'foggy':
            enhanced = self.dehaze(frame)
            enhanced = self.clahe_enhance(enhanced)
        elif scene_type == 'rainy':
            enhanced = self.denoise(frame, strength=12)
            enhanced = self.clahe_enhance(enhanced)
        else:
            enhanced = self.clahe_enhance(frame)

        return enhanced

    def _detect_scene(self, frame):
        """
        自动检测场景类型
        Returns: 'night', 'foggy', 'rainy', 'normal'
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        if avg_brightness < 60:
            return 'night'
        elif avg_brightness > 150 and std_brightness < 40:
            return 'foggy'
        elif std_brightness > 60:
            return 'rainy'
        else:
            return 'normal'

    def get_enhancement_info(self, frame):
        """
        获取图像增强信息 (用于UI显示)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return {
            'avg_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'scene_type': self._detect_scene(frame),
            'enhancement_enabled': self.enabled
        }


class FramePreprocessor:
    """
    视频帧预处理器 - 管理帧处理流程
    """

    def __init__(self, target_size=(800, 600), enable_enhancement=True):
        self.target_size = target_size
        self.enhancer = ImageEnhancer()
        self.enhancer.enabled = enable_enhancement
        self.frame_count = 0
        self.skip_frames = 0
        self.scene_type = 'auto'

    def process(self, frame):
        """
        处理单帧图像
        Args:
            frame: 输入帧
        Returns:
            processed_frame: 处理后的帧
            info: 处理信息字典
        """
        self.frame_count += 1

        resized = cv2.resize(frame, self.target_size)

        enhanced = self.enhancer.enhance_for_detection(resized, self.scene_type)

        info = {
            'frame_id': self.frame_count,
            'original_size': frame.shape[:2][::-1],
            'processed_size': self.target_size,
            'enhancement_info': self.enhancer.get_enhancement_info(resized)
        }

        return enhanced, info

    def set_scene_type(self, scene_type):
        """设置场景类型"""
        self.scene_type = scene_type

    def set_enhancement(self, enabled):
        """启用/禁用图像增强"""
        self.enhancer.enabled = enabled


if __name__ == "__main__":
    enhancer = ImageEnhancer()

    test_image = np.random.randint(0, 50, (600, 800, 3), dtype=np.uint8)

    print("测试夜间低照度图像增强...")
    print(f"原始平均亮度: {np.mean(test_image):.2f}")

    enhanced = enhancer.enhance_for_detection(test_image, 'night')
    print(f"增强后平均亮度: {np.mean(enhanced):.2f}")
    print("图像增强模块测试完成!")
