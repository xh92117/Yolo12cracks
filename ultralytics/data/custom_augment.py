# custom_augment.py
import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from ultralytics.utils import LOGGER  # 集成YOLO日志系统


class CustomAugment:
    def __init__(self,
                 p=0.5,
                 black_thresh=0.05,
                 white_thresh=0.1,
                 enhance_intensity=0.4,
                 smooth_sigma=5):
        """
        完整参数说明：
        :param p: 增强触发概率 (0-1)
        :param black_thresh: 黑区阈值比例 (0-1)
        :param white_thresh: 白区阈值比例 (0-1)
        :param enhance_intensity: 基础增强强度
        :param smooth_sigma: 直方图平滑系数
        """
        self.p = p
        self.black_thresh = black_thresh
        self.white_thresh = white_thresh
        self.enhance_intensity = enhance_intensity
        self.smooth_sigma = smooth_sigma

    def __call__(self, labels):
        """ YOLOv8 标准接口 """
        # 概率过滤
        if random.random() > self.p:
            return labels

        # 提取图像并备份
        img = labels['img'].copy()
        h, w = img.shape[:2]

        try:
            # 执行核心算法
            enhanced = self._tanh_hist_equalization(img)

            # 通道验证
            if enhanced.ndim == 2:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            # 尺寸验证
            if enhanced.shape != img.shape:
                enhanced = cv2.resize(enhanced, (w, h))

            # 更新图像
            labels['img'] = enhanced.astype(np.uint8)

        except Exception as e:
            LOGGER.warning(f'Custom augmentation failed: {e}')
            labels['img'] = img  # 回退原始图像

        return labels

    def _tanh_hist_equalization(self, img):
        """ 核心增强逻辑（保持原有算法）"""
        # 转换为灰度处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 动态Canny阈值
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_low = max(0, int(otsu_thresh * 0.4))
        canny_high = min(255, int(otsu_thresh * 1.6))
        edges = cv2.Canny(gray, canny_low, canny_high)

        # 自适应形态学核
        min_dim = min(gray.shape)
        kernel_size = max(3, int(min_dim / 100))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 基础增强
        enhanced = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        # 直方图分析
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        smoothed_hist = gaussian_filter(hist, sigma=2)
        peaks, _ = find_peaks(smoothed_hist, prominence=np.max(smoothed_hist) * 0.1)

        # 动态强度计算
        mean_val = np.mean(gray)
        main_peak = peaks[np.argmax(smoothed_hist[peaks])] if len(peaks) > 0 else 128
        hist_skew = (mean_val - main_peak) / 255
        dynamic_intensity = 0.3 + 0.5 * abs(hist_skew)

        # Tanh映射
        x = np.linspace(0, 255, 256)
        mapped = 255 * (np.tanh((x - main_peak) / 128) + 1) / 2
        mapped = np.clip(mapped * dynamic_intensity + x * (1 - dynamic_intensity), 0, 255)

        # 应用LUT
        enhanced = cv2.LUT(enhanced, mapped.astype(np.uint8))

        # 边缘融合
        edge_strength = cv2.Sobel(enhanced, cv2.CV_64F, 1, 1, ksize=3)
        edge_weight = np.clip(cv2.normalize(edge_strength, None, 0, 1, cv2.NORM_MINMAX), 0, 1)
        final = cv2.addWeighted(gray, 0.3, enhanced, 0.7, 0)
        final = (final * (1 - edge_weight) + enhanced * edge_weight).astype(np.uint8)

        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=3.0 + 2 * hist_skew, tileGridSize=(8, 8))
        final = clahe.apply(final)

        return final
