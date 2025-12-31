#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Tuple

BLACK_LEVEL = 9.0

def generate_map() -> np.ndarray:
    # 读取16bit PNG图像
    lsc_img = cv2.imread('5000k_lsc.png', cv2.IMREAD_UNCHANGED)
    if lsc_img is None:
        raise ValueError("Failed to load '5000k_lsc.png'")

    # 假设图像是单通道16bit Bayer图案
    if lsc_img.dtype != np.uint16:
        raise ValueError("Image is not 16-bit")

    # 黑电平9.0 * 256 = 2304
    black_level = BLACK_LEVEL * 256
    # 在uint16上减黑电平，使用int32避免underflow，然后clip
    lsc_corrected = np.clip(lsc_img.astype(np.int32) - black_level, 0, 65535).astype(np.uint16)

    # 去马赛克：假设Bayer BGGR图案，使用bilinear interpolation (avg-like)
    linear_bgr = cv2.cvtColor(lsc_corrected, cv2.COLOR_BayerBGGR2BGR)

    # 转换为浮点以进行均值滤波
    linear_bgr_float = linear_bgr.astype(np.float32)

    # 均值滤波 (21,21) 核，输出浮点
    kernel = (21, 21)
    blurred_bgr = cv2.blur(linear_bgr_float, kernel)

    # 获取图像中心像素作为参考
    h, w = blurred_bgr.shape[:2]
    center_y, center_x = h // 2, w // 2
    ref_b, ref_g, ref_r = blurred_bgr[center_y, center_x]  # BGR顺序

    # 计算3通道增益地图：gain = ref / pixel (避免除零)
    epsilon = 1e-6  # 小值避免除零
    gain_map = np.zeros_like(blurred_bgr)
    gain_map[..., 0] = ref_b / (blurred_bgr[..., 0] + epsilon)  # B通道
    gain_map[..., 1] = ref_g / (blurred_bgr[..., 1] + epsilon)  # G通道
    gain_map[..., 2] = ref_r / (blurred_bgr[..., 2] + epsilon)  # R通道

    return gain_map

_base_gain_map = generate_map()

def apply_lsc(img):
    # 假设img是浮点BGR图像；如果不是，需要调整
    if img.shape != _base_gain_map.shape:
        raise ValueError("Image shape does not match the gain map shape")
    img_corrected = img * _base_gain_map.astype(np.float32)
    return img_corrected
