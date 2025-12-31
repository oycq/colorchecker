# isp.py
import numpy as np
import cv2
from .awb import awb_analysis
from . import lsc

BLACK_LEVEL = 9

def linear_to_srgb(img_linear):
    mask = img_linear > 0.0031308
    img_linear[mask] = 1.055 * (img_linear[mask] ** (1 / 2.4)) - 0.055
    img_linear[~mask] *= 12.92
    return (np.clip(img_linear, 0, 1) * 255 + 0.5).astype(np.uint8)

def tmo(bgr_float: np.ndarray) -> np.ndarray:
    # 1. 计算亮度 Y
    # 权重: Rec.709
    Y = 0.0722 * bgr_float[:, :, 0] + 0.7152 * bgr_float[:, :, 1] + 0.2126 * bgr_float[:, :, 2]

    # 2. 降采样统计 (128x128 = 16384 pixels)
    Y_small = cv2.resize(Y, (128, 128), interpolation=cv2.INTER_AREA)

    # 3. 映射到 0-1023 整数域
    Y_int_small = (Y_small * 1023).clip(0, 1023).astype(np.int32)

    # 4. 计算直方图
    hist = np.bincount(Y_int_small.flatten(), minlength=1024)

    # 5. 直方图非线性操作
    hist = hist.astype(np.float32) ** (1/3.0)

    # 6. 生成 LUT (累积分布)
    cdf = hist.cumsum()
    
    # 归一化到 0-1
    # 防止除以0 (处理全黑图的边缘情况)
    max_val = cdf[-1] if cdf[-1] > 0 else 1.0
    lut = (cdf / max_val).astype(np.float32)

    # 7. 应用 LUT
    Y_int_full = (Y * 1023).clip(0, 1023).astype(np.int32)
    Y_new = lut[Y_int_full]

    # 8. 颜色重构 (Ratio 保持色相)
    epsilon = 1e-5
    scale = Y_new / (Y + epsilon)
    tonemapped_bgr = bgr_float * scale[:, :, np.newaxis]

    return np.clip(tonemapped_bgr, 0.0, 1.0)

def isp_process(img_float: np.ndarray) -> np.ndarray:
    """
    ISP 处理流程:
    - 输入: float32 (h, w), 范围 0-1 (RAW Bayer 图像)
    - 输出: uint8 (h, w, 3), sRGB RGB 图像
    """
    # 步骤1: 转换为 uint16 (0-65535)
    img_uint16 = (img_float * 65535.0).astype(np.uint16)
    
    # 步骤2: 减去黑电平 (9 * 256 = 2304)
    black_level = BLACK_LEVEL * 256
    img_corrected = np.clip(img_uint16.astype(np.int32) - black_level, 0, 65535).astype(np.uint16)
    
    # 步骤3: 去马赛克 (demosaicing), 转换为线性 BGR (uint16)
    linear_bgr_uint16 = cv2.cvtColor(img_corrected, cv2.COLOR_BayerBGGR2BGR_EA)
    
    # 步骤4: 转换为 float32 (0-1)
    linear_bgr_float = linear_bgr_uint16.astype(np.float32) / 65535.0    
    
    # 步骤4.1: LSC
    linear_bgr_float = lsc.apply_lsc(linear_bgr_float)
    linear_bgr_float = np.clip(linear_bgr_float, 0.0, 1.0)

    # 步骤5.1: 白平衡和CCM计算
    k_b, k_r, ccm = awb_analysis(linear_bgr_float)
    linear_bgr_float[:,:,0] *= k_b
    linear_bgr_float[:,:,2] *= k_r
    wb_bgr_float = np.clip(linear_bgr_float, 0.0, 1.0)
    
    # 步骤5.2: 应用CCM (先转换为RGB, 应用矩阵, 再转回BGR)
    wb_rgb_float = wb_bgr_float[..., ::-1]  # BGR to RGB
    h, w = wb_rgb_float.shape[:2]
    corrected_rgb = (wb_rgb_float.reshape(-1, 3) @ ccm.T).reshape(h, w, 3)
    corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
    corrected_bgr = corrected_rgb[..., ::-1]  # RGB to BGR

    # 步骤8: gamma
    rgb_srgb_uint8 = linear_to_srgb(corrected_bgr)

    return rgb_srgb_uint8  # 返回 BGR uint8, 便于 cv2.imshow 和 putText