#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# ========= Lab → BGR =========

def lab_to_bgr255(lab_color: np.ndarray) -> tuple:
    """CIE Lab → BGR(0~255)，返回 Python int tuple"""
    lab_uint8 = np.zeros((1, 1, 3), dtype=np.uint8)
    lab_uint8[..., 0] = np.clip(lab_color[0] * 255.0 / 100.0, 0, 255).astype(np.uint8)
    lab_uint8[..., 1] = np.clip(lab_color[1] + 128.0, 0, 255).astype(np.uint8)
    lab_uint8[..., 2] = np.clip(lab_color[2] + 128.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_Lab2BGR)[0, 0, :]
    return tuple(int(v) for v in bgr)

# ========= 虚线绘制 =========

def _draw_dashed_line(img, p1, p2, color, thickness=1, dash_len=8, gap_len=6):
    """在 img 上画虚线"""
    p1 = np.array(p1, dtype=np.int32)
    p2 = np.array(p2, dtype=np.int32)
    vec = p2 - p1
    length = np.hypot(vec[0], vec[1])
    if length == 0:
        return
    direction = vec / length
    n_dashes = int(length // (dash_len + gap_len)) + 1
    for i in range(n_dashes):
        start = (p1 + direction * (i * (dash_len + gap_len))).astype(int)
        end = (p1 + direction * (i * (dash_len + gap_len) + dash_len)).astype(int)
        cv2.line(img, tuple(start), tuple(end), color, thickness, cv2.LINE_AA)

# ========= 主绘图函数 =========

def create_lab_delta_canvas(
    cam_lab: np.ndarray,
    ref_lab: np.ndarray,
    size: int = 500,
    L_bg: float = 100.0,
    a_label_min: float = -60.0,
    a_label_max: float = 80.0,
    b_label_min: float = -60.0,
    b_label_max: float = 100.0,
    pad_a: float = 10.0,
    pad_b: float = 10.0
) -> np.ndarray:
    """
    Imatest 风格 a*b* 平面偏差图
    cam_lab, ref_lab: (24,3) 的 Lab 值
    """
    assert cam_lab.shape == (24, 3)
    assert ref_lab.shape == (24, 3)

    # ===== 坐标范围 =====
    a_min, a_max = a_label_min - pad_a, a_label_max + pad_a
    b_min, b_max = b_label_min - pad_b, b_label_max + pad_b

    # ===== 背景生成（高效）=====
    xs = np.linspace(a_min, a_max, num=size, dtype=np.float32)
    ys = np.linspace(b_max, b_min, num=size, dtype=np.float32)
    A, B = np.meshgrid(xs, ys)
    L_channel = np.full((size, size), L_bg, dtype=np.float32)

    lab_float = np.stack([L_channel, A, B], axis=2)
    lab_uint8 = np.zeros_like(lab_float, dtype=np.uint8)
    lab_uint8[..., 0] = np.clip(lab_float[..., 0] * 255.0 / 100.0, 0, 255).astype(np.uint8)
    lab_uint8[..., 1] = np.clip(lab_float[..., 1] + 128.0, 0, 255).astype(np.uint8)
    lab_uint8[..., 2] = np.clip(lab_float[..., 2] + 128.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(lab_uint8, cv2.COLOR_Lab2BGR)

    # ===== 坐标转换函数 =====
    def to_px(a, b):
        x = int(round((a - a_min) / (a_max - a_min) * (size - 1)))
        y = int(round((b_max - b) / (b_max - b_min) * (size - 1)))
        return x, y

    # ===== ab=0 虚线 =====
    _draw_dashed_line(img, to_px(a_min, 0), to_px(a_max, 0), (0, 0, 0), 1, 10, 7)
    _draw_dashed_line(img, to_px(0, b_min), to_px(0, b_max), (0, 0, 0), 1, 10, 7)

    # ===== 连线 1~18 =====
    for i in range(18):
        p_ref = to_px(ref_lab[i, 1], ref_lab[i, 2])
        p_cam = to_px(cam_lab[i, 1], cam_lab[i, 2])
        cv2.line(img, p_ref, p_cam, lab_to_bgr255(cam_lab[i]), 1, cv2.LINE_AA)

    # ===== 标记 =====
    circle_radius = 6
    for i in range(24):
        p_ref = to_px(ref_lab[i, 1], ref_lab[i, 2])
        p_cam = to_px(cam_lab[i, 1], cam_lab[i, 2])
        color_cam = lab_to_bgr255(cam_lab[i])
        color_ref = lab_to_bgr255(ref_lab[i])

        if i < 18:
            # 参考小方块
            cv2.rectangle(img, (p_ref[0]-3, p_ref[1]-3),
                          (p_ref[0]+3, p_ref[1]+3), color_ref, -1, cv2.LINE_AA)
            cv2.rectangle(img, (p_ref[0]-4, p_ref[1]-4),
                          (p_ref[0]+4, p_ref[1]+4), (0, 0, 0), 1, cv2.LINE_AA)
            # 相机点
            cv2.circle(img, p_cam, circle_radius, color_cam, -1, cv2.LINE_AA)
            cv2.circle(img, p_cam, circle_radius, (0, 0, 0), 1, cv2.LINE_AA)
            # 标注数字
            cv2.putText(img, f"{i+1}", (p_cam[0]+7, p_cam[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # 19~24 只画相机点
            cv2.circle(img, p_cam, circle_radius, color_cam, -1, cv2.LINE_AA)
            cv2.circle(img, p_cam, circle_radius, (0, 0, 0), 1, cv2.LINE_AA)

    return img

# ========= 测试 =========
if __name__ == "__main__":
    np.random.seed(0)
    ref_lab = np.zeros((24, 3), dtype=np.float32)
    ref_lab[:, 0] = 65.0
    ref_lab[:, 1:] = np.random.uniform(-50, 50, (24, 2))

    cam_lab = ref_lab.copy()
    cam_lab[:, 1:] += np.random.normal(0, 10, (24, 2))

    canvas = create_lab_delta_canvas(cam_lab, ref_lab)
    cv2.imshow("LAB Plot Fast", canvas)
    cv2.waitKey(0)
