#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from detect_colorchecker import detect_colorchecker
from calc_E import compute_e00_report
from plot_lab_ab import create_lab_delta_canvas


def summarize_colorchecker(img_path: str):
    """
    输入: 图像路径
    输出:
      merged: np.ndarray, 单张合并图 (检测500x500 + rectify高=500 + a*b* 500x500, 中间10px分隔)
      avg_e: float, 平均 E
      avg_c: float, 平均 |C|
      avg_l: float, 平均 |L|
    """

    # ====================== 内部小工具 ======================

    def clamp01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    def letterbox_square(img: np.ndarray, size: int = 500, color=(0, 0, 0)) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(size / w, size / h) if min(w, h) > 0 else 1.0
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size, 3), color, dtype=np.uint8)
        x0 = (size - nw) // 2
        y0 = (size - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    def resize_to_height(img: np.ndarray, target_h: int = 500) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0:
            return img
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    def ensure_square_500(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == 500 and img.shape[1] == 500:
            return img
        return letterbox_square(img, 500, color=(0, 0, 0))

    def hstack_variable_height(images: list, sep: int = 0, sep_color=(0, 0, 0)) -> np.ndarray:
        if not images:
            raise ValueError("空图列表")
        heights = [im.shape[0] for im in images]
        if len(set(heights)) != 1:
            raise ValueError("输入图片高度必须一致")
        H = heights[0]
        widths = [im.shape[1] for im in images]
        total_w = sum(widths) + sep * (len(images) - 1)
        canvas = np.full((H, total_w, 3), sep_color, dtype=np.uint8)
        x = 0
        for i, im in enumerate(images):
            w = im.shape[1]
            canvas[:, x:x+w] = im
            x += w
            if sep and i < len(images) - 1:
                canvas[:, x:x+sep] = np.uint8(sep_color)
                x += sep
        return canvas

    def draw_metrics_on_rectified(rect_img: np.ndarray,
                                  E: np.ndarray, C: np.ndarray, L: np.ndarray,
                                  rows: int = 4, cols: int = 6) -> np.ndarray:
        vis = rect_img.copy()
        h, w = vis.shape[:2]
        tile_w = w / cols
        tile_h = h / rows

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        th = 1
        pad_x = 6
        pad_y = 3
        line_gap = 2

        (_, th_text), _ = cv2.getTextSize("E 00.00", font, fs, th)
        text_block_h = 4 * th_text + 3 * line_gap + 2 * pad_y
        banner_h = max(text_block_h, int(0.35 * tile_h))

        for i in range(rows * cols):
            r = i // cols
            c = i % cols

            x0 = int(round(c * tile_w))
            y0 = int(round(r * tile_h))
            x1 = int(round((c + 1) * tile_w)) - 1
            y1 = int(round((r + 1) * tile_h)) - 1

            banner_y1 = min(y0 + int(banner_h), y1)
            cv2.rectangle(vis, (x0, y0), (x1, banner_y1), (0, 0, 0), -1)

            ty_id = y0 + pad_y + th_text
            ty_e = ty_id + th_text + line_gap
            ty_c = ty_e + th_text + line_gap
            ty_l = ty_c + th_text + line_gap

            label_x = x0 + pad_x
            value_right_x = x1 - pad_x

            e_val = float(E[i])
            c_val = float(C[i])
            l_val = float(L[i])

            id_text = f"{i+1}"
            id_size = cv2.getTextSize(id_text, font, fs, th)[0]
            id_x = x0 + (x1 - x0 - id_size[0]) // 2
            cv2.putText(vis, id_text, (id_x, ty_id), font, fs, (255, 255, 255), th, cv2.LINE_AA)

            label_e, label_c, label_l = "E", "C", "L"
            val_e = f"{e_val:.2f}"
            val_c = f"{c_val:.2f}"
            val_l = f"{l_val:+.2f}"

            tw_e, _ = cv2.getTextSize(val_e, font, fs, th)[0]
            tw_c, _ = cv2.getTextSize(val_c, font, fs, th)[0]
            tw_l, _ = cv2.getTextSize(val_l, font, fs, th)[0]

            vx_e = value_right_x - tw_e
            vx_c = value_right_x - tw_c
            vx_l = value_right_x - tw_l

            cv2.putText(vis, label_e, (label_x, ty_e), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            cv2.putText(vis, label_c, (label_x, ty_c), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            cv2.putText(vis, label_l, (label_x, ty_l), font, fs, (255, 255, 255), th, cv2.LINE_AA)

            cv2.putText(vis, val_e, (vx_e, ty_e), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            cv2.putText(vis, val_c, (vx_c, ty_c), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            cv2.putText(vis, val_l, (vx_l, ty_l), font, fs, (255, 255, 255), th, cv2.LINE_AA)

        return vis

    def draw_summary_avgs(merged: np.ndarray, avg_e: float, avg_abs_c: float, avg_abs_l: float) -> np.ndarray:
        vis = merged.copy()
        H, W = vis.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.7
        th = 2
        pad_x = 2
        pad_y = 2
        line_gap = 6
        margin = 12

        lines = [
            f"AVG E: {avg_e:.2f}",
            f"AVG C: {avg_abs_c:.2f}",
            f"AVG L: {avg_abs_l:.2f}",
        ]

        sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
        text_w = max(w for (w, h) in sizes)
        text_h = sum(h for (w, h) in sizes) + line_gap * (len(lines) - 1)

        box_w = text_w + pad_x * 2
        box_h = text_h + pad_y * 2

        x1 = W - margin
        y0 = margin
        x0 = max(0, x1 - box_w)
        # y1 = y0 + box_h  # 如需画底色可用

        ty = y0 + pad_y
        for t, (tw, th_text) in zip(lines, sizes):
            ty += th_text
            cv2.putText(vis, t, (x0 + pad_x, ty), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            ty += line_gap

        return vis

    # ====================== 读图与检测 ======================

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未找到图像: {img_path}")

    # detect_colorchecker 返回:
    #   detect_vis: 原图上绘制了 ColorChecker 外框与 AprilTag 的检测可视化图
    #   rect_vis: 4x6 展开图 (后续将写编号和 E/C/L)
    #   rgb_means: 24x3, 每格 BGR 均值 (0~255)
    detect_vis, rect_vis, rgb_means = detect_colorchecker(img)

    # 计算 ΔE00 等指标
    cam_srgb01 = clamp01(rgb_means / 255.0)  # BGR->RGB 后再 0~1
    E, C, L, cam_lab, ref_lab = compute_e00_report(cam_srgb01, unify_luminance=True)

    # 顶部写编号与 E/C/L
    rect_annot = draw_metrics_on_rectified(rect_vis, E, C, L, rows=4, cols=6)

    # 生成 Imatest 风格 a*b* 图 (500x500)
    lab_img = create_lab_delta_canvas(
        cam_lab, ref_lab,
        size=500, L_bg=80.0,
        a_label_min=-70.0, a_label_max=90.0,
        b_label_min=-70.0, b_label_max=104.0,
        pad_a=10.0, pad_b=10.0
    )

    # 统一尺寸
    left_sq = letterbox_square(detect_vis, 500, color=(0, 0, 0))
    mid_500h = resize_to_height(rect_annot, 500)
    right_sq = ensure_square_500(lab_img)

    # 合并图
    merged = hstack_variable_height([left_sq, mid_500h, right_sq], sep=10, sep_color=(0, 0, 0))

    # 计算并写 AVG (C 和 L 取绝对值)
    avg_e = float(np.mean(E))
    avg_c = float(np.mean(np.abs(C)))
    avg_l = float(np.mean(np.abs(L)))
    merged = draw_summary_avgs(merged, avg_e, avg_c, avg_l)

    return merged, avg_e, avg_c, avg_l


# ============ 示例用法 ============
if __name__ == "__main__":
    merged_img, avg_e, avg_c, avg_l = summarize_colorchecker("data/iphone.png")
    print(f"AVG E = {avg_e:.3f}, AVG C = {avg_c:.3f}, AVG L = {avg_l:.3f}")
    cv2.imshow("summary_merged", merged_img)
    cv2.waitKey(0)
