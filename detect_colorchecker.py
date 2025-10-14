#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import List, Tuple

# ========== 参数 ==========
IMG_PATH = "data/detect_colorchecker.png"

# tag 左上角点的真实世界坐标 单位 mm 原点是页面中心 x 向右 y 向上
WORLD_MM_BY_ID = {
    0: ( -154.50, +134.50 ),
    1: ( +112.50, +134.50 ),
    2: ( -154.50,  -92.50 ),
    3: ( +112.50,  -92.50 ),
}

# ColorChecker 的真实尺寸 mm
CC_W_MM = 204.0
CC_H_MM = 290.0
# ColorChecker 的裁剪尺寸 覆盖前面尺寸
CC_W_MM = 184.0
CC_H_MM = 276.0

# 展开后固定输出尺寸 px
OUT_W = 408
OUT_H = 580

# ========== 工具函数 ==========
def detect_tags_16h5(img: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    # 检测 AprilTag 16h5 返回 corners 和 ids
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids

def pick_top_left_point(quad: np.ndarray) -> np.ndarray:
    # 直接返回检测到的原始第一个点（不随图像旋转而改变）
    return quad[0]

def world_to_image(H: np.ndarray, pts_xy_mm: np.ndarray) -> np.ndarray:
    # 将世界坐标 mm 通过单应性矩阵投影到图像像素坐标
    pts = np.hstack([pts_xy_mm.astype(np.float64), np.ones((pts_xy_mm.shape[0], 1))])
    proj = (H @ pts.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj

def grid_centers_wh(width: int, height: int, rows: int = 4, cols: int = 6) -> np.ndarray:
    # 基于整幅图的宽高 计算 rows x cols 均匀中心坐标 返回 Nx2 顺序按行优先
    ys = (np.arange(rows) + 0.5) / rows * height
    xs = (np.arange(cols) + 0.5) / cols * width
    centers = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return centers

def sample_box_means_rgb(img_bgr: np.ndarray, centers: np.ndarray, box: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    # 在每个中心处取 box x box 方框 计算 RGB 均值 返回 rgb_means 和可视化图像
    h, w = img_bgr.shape[:2]
    half = box // 2
    vis = img_bgr.copy()
    rgb_means = []
    for cx, cy in centers:
        x0 = max(int(round(cx)) - half, 0)
        y0 = max(int(round(cy)) - half, 0)
        x1 = min(x0 + box, w)
        y1 = min(y0 + box, h)
        roi_bgr = img_bgr[y0:y1, x0:x1]
        if roi_bgr.size == 0:
            rgb_means.append([0.0, 0.0, 0.0])
        else:
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)
            rgb_means.append(mean_rgb.tolist())
        # 绘制黄色方框 线宽 1 注意 OpenCV 使用 BGR
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1, cv2.LINE_AA)
    rgb_means = np.array(rgb_means, dtype=np.float32)  # 形状 24x3 RGB 顺序
    return rgb_means, vis

# ========== 核心流程函数 ==========
def detect_colorchecker(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 输入原图 输出 检测叠加图 rectify 图 以及 24x3 的 RGB 均值矩阵
    if img_bgr is None:
        raise ValueError("输入图像为空")

    draw = img_bgr.copy()

    # 检测 aruco 16h5
    corners, ids = detect_tags_16h5(img_bgr)
    if ids is None or len(corners) == 0:
        raise RuntimeError("未检测到 AprilTag 16h5")

    # 叠加显示 每个 tag 用绿色框 左上角点画红点
    for c, id_ in zip(corners, ids.flatten()):
        pts = c.reshape(-1, 2)
        pts_i = pts.astype(int)
        cv2.polylines(draw, [pts_i.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        tl = pick_top_left_point(pts)
        cv2.circle(draw, tuple(np.round(tl).astype(int)), 4, (0, 0, 255), -1)
        cv2.putText(draw, f"id {int(id_)}", tuple(pts_i[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    # 组装配准点 仅使用左上角点
    img_pts = []
    world_pts = []
    for c, id_ in zip(corners, ids.flatten()):
        if int(id_) in WORLD_MM_BY_ID:
            quad = c.reshape(-1, 2)
            tl = pick_top_left_point(quad)
            img_pts.append(tl)
            world_pts.append(WORLD_MM_BY_ID[int(id_)])
    img_pts = np.asarray(img_pts, dtype=np.float64)
    world_pts = np.asarray(world_pts, dtype=np.float64)

    if len(img_pts) < 4:
        raise RuntimeError("用于单应性估计的点不足 需要 4 个 tag")

    # 单应性
    H, _ = cv2.findHomography(world_pts, img_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("单应性估计失败")

    # 投影色卡四角
    half_w = CC_W_MM / 2.0
    half_h = CC_H_MM / 2.0
    cc_world = np.array([
        [-half_w, +half_h],
        [+half_w, +half_h],
        [+half_w, -half_h],
        [-half_w, -half_h],
    ], dtype=np.float64)
    cc_img = world_to_image(H, cc_world)
    cc_img_i = np.round(cc_img).astype(int)
    cv2.polylines(draw, [cc_img_i.reshape(-1, 1, 2)], True, (255, 0, 0), 2)

    # 透视展开 然后顺时针旋转 90 度
    src_quad = cc_img.astype(np.float32)  # TL TR BR BL
    dst_quad = np.array([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]], dtype=np.float32)
    H_img2rect = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cc_rectified = cv2.warpPerspective(img_bgr, H_img2rect, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)
    cc_rectified = cv2.rotate(cc_rectified, cv2.ROTATE_90_CLOCKWISE)

    # 计算 4x6 中心 并在每个中心处画 20x20 方框 统计 RGB 均值
    h, w = cc_rectified.shape[:2]
    centers = grid_centers_wh(w, h, rows=4, cols=6)
    rgb_means, rect_vis = sample_box_means_rgb(cc_rectified, centers, box=20)

    # 返回 检测叠加图 展开可视化图 RGB 均值矩阵
    return draw, rect_vis, rgb_means

# ========== 主程序 ==========
def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"未找到 {IMG_PATH}")

    detect_vis, rect_vis, rgb_means = detect_colorchecker(img)

    # 显示窗口
    h, w = detect_vis.shape[:2]
    show = cv2.resize(detect_vis, (int(w * 0.75), int(h * 0.75)))
    cv2.imshow("img_detect", show)
    cv2.imshow("colorchecker_rectified", rect_vis)

    # 打印 24x3 RGB 均值矩阵
    np.set_printoptions(precision=3, suppress=True)
    print("RGB means shape", rgb_means.shape)
    print(rgb_means)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
