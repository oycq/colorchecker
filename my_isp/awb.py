#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import sys

LUM_MIN_THROAT = 10 / 255.0
LUM_MAX_THROAT = 220 / 255.0
update_ratio = 0.9995

cv2.namedWindow('AWB Scatter')
cv2.setWindowProperty('AWB Scatter', cv2.WND_PROP_TOPMOST, 1)

# 加载 JSON 数据（全局加载一次）
with open('results.json', 'r') as f:
    data = json.load(f)
calibration_results = data['calibration_results']

# 提取拟合参数
planck_a = data['planck_fit']['a']
planck_b = data['planck_fit']['b']
upper_a = data['planck_y_max']['a']
upper_b = data['planck_y_max']['b']
lower_a = data['planck_y_min']['a']
lower_b = data['planck_y_min']['b']
temp_a = data['temp_fit']['a']
temp_b = data['temp_fit']['b']
x_min = data['rg_limits']['min']
x_max = data['rg_limits']['max']

# 参考色温标签、色温和CCM（用于插值）
ref_labels = ['HZ', 'A', 'D65', 'D75']
ref_temps = [2300, 2856, 6500, 7500]
ref_ccms = [np.array(calibration_results[lbl]['ccm']) for lbl in ref_labels]

# 提取参考点
reference_rg_bg = []
reference_labels = []
for key, value in calibration_results.items():
    awb = value['awb']
    r_gain, _, b_gain = awb
    if r_gain != 0 and b_gain != 0:
        rg = 1 / r_gain  # R/G
        bg = 1 / b_gain  # B/G
        reference_rg_bg.append((rg, bg))
        reference_labels.append(key)

# 当前rg, bg (初始D65)
avg_rg = 0.54
avg_bg = 0.62

# 全局散点模板
scatter_template = None

def awb_analysis(img):
    original_img = cv2.resize(img, (32, 32)).astype(np.float32)
    
    analysis_scatter(original_img)
    
    # 计算色温
    T = temp_a / avg_rg + temp_b
    min_t = ref_temps[0]
    max_t = ref_temps[-1]
    T = max(min_t, min(max_t, T))
    
    # 找到最近两个参考色温并插值CCM
    for i in range(len(ref_temps) - 1):
        if ref_temps[i] <= T <= ref_temps[i + 1]:
            weight = (T - ref_temps[i]) / (ref_temps[i + 1] - ref_temps[i])
            ccm = ref_ccms[i] * (1 - weight) + ref_ccms[i + 1] * weight
            break
    else:
        if T <= ref_temps[0]:
            ccm = ref_ccms[0]
        else:
            ccm = ref_ccms[-1]
    
    # 行归一化，确保每一行和为1
    for i in range(3):
        row_sum = np.sum(ccm[i])
        if row_sum != 0:
            ccm[i] /= row_sum
    
    return 1 / avg_bg, 1 / avg_rg, ccm

def analysis_scatter(original_img):
    global avg_rg, avg_bg, scatter_template
    IMG_SHAPE = 450
    scale = IMG_SHAPE / 1.5
    
    # 如果模板未初始化，创建模板
    if scatter_template is None:
        scatter_template = np.full((IMG_SHAPE, IMG_SHAPE, 3), (20 / 255.0, 20 / 255.0, 20 / 255.0), np.float32)
        
        # 绘制上下限曲线（框框）
        x_vals = np.linspace(x_min, x_max, 100)
        for idx in range(len(x_vals) - 1):
            x1 = int(x_vals[idx] * scale)
            x2 = int(x_vals[idx + 1] * scale)
            
            y_upper1 = IMG_SHAPE - 1 - int((upper_a / x_vals[idx] + upper_b) * scale)
            y_upper2 = IMG_SHAPE - 1 - int((upper_a / x_vals[idx + 1] + upper_b) * scale)
            y_lower1 = IMG_SHAPE - 1 - int((lower_a / x_vals[idx] + lower_b) * scale)
            y_lower2 = IMG_SHAPE - 1 - int((lower_a / x_vals[idx + 1] + lower_b) * scale)
            
            cv2.line(scatter_template, (x1, y_upper1), (x2, y_upper2), (0.3, 0.3, 0.3), 1)
            cv2.line(scatter_template, (x1, y_lower1), (x2, y_lower2), (0.3, 0.3, 0.3), 1)
        
        # 绘制参考点和标签
        for (rg, bg), label in zip(reference_rg_bg, reference_labels):
            px = int(rg * scale)
            py = IMG_SHAPE - 1 - int(bg * scale)
            if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
                cv2.circle(scatter_template, (px, py), 5, (1, 0, 0), 2)
                cv2.putText(scatter_template, label, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)
    
    # 复制模板作为当前帧
    scatter_img = scatter_template.copy()
    
    # 绘制像素散点
    white_point_count = 0
    for i in range(32):
        for j in range(32):
            # 过滤过曝和欠曝像素
            b, g, r = original_img[i, j]
            max_val = max(b, g, r)
            avg_val = (b + g + r) / 3
            if not (max_val < LUM_MAX_THROAT and avg_val > LUM_MIN_THROAT):
                continue

            if g == 0:
                continue
            rg = r / g
            bg = b / g
            
            # 检查是否在白点区域（x_min到x_max，y_min到y_max）
            if not (x_min <= rg <= x_max):
                continue
            y_max_val = upper_a / rg + upper_b
            y_min_val = lower_a / rg + lower_b
            if not (y_min_val <= bg <= y_max_val):
                continue

            # 更新awb比值
            avg_rg = avg_rg * update_ratio + rg * (1 - update_ratio)
            avg_bg = avg_bg * update_ratio + bg * (1 - update_ratio)
            white_point_count += 1

            px = int(rg * scale)
            py = IMG_SHAPE - 1 - int(bg * scale)
            if 0 <= px < IMG_SHAPE and 0 <= py < IMG_SHAPE:
                scatter_img[py, px] = (1, 1, 1)  # 白点像素

    # 绘制白点数量
    cv2.putText(scatter_img, str(white_point_count), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    # 绘制(avg_rg, avg_bg)
    px = int(avg_rg * scale)
    py = IMG_SHAPE - 1 - int(avg_bg * scale)
    cv2.circle(scatter_img, (px, py), 5, (0, 0, 255), 2)
    
    # 计算色温并绘制
    T = temp_a / avg_rg + temp_b
    min_t = ref_temps[0]
    max_t = ref_temps[-1]
    T = max(min_t, min(max_t, T))
    cv2.putText(scatter_img, f"T: {int(T)}K", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

    # 计算CCM并在右上角显示（保留两位小数）
    # (CCM计算已在awb_analysis中，此处仅为显示使用相同的逻辑)
    for i in range(len(ref_temps) - 1):
        if ref_temps[i] <= T <= ref_temps[i + 1]:
            weight = (T - ref_temps[i]) / (ref_temps[i + 1] - ref_temps[i])
            ccm = ref_ccms[i] * (1 - weight) + ref_ccms[i + 1] * weight
            break
    else:
        if T <= ref_temps[0]:
            ccm = ref_ccms[0]
        else:
            ccm = ref_ccms[-1]
    
    # 行归一化（显示前）
    for i in range(3):
        row_sum = np.sum(ccm[i])
        if row_sum != 0:
            ccm[i] /= row_sum
    
    x_pos = IMG_SHAPE - 150
    y_pos = 20
    for i in range(3):
        row_str = f"{ccm[i,0]:.2f} {ccm[i,1]:.2f} {ccm[i,2]:.2f}"
        cv2.putText(scatter_img, row_str, (x_pos, y_pos + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)
    
    # 显示散点图
    cv2.imshow('AWB Scatter', scatter_img)