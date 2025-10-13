#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from reportlab.lib.pagesizes import landscape, A3
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import black
from reportlab.lib.utils import ImageReader

from PIL import Image
from io import BytesIO
import numpy as np
import cv2

# ================== 可调参数 ==================
TAG_IDS = (0, 1, 2, 3)      # 顺序：LT, RT, LB, RB
BASE_TAG_SIZE_MM = 70.0     # 基准尺寸；实际绘制会乘以 SHRINK_RATIO
SHRINK_RATIO = 0.9
OUTER_WHITE_BITS = 1.5      # ★ 外白边厚度（bit）= 1.5
BORDER_BITS = 1             # AprilTag 的黑色环（标准=1）
MODULE_PX = 90              # 码元像素（60~120均可；90→1.5bit=135px）

# ColorChecker 实际尺寸（mm）：20.4 x 29.0 cm
CC_W = 204.0
CC_H = 290.0

# 输出文件
PDF_PATH = "tag.pdf"

# ================== 工具函数 ==================
def pil_to_ir(img: Image.Image) -> ImageReader:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)

def generate_apriltag_16h5(tag_id: int, inner_plus_border_px: int, border_bits: int = 1) -> Image.Image:
    """用 OpenCV aruco 生成 16h5（含标准黑边 border_bits；不含自定义外白边）。"""
    aruco = cv2.aruco
    dict_ = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
    if hasattr(aruco, "generateImageMarker"):
        img = aruco.generateImageMarker(dict_, tag_id, inner_plus_border_px, borderBits=border_bits)
    else:
        img = np.zeros((inner_plus_border_px, inner_plus_border_px), dtype=np.uint8)
        aruco.drawMarker(dict_, tag_id, inner_plus_border_px, img, borderBits=border_bits)
    return Image.fromarray(img).convert("L")

def add_outer_white_margin(img: Image.Image, outer_white_bits: float, module_px: int) -> Image.Image:
    """在标签（已含黑边）外再加 outer_white_bits 个 bit 的白边。允许小数（如 1.5）。"""
    if outer_white_bits <= 0:
        return img
    pad = int(round(outer_white_bits * module_px))
    w, h = img.size
    out = Image.new("L", (w + 2*pad, h + 2*pad), color=255)
    out.paste(img, (pad, pad))
    return out

# ================== 版式与绘制 ==================
def make_pdf(
    pdf_path: str = PDF_PATH,
    tag_ids = TAG_IDS,
    base_tag_size_mm: float = BASE_TAG_SIZE_MM,
    shrink_ratio: float = SHRINK_RATIO,
    module_px: int = MODULE_PX,
    outer_white_bits: float = OUTER_WHITE_BITS,
    border_bits: int = BORDER_BITS,
):
    # A3 横板尺寸（mm）
    A3_W_MM, A3_H_MM = 420.0, 297.0

    # 画布
    page_w_pt, page_h_pt = landscape(A3)
    c = canvas.Canvas(pdf_path, pagesize=(page_w_pt, page_h_pt))

    # ColorChecker 居中，画左/右竖线 + 顶部横线
    cc_x = (A3_W_MM - CC_W) / 2.0
    cc_y = (A3_H_MM - CC_H) / 2.0
    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    # 左右边界
    c.line(cc_x*mm, cc_y*mm, cc_x*mm, (cc_y + CC_H)*mm)
    c.line((cc_x + CC_W)*mm, cc_y*mm, (cc_x + CC_W)*mm, (cc_y + CC_H)*mm)
    # 顶部边界
    c.line(cc_x*mm, (cc_y + CC_H)*mm, (cc_x + CC_W)*mm, (cc_y + CC_H)*mm)

    # 标签外框物理边长（mm）
    s_mm = base_tag_size_mm * shrink_ratio

    # 以码元对齐生成图像：数据4×4，黑边=1，外白=outer_white_bits
    inner_plus_border_modules = 6                          # 数据4 + 黑边1*2 的总宽度始终为 6
    inner_plus_border_px = int(inner_plus_border_modules * int(module_px))

    # 生成四个 tag（先生成含黑边的，再加外白边）
    tags = []
    for tid in tag_ids:
        tag_core = generate_apriltag_16h5(tid, inner_plus_border_px, border_bits=border_bits)
        tag_full = add_outer_white_margin(tag_core, outer_white_bits, int(module_px))
        tags.append(pil_to_ir(tag_full))

    s = s_mm  # 简写

    # —— 摆放：以“外白边最外侧”与 ColorChecker 左/右边贴齐
    x_left  = cc_x - s            # 左侧两枚：右边刚好到 CC 左边 => 左上角 x = cc_x - s
    x_right = cc_x + CC_W         # 右侧两枚：左边刚好到 CC 右边
    y_top   = cc_y + CC_H - s     # 上对齐
    y_bot   = cc_y                # 下对齐

    # 坐标点 = 外框“左上角”
    positions_mm = [
        (x_left,  y_top),   # LT
        (x_right, y_top),   # RT
        (x_left,  y_bot),   # LB
        (x_right, y_bot),   # RB
    ]

    # 绘制 tag 与最外沿描线
    c.setLineWidth(0.5)
    for ir, (x_mm, y_mm) in zip(tags, positions_mm):
        c.drawImage(ir, x_mm*mm, y_mm*mm, width=s*mm, height=s*mm, mask='auto')
        c.rect(x_mm*mm, y_mm*mm, s*mm, s*mm, stroke=1, fill=0)

    # ====== 中央打印：等宽 & 带符号定宽 ======
    origin_x = A3_W_MM / 2.0  # PDF 中心为原点
    origin_y = A3_H_MM / 2.0
    c.setFont("Courier", 11)
    line_gap_mm = 6.0
    center_x_mm = origin_x
    center_y_mm = origin_y

    lines = []
    # 每个模块的物理尺寸（mm）
    module_size_mm = s / (4 + 2 * border_bits + 2 * outer_white_bits)
    offset_mm = outer_white_bits * module_size_mm  # 1.5 bit 对应的物理长度

    for (x_mm, y_mm), tid in zip(positions_mm, tag_ids):
        # ★ 不含外白边的左上角（外白边往右、往上各偏移 offset_mm）
        x_top_left = x_mm + offset_mm
        y_top_left = y_mm + s - offset_mm
        # 转为中心原点坐标
        cx = x_top_left - origin_x
        cy = y_top_left - origin_y
        lines.append(f"{tid:d}, {cx:+7.2f}, {cy:+7.2f} mm")

    # 追加一行：整体可用尺寸（左上tag左上角 → 右下tag右下角）
    overall_w = (x_right + s) - x_left    # 右下 tag 的右边 - 左上 tag 的左边
    overall_h = (y_top + s) - y_bot       # 右下 tag 的下边 - 左上 tag 的上边
    lines.append(f"size: {overall_w:0.2f} x {overall_h:0.2f} mm")

    total_h = (len(lines)-1) * line_gap_mm
    start_y = center_y_mm + total_h/2.0
    for i, text in enumerate(lines):
        y_mm = start_y - i*line_gap_mm
        c.drawCentredString(center_x_mm*mm, y_mm*mm, text)

    c.showPage()
    c.save()
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    make_pdf(PDF_PATH)
