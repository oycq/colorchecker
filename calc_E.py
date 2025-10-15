"""
输入 24x3 的相机 sRGB 0-1 数组
输出:
- 五个 24 长度的一维数组: E, L, C, H, corr
- 24x3 的 cam_lab
- 24x3 的 ref_lab
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# 参考 sRGB 24 色块 0-1
REF_RGB24 = np.array([
    [0.447, 0.317, 0.265],
    [0.764, 0.580, 0.501],
    [0.364, 0.480, 0.612],
    [0.355, 0.422, 0.253],
    [0.507, 0.502, 0.691],
    [0.382, 0.749, 0.670],
    [0.867, 0.481, 0.187],
    [0.277, 0.356, 0.668],
    [0.758, 0.322, 0.382],
    [0.361, 0.225, 0.417],
    [0.629, 0.742, 0.242],
    [0.895, 0.630, 0.162],
    [0.155, 0.246, 0.576],
    [0.277, 0.588, 0.285],
    [0.681, 0.199, 0.223],
    [0.928, 0.777, 0.077],
    [0.738, 0.329, 0.594],
    [0.000, 0.540, 0.660],
    [0.960, 0.962, 0.950],
    [0.786, 0.793, 0.793],
    [0.631, 0.639, 0.640],
    [0.474, 0.475, 0.477],
    [0.324, 0.330, 0.336],
    [0.191, 0.194, 0.199],
], dtype=np.float64)

# D65 XYZ 白点 Y=1
_D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
# 线性 sRGB 到 XYZ 矩阵 IEC 61966-2-1
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float64)

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1 / 2.4) - 0.055
    )

def _rgb_to_xyz_d65(rgb: np.ndarray) -> np.ndarray:
    rgb_lin = _srgb_to_linear(rgb)
    return rgb_lin @ _SRGB_TO_XYZ.T

def _xyz_to_lab_d65(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    Xn, Yn, Zn = _D65_XYZ
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def _rgb_to_lab_d65(rgb: np.ndarray) -> np.ndarray:
    return _xyz_to_lab_d65(_rgb_to_xyz_d65(rgb))

def _ciede2000_components(lab1: np.ndarray, lab2: np.ndarray,
                          kL: float = 1.0, kC: float = 1.0, kH: float = 1.0):
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt((C_bar ** 7) / (C_bar ** 7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    Cbp = 0.5 * (C1p + C2p)

    h1p = (np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = (np.degrees(np.arctan2(b2, a2p)) % 360.0)

    dLp = L2 - L1
    dCp = C2p - C1p

    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, dh)
    dh = np.where(dh < -180.0, dh + 360.0, dh)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2.0)

    Lbp = 0.5 * (L1 + L2)
    hbp = (h1p + h2p) / 2.0
    hbp = np.where(np.abs(h1p - h2p) > 180.0, hbp + 180.0, hbp) % 360.0

    T = 1.0 \
        - 0.17 * np.cos(np.radians(hbp - 30.0)) \
        + 0.24 * np.cos(np.radians(2.0 * hbp)) \
        + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0)) \
        - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0))

    SL = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T

    dth = 30.0 * np.exp(-((hbp - 275.0) / 25.0) ** 2)
    RC = 2.0 * np.sqrt((Cbp ** 7) / (Cbp ** 7 + 25.0 ** 7))
    RT = -np.sin(np.radians(2.0 * dth)) * RC

    Lt = dLp / (kL * SL)
    Ct = dCp / (kC * SC)
    Ht = dHp / (kH * SH)

    L_comp = Lt ** 2
    C_comp = Ct ** 2
    H_comp = Ht ** 2
    corr = RT * Ct * Ht

    E2 = L_comp + C_comp + H_comp + corr
    E = np.sqrt(np.maximum(E2, 0.0))
    return E, Lt, Ct, Ht, corr

def unify_luminance_by_patch21(
    cam_sRGB24: np.ndarray,
    ref_sRGB24: np.ndarray,
) -> np.ndarray:
    """
    用 24x3 的 21# 色块(默认索引 20) 在线性域做亮度统一化(0-255)
    """
    cam = np.asarray(cam_sRGB24, dtype=np.float64)
    ref = np.asarray(ref_sRGB24, dtype=np.float64)
    assert cam.shape == (24, 3) and ref.shape == (24, 3)

    # 转 0~1
    cam01 = cam / 255.0
    ref01 = ref / 255.0

    # sRGB -> Linear
    cam_lin = _srgb_to_linear(cam01)
    ref_lin = _srgb_to_linear(ref01)

    # 计算指定色块的线性亮度 (r+g+b)/3
    cam_L = float(cam_lin[20].mean())
    ref_L = float(ref_lin[20].mean())

    scale = ref_L / cam_L
    cam_lin = np.clip(cam_lin * scale, 0.0, 1.0)

    # Linear -> sRGB -> 0~255
    cam01_adj = _linear_to_srgb(cam_lin)
    cam_adj = np.clip(cam01_adj * 255.0, 0.0, 255.0)
    return cam_adj


def compute_e00_report(
    cam_sRGB24: np.ndarray,
    ref_sRGB24: np.ndarray = REF_RGB24,
    unify_luminance: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    输入 24x3 相机 sRGB 数组, 输出:
      E, L 两个一维数组 (24,)
      cam_lab, ref_lab 各 24x3

    当 unify_luminance=True 时:
      先调用 unify_luminance_by_patch21 在线性域用 21# 色块(索引 20) 做亮度统一化
      再进行 Lab/ΔE00 计算
    """
    cam = np.asarray(cam_sRGB24, dtype=np.float64)
    ref = np.asarray(ref_sRGB24, dtype=np.float64)
    assert cam.shape == (24, 3) and ref.shape == (24, 3)

    if unify_luminance:
        cam = unify_luminance_by_patch21(cam, ref)

    cam_lab = _rgb_to_lab_d65(cam)
    ref_lab = _rgb_to_lab_d65(ref)

    E, L, C, H, corr = _ciede2000_components(cam_lab, ref_lab)
    return E, C, L, cam_lab, ref_lab

if __name__ == "__main__":
    cam_rgb24 = np.array([
        [0.365, 0.240, 0.193],
        [0.750, 0.469, 0.390],
        [0.276, 0.388, 0.530],
        [0.307, 0.360, 0.232],
        [0.394, 0.362, 0.542],
        [0.253, 0.664, 0.601],
        [0.909, 0.379, 0.000],
        [0.148, 0.226, 0.589],
        [0.772, 0.033, 0.196],
        [0.296, 0.113, 0.304],
        [0.506, 0.653, 0.057],
        [0.853, 0.522, 0.000],
        [0.000, 0.097, 0.483],
        [0.260, 0.573, 0.269],
        [0.726, 0.000, 0.000],
        [0.930, 0.759, 0.000],
        [0.729, 0.000, 0.439],
        [0.000, 0.425, 0.580],
        [0.910, 0.899, 0.879],
        [0.771, 0.769, 0.764],
        [0.614, 0.613, 0.609],
        [0.409, 0.410, 0.407],
        [0.280, 0.283, 0.279],
        [0.130, 0.131, 0.131],
    ], dtype=np.float64)

    E, C, L, cam_lab, ref_lab = compute_e00_report(cam_rgb24, unify_luminance = False)
    print("E_mean", E.mean())
    print("E", E) #imgtest 6.77 9.12 8.91 ..

