#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import my_isp


import os
import glob
import matplotlib.pyplot as plt

from colorchecker_analyze import summarize_colorchecker


BASE_PATH = "./scene"
OUTPUT_DIR = "./output_my_isp"

CAMERAS = ["OLD_ISP", "MY_ISP"]
CAMERA_PATTERNS = {
    "OLD_ISP": "*cam2_isp.png",
    "MY_ISP": "my_isp.png",
}

COLORS = {
    "OLD_ISP": "#6C757D",
    "MY_ISP": "#2E86AB",
}

plt.rcParams.update({
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
})

FIG_SIZE = (17.3, 5.0)
BAR_WIDTH = 0.35
BAR_OFFSETS = [-0.20, 0.20]
BAR_SPACING = BAR_WIDTH * 1.0
SCENE_SPACING = BAR_WIDTH * 1.2


def run_my_isp():
    from pathlib import Path

    inputs = sorted(Path("scene").rglob("*raw.png"))
    for p in inputs:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img = raw.astype(np.float32) / 65535.0
        out = my_isp.process(img)
        cv2.imwrite(str(p.parent / "my_isp.png"), out)
        print(p)


def get_scene_data():
    os.makedirs(f"{OUTPUT_DIR}/merged_images", exist_ok=True)
    scenes = []

    for scene_folder in os.listdir(BASE_PATH):
        scene_path = os.path.join(BASE_PATH, scene_folder)
        if not os.path.isdir(scene_path):
            continue

        scene_data = {}
        merged_imgs = {}

        for camera, pattern in CAMERA_PATTERNS.items():
            img_paths = glob.glob(os.path.join(scene_path, pattern))
            scene_data[camera] = []
            merged_imgs[camera] = []

            for path in img_paths:
                try:
                    merged_img, avg_e, avg_c, avg_l = summarize_colorchecker(path)
                except Exception as e:
                    print("skip:", path, str(e))
                    continue

                merged_img_rgb = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)

                img_path = f"{OUTPUT_DIR}/merged_images/{scene_folder}/{camera}.png"
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                plt.imsave(img_path, merged_img_rgb)
                merged_imgs[camera].append(img_path)

                scene_data[camera].append((avg_e, avg_c, avg_l))

        scenes.append({
            "folder_name": scene_folder,
            "data": scene_data,
            "images": merged_imgs,
        })

    return scenes


def calculate_averages(scenes):
    avg_values = {camera: {"E": [], "C": [], "L": []} for camera in CAMERAS}
    for scene in scenes:
        for camera in CAMERAS:
            if scene["data"][camera]:
                e_vals, c_vals, l_vals = zip(*scene["data"][camera])
                avg_values[camera]["E"].append(float(np.mean(e_vals)))
                avg_values[camera]["C"].append(float(np.mean(c_vals)))
                avg_values[camera]["L"].append(float(np.mean(l_vals)))
            else:
                avg_values[camera]["E"].append(np.nan)
                avg_values[camera]["C"].append(np.nan)
                avg_values[camera]["L"].append(np.nan)
    return avg_values


def plot_bars(avg_values, scenes, metric, ylabel, save_path):
    x = np.arange(len(scenes)) * (BAR_WIDTH * 2 + BAR_SPACING + SCENE_SPACING)
    scene_names = [scene["folder_name"] for scene in scenes]

    plt.figure(figsize=FIG_SIZE)
    for i, camera in enumerate(CAMERAS):
        values = avg_values[camera][metric]
        bars = plt.bar(
            x + BAR_OFFSETS[i],
            values,
            width=BAR_WIDTH,
            label=camera,
            align="center",
            color=COLORS[camera],
        )
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    plt.xticks(x, scene_names, rotation=0, ha="center", fontsize=11)
    plt.xlabel("scene", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(0, 12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_table(avg_values, save_path):
    summary_data = {}
    for camera in CAMERAS:
        summary_data[camera] = {
            "E": float(np.nanmean(avg_values[camera]["E"])) if avg_values[camera]["E"] else 0.0,
            "C": float(np.nanmean(avg_values[camera]["C"])) if avg_values[camera]["C"] else 0.0,
            "L": float(np.nanmean(avg_values[camera]["L"])) if avg_values[camera]["L"] else 0.0,
        }

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    metrics = ["E", "C", "L"]
    titles = ["DeltaE", "DeltaC", "DeltaL"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = [summary_data[camera][metric] for camera in CAMERAS]
        bars = ax.bar(CAMERAS, values, width=0.8, color=[COLORS[camera] for camera in CAMERAS])
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(0, 10)
        ax.tick_params(axis="x", labelsize=12)

        for bar, val in zip(bars, values):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.3,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_html(scenes):
    num_scenes = len(scenes)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ISP compare report</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', Arial; margin: 20px; }}
    .container {{ max-width: 80%; margin: auto; }}
    .header {{ text-align: center; margin-bottom: 40px; }}
    .plot {{ text-align: center; margin: 30px 0; }}
    .plot img {{ width: 100%; height: auto; max-width: 100%; }}
    .scene-gallery {{ display: grid; grid-template-columns: 1fr; gap: 60px; margin-top: 40px; }}
    .scene {{ margin-bottom: 40px; }}
    .scene h3 {{ margin: 0 0 20px 0; text-align: center; }}
    .camera-label {{ font-weight: bold; margin: 10px 0 5px 0; font-size: 18px; text-align: center; }}
    .scene img {{ width: 100%; height: auto; max-width: 100%; object-fit: contain; display: block; margin: 5px auto; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ISP compare</h1>
      <p>OLD_ISP vs MY_ISP | {num_scenes} scenes</p>
    </div>
    <div class="plot">
      <h2>summary</h2>
      <img src="summary_plots/summary.png" alt="summary">
    </div>
    <div class="plot">
      <h2>DeltaE by scene</h2>
      <img src="summary_plots/delta_e.png" alt="delta_e">
    </div>
    <div class="plot">
      <h2>DeltaC by scene</h2>
      <img src="summary_plots/delta_c.png" alt="delta_c">
    </div>
    <div class="plot">
      <h2>DeltaL by scene</h2>
      <img src="summary_plots/delta_l.png" alt="delta_l">
    </div>
    <div class="scene-gallery">
"""

    for scene in scenes:
        html += f'<div class="scene"><h3>{scene["folder_name"]}</h3>'
        for camera in CAMERAS:
            html += f'''
  <div class="camera-img">
    <div class="camera-label">{camera}</div>
    <img src="merged_images/{scene["folder_name"]}/{camera}.png" alt="{camera}">
  </div>
'''
        html += "</div>"

    html += """
    </div>
  </div>
</body>
</html>
"""

    with open(f"{OUTPUT_DIR}/index.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    run_my_isp()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/summary_plots", exist_ok=True)

    scenes = get_scene_data()
    avg_values = calculate_averages(scenes)

    plot_bars(avg_values, scenes, "E", "DeltaE", f"{OUTPUT_DIR}/summary_plots/delta_e.png")
    plot_bars(avg_values, scenes, "C", "DeltaC", f"{OUTPUT_DIR}/summary_plots/delta_c.png")
    plot_bars(avg_values, scenes, "L", "DeltaL", f"{OUTPUT_DIR}/summary_plots/delta_l.png")
    plot_summary_table(avg_values, f"{OUTPUT_DIR}/summary_plots/summary.png")

    generate_html(scenes)
    print(f"done, open: {OUTPUT_DIR}/index.html")


if __name__ == "__main__":
    main()