import os
#防止多线程冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

import my_isp
from colorchecker_analyze import summarize_colorchecker


SCENE_DIR = "scene2"
BASE_PATH = "./" + SCENE_DIR

# html output fixed here
OUTPUT_DIR = "./output2"

# names used in report
CAMERAS = ["old_isp", "new_isp"]
CAMERA_PATTERNS = {
    "old_isp": "*cam2_isp.png",
    "new_isp": "new_isp.png",
}

CAMERA_DISPLAY = {
    "old_isp": "地平线AWB",
    "new_isp": "自研AWB",
}

COLORS = {
    "old_isp": "#6C757D",
    "new_isp": "#2E86AB",
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


def run_new_isp():
    from pathlib import Path

    inputs = sorted(Path(SCENE_DIR).rglob("*raw.png"))
    for p in inputs:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img = raw.astype(np.float32) / 65535.0
        out = my_isp.process(img)
        cv2.imwrite(str(p.parent / "new_isp.png"), out)
        print(p)


def get_scene_data():
    os.makedirs(f"{OUTPUT_DIR}/merged_images", exist_ok=True)
    scenes = []

    for scene_folder in os.listdir(BASE_PATH):
        scene_path = os.path.join(BASE_PATH, scene_folder)
        if not os.path.isdir(scene_path):
            continue

        # scene-level rule: if any camera fails (no tag / no file / exception), skip the whole scene
        per_cam = {}
        ok = True

        for camera, pattern in CAMERA_PATTERNS.items():
            img_paths = glob.glob(os.path.join(scene_path, pattern))
            if not img_paths:
                ok = False
                break

            path = img_paths[0]
            try:
                merged_img, avg_e, avg_c, avg_l = summarize_colorchecker(path)
            except Exception:
                ok = False
                break

            per_cam[camera] = {
                "merged_img": merged_img,
                "avg_e": avg_e,
                "avg_c": avg_c,
                "avg_l": avg_l,
            }

        if not ok:
            print("skip scene:", scene_folder)
            continue

        # save merged images only when both cameras are ok
        images = {}
        data = {}
        for camera in CAMERAS:
            merged_img_rgb = cv2.cvtColor(per_cam[camera]["merged_img"], cv2.COLOR_BGR2RGB)
            img_path = f"{OUTPUT_DIR}/merged_images/{scene_folder}/{camera}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.imsave(img_path, merged_img_rgb)

            images[camera] = [img_path]
            data[camera] = [(per_cam[camera]["avg_e"], per_cam[camera]["avg_c"], per_cam[camera]["avg_l"])]

        scenes.append({
            "folder_name": scene_folder,
            "data": data,
            "images": images,
        })

    return scenes


def calculate_averages(scenes):
    avg_values = {camera: {"E": [], "C": [], "L": []} for camera in CAMERAS}
    for scene in scenes:
        for camera in CAMERAS:
            e_vals, c_vals, l_vals = zip(*scene["data"][camera])
            avg_values[camera]["E"].append(float(np.mean(e_vals)))
            avg_values[camera]["C"].append(float(np.mean(c_vals)))
            avg_values[camera]["L"].append(float(np.mean(l_vals)))
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
            label=CAMERA_DISPLAY[camera],
            align="center",
            color=COLORS[camera],
        )
        for bar, val in zip(bars, values):
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
    plt.xlabel("场景", fontsize=16)
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
            "E": float(np.mean(avg_values[camera]["E"])) if avg_values[camera]["E"] else 0.0,
            "C": float(np.mean(avg_values[camera]["C"])) if avg_values[camera]["C"] else 0.0,
            "L": float(np.mean(avg_values[camera]["L"])) if avg_values[camera]["L"] else 0.0,
        }

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    metrics = ["E", "C", "L"]
    titles = ["DeltaE", "DeltaC", "DeltaL"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = [summary_data[camera][metric] for camera in CAMERAS]
        bars = ax.bar([CAMERA_DISPLAY[c] for c in CAMERAS], values, width=0.8, color=[COLORS[camera] for camera in CAMERAS])
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
  <title>新旧AWB效果对比报告</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', Arial; margin: 20px; }}
    .container {{ max-width: 80%; margin: auto; }}
    .header {{ text-align: center; margin-bottom: 20px; }}
    .plot {{ text-align: center; margin: 30px 0; }}
    .plot img {{ width: 100%; height: auto; max-width: 100%; }}
    .first-page {{ page-break-inside: avoid; break-inside: avoid; }}
    .second-page {{ page-break-inside: avoid; break-inside: avoid; }}
    .intro {{ margin: 10px 0 20px 0; padding: 0; border: none; }}
    .intro h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .intro p {{ margin: 6px 0; line-height: 1.45; }}
    .intro ul {{ margin: 8px 0 0 18px; }}
    .intro li {{ margin: 4px 0; line-height: 1.45; }}
    .scene-gallery {{ display: grid; grid-template-columns: 1fr; gap: 60px; margin-top: 40px; }}
    .scene {{ margin-bottom: 40px; break-inside: avoid; page-break-inside: avoid; }}
    .scene h3 {{ margin: 0 0 20px 0; text-align: center; }}
    .camera-label {{ font-weight: bold; margin: 10px 0 5px 0; font-size: 18px; text-align: center; }}
    .scene img {{ width: 100%; height: auto; max-width: 100%; object-fit: contain; display: block; margin: 5px auto; }}

    @media print {{
      body {{ margin: 10mm; }}
      .container {{ max-width: 100%; }}
      .first-page {{
        min-height: 277mm;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }}
      .first-page {{ page-break-after: always; break-after: page; }}
      .second-page {{ page-break-after: always; break-after: page; }}
      .scene-gallery {{ gap: 20px; margin-top: 20px; }}
      .scene {{ margin-bottom: 20px; }}
      .scene-gallery > .scene:nth-child(2n+1):not(:first-child) {{ page-break-before: always; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="first-page">
      <div class="header">
        <h1>新旧AWB效果对比报告</h1>
        <p>{CAMERA_DISPLAY["old_isp"]} vs {CAMERA_DISPLAY["new_isp"]}</p>
      </div>
      <div class="intro">
        <h2>报告说明</h2>
        <p>本报告对比 {CAMERA_DISPLAY["old_isp"]} 与 {CAMERA_DISPLAY["new_isp"]} 的色彩还原表现. 评估基于 ColorChecker 色卡, 将每个色块的测量颜色与标准参考颜色对比.</p>
        <p>核心指标为 CIEDE2000(DeltaE/E00). CIEDE2000 是业界常用的颜色差异评价标准, 更贴近人眼感知. 本报告所有指标均为数值越小越好.</p>
        <ul>
          <li><b>DeltaE</b>: 综合色差, 衡量整体颜色与标准色的差异.</li>
          <li><b>DeltaC</b>: 饱和度(Chroma)误差, 衡量饱和度偏差.</li>
          <li><b>DeltaL</b>: 亮度(Lightness)误差, 衡量亮度偏差.</li>
        </ul>
        <p>第一页为总体汇总(所有场景平均). 第二页为分场景柱状对比. 后续为每个场景的可视化结果.</p>
      </div>
      <div class="plot">
        <h2>总体汇总(综合色差/饱和度误差/亮度误差，数值越小越好)</h2>
        <img src="summary_plots/summary.png" alt="summary">
      </div>
    </div>
    <div class="second-page">
      <div class="plot">
        <h2>各场景 DeltaE 对比(综合色差，数值越小越好)</h2>
        <img src="summary_plots/delta_e.png" alt="delta_e">
      </div>
      <div class="plot">
        <h2>各场景 DeltaC 对比(饱和度误差，数值越小越好)</h2>
        <img src="summary_plots/delta_c.png" alt="delta_c">
      </div>
      <div class="plot">
        <h2>各场景 DeltaL 对比(亮度误差，数值越小越好)</h2>
        <img src="summary_plots/delta_l.png" alt="delta_l">
      </div>
    </div>
    <div class="scene-gallery">
"""

    for scene in scenes:
        html += f'<div class="scene"><h3>{scene["folder_name"]}</h3>'
        for camera in CAMERAS:
            html += f'''
  <div class="camera-img">
    <div class="camera-label">{CAMERA_DISPLAY[camera]}</div>
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
    run_new_isp()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/summary_plots", exist_ok=True)

    scenes = get_scene_data()
    avg_values = calculate_averages(scenes)

    plot_bars(avg_values, scenes, "E", "DeltaE", f"{OUTPUT_DIR}/summary_plots/delta_e.png")
    plot_bars(avg_values, scenes, "C", "DeltaC", f"{OUTPUT_DIR}/summary_plots/delta_c.png")
    plot_bars(avg_values, scenes, "L", "DeltaL", f"{OUTPUT_DIR}/summary_plots/delta_l.png")
    plot_summary_table(avg_values, f"{OUTPUT_DIR}/summary_plots/summary.png")

    generate_html(scenes)
    print("done:", OUTPUT_DIR + "/index.html")


if __name__ == "__main__":
    main()


