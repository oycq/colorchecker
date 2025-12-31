import os
#防止多线程冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from colorchecker_analyze import summarize_colorchecker

# 配置
BASE_PATH = "./scene"
OUTPUT_DIR = "./output"
CAMERAS = ["iPhone", "DE", "Xiaomi"]
CAMERA_PATTERNS = {
    "iPhone": "iphone.jpg",
    "DE": "*isp.png", 
    "Xiaomi": "xiaomi.jpg"
}

# 🎨 友商蓝绿 + 自家灰
COLORS = {
    "iPhone": "#2E86AB",   # 💙 友商iPhone
    "DE": "#6C757D",       # ⚫ 自家DE
    "Xiaomi": "#58A14B"    # 🟢 友商Xiaomi
}

# 设置中文字体
plt.rcParams.update({
    'font.sans-serif': ['SimHei'],
    'axes.unicode_minus': False
})

# ✅ 统一尺寸！1730宽 × 500高
FIG_SIZE = (17.3, 5.0)
BAR_WIDTH = 0.25
BAR_OFFSETS = [-0.25, 0, 0.25]
BAR_SPACING = BAR_WIDTH * 2      # 3bar内：0.5间距
SCENE_SPACING = BAR_WIDTH * 1.2  # ✅ 场景间：1.2个bar=0.3间距

def get_scene_data():
    """获取所有场景数据 + 保存merged_img (1730x500)"""
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
                merged_img, avg_e, avg_c, avg_l = summarize_colorchecker(path)
                
                # ✅ BGR → RGB
                merged_img_rgb = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)
                
                # 💾 保存1730x500 merged_img
                img_path = f"{OUTPUT_DIR}/merged_images/{scene_folder}/{camera}.png"
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                plt.imsave(img_path, merged_img_rgb)
                merged_imgs[camera].append(img_path)
                
                scene_data[camera].append((avg_e, avg_c, avg_l))
        
        scenes.append({
            "folder_name": scene_folder, 
            "data": scene_data, 
            "images": merged_imgs
        })
    return scenes

def calculate_averages(scenes):
    """计算各相机各指标的平均值"""
    avg_values = {camera: {"E": [], "C": [], "L": []} for camera in CAMERAS}
    for scene in scenes:
        for camera in CAMERAS:
            if scene["data"][camera]:
                e_vals, c_vals, l_vals = zip(*scene["data"][camera])
                avg_values[camera]["E"].append(np.mean(e_vals))
                avg_values[camera]["C"].append(np.mean(c_vals))
                avg_values[camera]["L"].append(np.mean(l_vals))
    return avg_values

def plot_bars(avg_values, scenes, metric, title, ylabel, save_path):
    """绘制柱状图 + 保存 (1730x500) + ✅ 场景间1.2bar距离！"""
    # ✅ 精确计算：3bar(0.75) + 3bar间距(0.5) + 1.2bar场景间距(0.3) = 1.55
    x = np.arange(len(scenes)) * (BAR_WIDTH * 3 + BAR_SPACING + SCENE_SPACING)
    scene_names = [scene["folder_name"] for scene in scenes]
    
    plt.figure(figsize=FIG_SIZE)
    for i, camera in enumerate(CAMERAS):
        values = avg_values[camera][metric]
        bars = plt.bar(x + BAR_OFFSETS[i], values, 
                      width=BAR_WIDTH, label=camera, align='center', color=COLORS[camera])
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    plt.xticks(x, scene_names, rotation=0, ha='center', fontsize=11)
    plt.xlabel("场景", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(0, 12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_summary_table(avg_values, save_path):
    """绘制1×3汇总表 + 保存 (1730x500)"""
    summary_data = {}
    for camera in CAMERAS:
        summary_data[camera] = {
            "E": np.mean(avg_values[camera]["E"]),
            "C": np.mean(avg_values[camera]["C"]),
            "L": np.mean(avg_values[camera]["L"])
        }
    
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    metrics = ["E", "C", "L"]
    titles = ["ΔE (色差)", "ΔC (饱和度误差)", "ΔL (亮度误差)"]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = [summary_data[camera][metric] for camera in CAMERAS]
        bars = ax.bar(CAMERAS, values, width=0.8, color=[COLORS[camera] for camera in CAMERAS])
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_ylabel(f"{metric}值", fontsize=12)
        ax.set_ylim(0, 10)
        ax.tick_params(axis='x', labelsize=12)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            y_offset = max(height * 0.05, 0.3)
            ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_html(scenes, avg_values):
    """生成HTML报告 - 无边框大图 + 真每页3个！"""
    num_scenes = len(scenes)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>相机色差对比报告</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial; margin: 20px; }}
            .container {{ max-width: 80%; margin: auto; }}
            .header {{ text-align: center; margin-bottom: 40px; page-break-after: avoid; }}
            .plot {{ text-align: center; margin: 30px 0; page-break-inside: avoid; }}
            .plot img {{ 
                width: 100%; 
                height: auto; 
                max-width: 100%; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
            }}
            .scene-gallery {{ display: grid; grid-template-columns: 1fr; gap: 60px; margin-top: 40px; }}
            .scene {{ 
                padding: 0; 
                margin-bottom: 40px; 
                page-break-inside: avoid; 
            }}
            .scene h3 {{ 
                margin: 0 0 20px 0; 
                color: #333; 
                width: 100%; 
                font-size: 20px; 
                text-align: center; 
            }}
            .camera-img {{ margin: 10px; text-align: center; }}
            .scene img {{ 
                width: 100%; 
                height: auto; 
                max-width: 100%; 
                object-fit: contain; 
                border-radius: 4px; 
                display: block; 
                margin: 5px auto; 
            }}
            .camera-label {{ 
                font-weight: bold; 
                color: #333; 
                margin: 10px 0 5px 0; 
                font-size: 18px; 
                text-align: center; 
            }}
            @media print {{
                .scene-gallery > *:nth-child(3n) {{ margin-bottom: 0; }}
                .scene-gallery > *:nth-child(3n+1):not(:first-child) {{ page-break-before: always; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 相机色差对比报告</h1>
                <p>DE vs iPhone vs Xiaomi | {num_scenes} 场景分析</p>
            </div>
    """
    
    # 1️⃣ 首先：4个plot（第1页）
    html_content += """
            <div class="plot">
                <h2>📈 总体性能对比（所有场景平均）</h2>
                <img src="summary_plots/summary.png" alt="总体性能对比">
            </div>
    """
    
    # 2️⃣ 然后：E C L场景对比
    PLOTS = [
        ("delta_e.png", "ΔE (色差) - 各场景对比"),
        ("delta_c.png", "ΔC (饱和度误差) - 各场景对比"),
        ("delta_l.png", "ΔL (亮度误差) - 各场景对比")
    ]
    
    for img_file, title in PLOTS:
        html_content += f"""
            <div class="plot">
                <h2>{title}</h2>
                <img src="summary_plots/{img_file}" alt="{title}">
            </div>
        """
    
    # 3️⃣ 最后：所有merged_img（无边框大图！）
    html_content += '<div class="scene-gallery">'
    for scene in scenes:
        html_content += f'<div class="scene">'
        html_content += f'<h3>🎬 {scene["folder_name"]}</h3>'
        for camera in CAMERAS:
            html_content += f'''
                <div class="camera-img">
                    <div class="camera-label">{camera}</div>
                    <img src="merged_images/{scene["folder_name"]}/{camera}.png" alt="{camera}">
                </div>
            '''
        html_content += '</div>'
    html_content += '</div>'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(f"{OUTPUT_DIR}/index.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

# 🚀 主程序
if __name__ == "__main__":
    print("🚀 开始生成报告...")
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/summary_plots", exist_ok=True)
    
    # 1. 获取数据 + 保存图片
    scenes = get_scene_data()
    print(f"✅ 数据采集完成：{len(scenes)}个场景")
    
    # 2. 计算平均值
    avg_values = calculate_averages(scenes)
    
    # 3. 生成所有图表
    PLOTS = [
        ("ΔE (色差)", "ΔE值", f"{OUTPUT_DIR}/summary_plots/delta_e.png"),
        ("ΔC (饱和度误差)", "ΔC值", f"{OUTPUT_DIR}/summary_plots/delta_c.png"),
        ("ΔL (亮度误差)", "ΔL值", f"{OUTPUT_DIR}/summary_plots/delta_l.png")
    ]
    
    for title, ylabel, save_path in PLOTS:
        metric = title.split()[0][1]
        plot_bars(avg_values, scenes, metric, title, ylabel, save_path)
    
    plot_summary_table(avg_values, f"{OUTPUT_DIR}/summary_plots/summary.png")
    print("✅ 图表生成完成")
    
    # 4. 生成HTML
    generate_html(scenes, avg_values)
    print("✅ HTML报告生成完成")
    
    print(f"\n🎉 报告完成！")
    print(f"📁 文件夹: {OUTPUT_DIR}")
    print(f"🌐 打开: {OUTPUT_DIR}/index.html")
    print(f"📊 包含: {len(scenes)}场景 × 3相机 = {len(scenes)*3}张merged_img")
    print(f"🖼️  统一尺寸: **1730×500像素**")
    print(f"🖨️  **PDF分页：第1页4个plot + 每页3个场景（9张大图！）**")
    print(f"🎨 **20%页边距** + **3bar 0.5间距** + **场景间1.2bar距离(0.3)** + **无边框大图**")