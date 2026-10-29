"""
説明資料と動画用の図（report/ に保存）。

  python make_figures.py
- robot.jpg：犬型ロボットの外観（斜め前と真横）
- curriculum.png：学習中の地形ごとの難しさ（カリキュラム）の推移
- terrain_eval.png：地形の種類×難しさごとの成功率
- house_map.png：家と庭の間取り図と、3 つの評価経路の計画結果
"""
import json
import os
import re

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")
RUN = os.path.join(HERE, "runs", "terrain_dog")
# 講義資料の配色
INK, YELLOW, LIGHT, SLATE, RED, GRAY = "#141414", "#ffd966", "#fff2cc", "#44546a", "#c00000", "#a6a6a6"
TERRAIN_JA = {"flat": "平地", "rough": "凸凹", "hills": "丘（坂）", "stairs_up": "上り階段", "stairs_down": "下り階段"}
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]


def curriculum():
    steps, levels = [], []
    for line in open(os.path.join(HERE, "train_terrain.log"), encoding="utf-8"):
        m = re.search(r"step=(\d+) .*level (.*)$", line)
        if m:
            steps.append(int(m.group(1)) / 1e6)
            levels.append(dict(kv.split("=") for kv in m.group(2).split()))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    styles = {"flat": (GRAY, "-"), "rough": (SLATE, "--"), "hills": (SLATE, "-"), "stairs_up": (RED, "-"), "stairs_down": (RED, "--")}
    for k, (c, ls) in styles.items():
        ax.plot(steps, [float(lv[k]) for lv in levels], ls, color=c, lw=2, label=TERRAIN_JA[k])
    ax.set_xlabel("学習ステップ数 [百万]")
    ax.set_ylabel("難しさ（0〜1）")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("地形ごとの難しさの推移（直近 100 エピソードの平均）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curriculum.png"), dpi=150)


def terrain_eval():
    ev = json.load(open(os.path.join(RUN, "eval_final.json")))
    lvs = list(ev)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    kinds = list(TERRAIN_JA)
    colors = [GRAY, LIGHT, YELLOW, RED, SLATE]
    w = 0.16
    for i, k in enumerate(kinds):
        xs = np.arange(len(lvs)) + (i - 2) * w
        ax.bar(xs, [100 * ev[lv][k]["success"] for lv in lvs], w, color=colors[i], edgecolor=INK, lw=0.6, label=TERRAIN_JA[k])
    ax.set_xticks(np.arange(len(lvs)))
    ax.set_xticklabels([f"難しさ {lv}\n段差 {2 + 8 * float(lv):.0f} cm" for lv in lvs])
    ax.set_ylabel("成功率 [%]")
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc="upper center")
    ax.set_title("地形の種類×難しさごとの成功率（各 30 回）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "terrain_eval.png"), dpi=150)


def robot():
    from house import HouseDogEnv

    env = HouseDogEnv(render_mode="rgb_array", width=960, height=600, camera_id=-1)
    env.reset(seed=0)
    env.set_goal([30.0, 30.0])  # ゴール旗は写さない
    for _ in range(int(1.0 / env.dt)):
        env.step(np.zeros(env.action_space.shape))
    env.render()
    v = env.mujoco_renderer._get_viewer("rgb_array")
    tiles = []
    for az, el in [(130, -18), (180, -6)]:
        v.cam.lookat[:] = [*env.xy(), 0.22]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.4, az, el
        tiles.append(Image.fromarray(env.render()))
    img = Image.new("RGB", (960 * 2 + 20, 600), "white")
    for i, t in enumerate(tiles):
        img.paste(t, (980 * i, 0))
    img.save(os.path.join(OUT, "robot.jpg"), quality=90)


def house_map():
    from evaluate_terrain import HOUSE_ROUTES
    from house import HOME, PLACES, plan_path
    from house_demo import NAMES, house_map as draw

    img = draw(900, [], [], None, [])
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = -7.0, 7.0, -7.2, 8.6
    w, h = img.size

    def px(x, y):
        return ((x - x0) / (x1 - x0) * w, (y1 - y) / (y1 - y0) * h)

    styles = [((192, 0, 0, 255), 0), ((68, 84, 106, 255), 1), ((120, 120, 120, 255), 2)]
    for route, (col, k) in zip(HOUSE_ROUTES, styles):
        cur = np.array(PLACES[HOME][1])
        pts = [cur]
        for name in route:
            pts += plan_path(cur, PLACES[name][1])
            cur = np.array(PLACES[name][1])
        off = (k - 1) * 4  # 重なりを見やすく少しずらす
        d.line([(px(*p)[0] + off, px(*p)[1] + off) for p in pts], fill=col, width=5)
    from overlay import font

    for key, (_, (x, y)) in PLACES.items():
        cx, cy = px(x, y)
        d.text((cx, cy + 22), NAMES[key], font=font(24, "bold"), fill=(20, 20, 20, 255), anchor="mt",
               stroke_width=3, stroke_fill=(255, 255, 255, 255))
    bg = Image.new("RGB", img.size, "white")
    bg.paste(img, mask=img.split()[3])
    bg.save(os.path.join(OUT, "house_map.png"))


if __name__ == "__main__":
    curriculum()
    robot()
    house_map()
    if os.path.exists(os.path.join(RUN, "eval_final.json")):
        terrain_eval()
